#!/usr/bin/env python3
"""Generate the shared fixed-text voice-cloning set with one supported model."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.run_benchmark import atomic_json, path_fingerprint


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def model_entry(config: dict, model_id: str) -> dict:
    for entry in config["models"]:
        if entry["id"] == model_id:
            return entry
    raise KeyError(f"Unknown model id: {model_id}")


def load_voice_model(entry: dict, device: str):
    import torch
    from transformers import AutoConfig

    checkpoint = entry["kwargs"]["checkpoint"]
    config = AutoConfig.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    if config.model_type == "minimind-o":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            checkpoint, trust_remote_code=True, local_files_only=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            local_files_only=True,
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            audio_encoder_path=None,
            vision_model_path=None,
        )
        generation_token_name = "enter_token_id"
        template_kwargs = {"open_thinking": False}
    else:
        from trainer.trainer_utils import (
            configure_token_ids,
            get_omni_model_class,
            infer_omni_model_arch,
            load_omni_tokenizer,
        )

        model_class = get_omni_model_class(infer_omni_model_arch(checkpoint))
        tokenizer = load_omni_tokenizer(checkpoint)
        model = model_class.from_pretrained(
            checkpoint, dtype=torch.bfloat16, audio_encoder_path=None, vision_model_path=None,
        )
        configure_token_ids(model.config, tokenizer)
        generation_token_name = "newline_token_id"
        template_kwargs = {"enable_thinking": False}
    return model.eval().to(device), tokenizer, generation_token_name, template_kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "benchmark/configs/models.json"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mimi", default=str(PROJECT_ROOT / "model/mimi"))
    parser.add_argument("--campplus", default=str(PROJECT_ROOT / "model/campplus"))
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    entry = model_entry(config, args.model)
    output_dir = Path(args.run_dir).resolve() / "voice_clone" / args.model
    audio_dir = output_dir / "audio"
    if args.num_shards > 1:
        audio_dir = audio_dir / f"shard-{args.shard_index:02d}"
    audio_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f".shard-{args.shard_index:02d}-of-{args.num_shards:02d}" if args.num_shards > 1 else ""
    output_jsonl = output_dir / f"per_sample{shard_suffix}.jsonl"
    metadata_path = output_dir / f"run_metadata{shard_suffix}.json"
    fingerprint_payload = {
        "manifest_sha256": hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
        "checkpoint_fingerprint": path_fingerprint(entry["kwargs"].get("checkpoint")),
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "text_decoding": "greedy", "max_new_tokens": args.max_new_tokens,
    }
    run_fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    if args.resume and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError("refusing stale voice-clone resume: manifest/model/settings changed")
    elif args.resume and output_jsonl.exists():
        raise RuntimeError(f"refusing voice-clone resume without metadata: {metadata_path}")
    atomic_json(metadata_path, {"run_fingerprint": run_fingerprint, **fingerprint_payload})
    samples = [
        (index, sample) for index, sample in enumerate(read_jsonl(Path(args.manifest)))
        if index % args.num_shards == args.shard_index
    ]
    existing_by_id = {}
    if args.resume:
        for candidate in (output_jsonl, output_dir / "per_sample.jsonl"):
            if candidate.exists():
                existing_by_id.update({row["sample_id"]: row for row in read_jsonl(candidate)})
    seeded_rows = []
    for benchmark_index, sample in samples:
        row = existing_by_id.get(sample["sample_id"])
        if row is not None:
            row = dict(row)
            row["benchmark_index"] = benchmark_index
            seeded_rows.append(row)
    if seeded_rows:
        output_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeded_rows),
            encoding="utf-8",
        )
    completed = {row["sample_id"] for row in seeded_rows}
    if len(completed) == len(samples):
        output_jsonl.touch(exist_ok=True)
        print(
            f"Voice-clone shard already complete: {args.model} "
            f"{args.shard_index + 1}/{args.num_shards} ({len(samples)} samples)",
            flush=True,
        )
        return

    import librosa
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    from funasr import AutoModel
    from transformers import MimiModel

    model, tokenizer, generation_token_name, template_kwargs = load_voice_model(entry, args.device)
    mimi = MimiModel.from_pretrained(args.mimi, local_files_only=True).eval().to(args.device)
    campplus = AutoModel(
        model=args.campplus, device="cpu", disable_update=True,
        log_level="ERROR", ncpu=2,
    )
    newline_ids = tokenizer.encode("\n", add_special_tokens=False)
    if len(newline_ids) != 1:
        raise ValueError(f"newline must encode to one token, got {newline_ids}")

    voice_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def voice_condition(sample: dict) -> tuple[torch.Tensor, torch.Tensor]:
        speaker = sample["speaker_id"]
        if speaker in voice_cache:
            return voice_cache[speaker]
        ref_audio = sample["ref_audio"]
        raw = campplus.generate(input=ref_audio, disable_pbar=True, disable_log=True)[0]["spk_embedding"][0]
        # Training data stores per-utterance normalized CAMPPlus vectors.
        spk_emb = F.layer_norm(raw.float(), raw.shape).to(
            device=args.device, dtype=next(model.talker.spk_proj.parameters()).dtype,
        ).unsqueeze(0)
        waveform, _ = librosa.load(ref_audio, sr=24000, mono=True)
        input_values = torch.tensor(waveform, dtype=torch.float32, device=args.device)[None, None, :]
        with torch.inference_mode():
            ref_codes = mimi.encode(input_values, num_quantizers=8).audio_codes.long()
        voice_cache[speaker] = (ref_codes, spk_emb)
        return ref_codes, spk_emb

    mode = "a" if seeded_rows else "w"
    with output_jsonl.open(mode, encoding="utf-8") as handle:
        for shard_position, (benchmark_index, sample) in enumerate(samples, 1):
            if sample["sample_id"] in completed:
                continue
            torch.manual_seed(20260914)
            if args.device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            status, error = "ok", None
            generated_text, generated_audio = "", None
            try:
                ref_codes, spk_emb = voice_condition(sample)
                messages = [
                    {"role": "system", "content": "You are a text-to-speech engine. Repeat the requested text exactly without adding commentary."},
                    {"role": "user", "content": f"Repeat exactly:\n{sample['target_text']}"},
                ]
                rendered = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, **template_kwargs,
                )
                input_ids = torch.tensor(
                    tokenizer(rendered, add_special_tokens=False)["input_ids"],
                    dtype=torch.long, device=args.device,
                ).unsqueeze(0)
                audio_frames, final_ids = [], None
                with torch.inference_mode():
                    generation_kwargs = {
                        generation_token_name: newline_ids[0],
                        "pad_token_id": tokenizer.pad_token_id,
                    }
                    stream = model.generate(
                        input_ids, max_new_tokens=args.max_new_tokens, stream=True,
                        return_audio_codes=True, open_thinking=False,
                        temperature=1.0, top_k=1, top_p=1.0,
                        ref_codes=ref_codes, spk_emb=spk_emb,
                        **generation_kwargs,
                    )
                    for ids, frame in stream:
                        if ids is not None:
                            final_ids = ids
                        if frame and len(frame) == 8:
                            audio_frames.append(frame)
                if final_ids is not None:
                    generated_text = tokenizer.decode(final_ids[0], skip_special_tokens=True).strip()
                if audio_frames:
                    codes = torch.tensor(audio_frames, dtype=torch.long, device=args.device).T.unsqueeze(0)
                    codes = torch.where(codes < 2048, codes, torch.zeros_like(codes))
                    with torch.inference_mode():
                        waveform = mimi.decode(codes).audio_values.squeeze().float().cpu().numpy()
                    generated_audio = str(audio_dir / f"{sample['sample_id']}.wav")
                    sf.write(generated_audio, waveform, 24000)
                else:
                    status = "empty_audio"
            except Exception as exc:
                status, error = "error", f"{type(exc).__name__}: {exc}"
                logging.exception("voice clone sample failed: %s", sample["sample_id"])

            row = {
                **sample,
                "benchmark_index": benchmark_index,
                "model_id": args.model,
                "generated_text": generated_text,
                "generated_audio": generated_audio,
                "latency_seconds": time.perf_counter() - started,
                "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30 if args.device.startswith("cuda") else None,
                "status": status,
                "error": error,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if shard_position == 1 or shard_position % 8 == 0 or shard_position == len(samples) or status != "ok":
                print(
                    f"[{shard_position}/{len(samples)}] shard "
                    f"{args.shard_index + 1}/{args.num_shards} "
                    f"{args.model} {sample['sample_id']}: {status}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
