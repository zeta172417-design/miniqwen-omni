#!/usr/bin/env python3
"""Prepare the fixed English standard manifest from ModelScope datasets."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import normalize_text


def stable_rows(dataset, count: int, seed: int):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return [dataset[index] for index in indices[:count]]


def duration_seconds(path: str | Path) -> float:
    import soundfile as sf
    return float(sf.info(path).duration)


def mcq_prompt(question: str, choices: list[str], context: str = "") -> str:
    body = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    prefix = f"{context.strip()}\n\n" if context and context != "nan" else ""
    return f"{prefix}{question.strip()}\n{body}\nAnswer with only the option letter."


def save_image(value: Any, path: Path) -> str:
    from PIL import Image
    import io
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Image.Image):
        value.convert("RGB").save(path, quality=92)
    elif isinstance(value, dict) and value.get("bytes"):
        Image.open(io.BytesIO(value["bytes"])).convert("RGB").save(path, quality=92)
    elif isinstance(value, dict) and value.get("path"):
        Image.open(value["path"]).convert("RGB").save(path, quality=92)
    elif isinstance(value, (str, Path)):
        Image.open(value).convert("RGB").save(path, quality=92)
    else:
        raise ValueError(f"Unsupported image value: {type(value)}")
    return str(path)


def synthesize(text: str, path: Path, voice: str = "slt") -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = hashlib.sha256(f"flite-v1\0{voice}\0{text}".encode()).hexdigest()
    signature_path = path.with_suffix(path.suffix + ".sha256")
    if (
        path.is_file()
        and path.stat().st_size > 44
        and signature_path.is_file()
        and signature_path.read_text().strip() == signature
    ):
        return str(path)
    text_path = path.with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")
    filter_value = f"flite=textfile='{text_path}':voice={voice}"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", filter_value, "-ar", "16000", "-ac", "1", str(path)],
        check=True,
    )
    text_path.unlink()
    signature_path.write_text(signature + "\n", encoding="utf-8")
    return str(path)


def save_audio(value: Any, path: Path) -> str:
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict) and "array" in value:
        sf.write(path, np.asarray(value["array"]), int(value["sampling_rate"]))
    elif isinstance(value, dict) and value.get("bytes"):
        path.write_bytes(value["bytes"])
    elif isinstance(value, dict) and value.get("path"):
        import shutil
        shutil.copyfile(value["path"], path)
    elif isinstance(value, (str, Path)):
        import shutil
        shutil.copyfile(value, path)
    else:
        raise ValueError(f"Unsupported audio value: {type(value)}")
    return str(path)


def add_noise(source: str, destination: Path, seed: int, snr_db: float = 10.0) -> str:
    import numpy as np
    import soundfile as sf
    source_digest = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    signature = hashlib.sha256(f"noise-v1\0{source_digest}\0{seed}\0{snr_db}".encode()).hexdigest()
    signature_path = destination.with_suffix(destination.suffix + ".sha256")
    if (
        destination.is_file()
        and destination.stat().st_size > 44
        and signature_path.is_file()
        and signature_path.read_text().strip() == signature
    ):
        return str(destination)
    audio, sr = sf.read(source, always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=audio.shape).astype(np.float32)
    signal_power = float(np.mean(audio ** 2)) + 1e-12
    noise *= (signal_power / (10 ** (snr_db / 10)) / (float(np.mean(noise ** 2)) + 1e-12)) ** 0.5
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, np.clip(audio + noise, -1, 1), sr)
    signature_path.write_text(signature + "\n", encoding="utf-8")
    return str(destination)


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode()).hexdigest()


def training_hashes(paths: list[str]) -> set[str]:
    hashes = set()
    try:
        import pyarrow.parquet as pq
        for raw_path in paths:
            path = Path(raw_path)
            if not path.exists():
                continue
            parquet = pq.ParquetFile(path)
            # Reading the audio/image payload columns makes this scan orders of
            # magnitude slower and consumes unnecessary memory. Decontamination
            # only depends on the serialized conversation text.
            for batch in parquet.iter_batches(batch_size=2048, columns=["conversations"]):
                for row in batch.to_pylist():
                    conversations = row.get("conversations") or row.get("conversation") or row.get("messages") or []
                    if isinstance(conversations, str):
                        try: conversations = json.loads(conversations)
                        except Exception: conversations = []
                    for turn in conversations:
                        if isinstance(turn, dict) and turn.get("role") in {"user", "human"}:
                            hashes.add(content_hash(str(turn.get("content") or turn.get("value") or "")))
    except Exception as exc:
        print(f"warning: training decontamination scan failed: {exc}", file=sys.stderr)
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "benchmark/configs/standard.json"))
    parser.add_argument("--output", default="/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/standard")
    parser.add_argument("--cache", default="/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/modelscope-cache")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-decontamination", action="store_true")
    parser.add_argument("--training-parquet", action="append", default=[])
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    counts, seed = config["counts"], int(config["seed"])
    output, media = Path(args.output).resolve(), Path(args.output).resolve() / "media"
    output.mkdir(parents=True, exist_ok=True)
    from modelscope.msdatasets import MsDataset

    def load(name, split, subset=None):
        return MsDataset.load(name, subset_name=subset, split=split, cache_dir=args.cache)

    rows = []
    arc = stable_rows(load(config["sources"]["arc_easy"], "test", "ARC-Easy"), counts["arc_easy"], seed)
    for i, row in enumerate(arc):
        choices = list(row["choices"]["text"])
        rows.append({"sample_id": f"arc-easy-{i:04d}", "task": "text", "metric": "mcq", "prompt": mcq_prompt(row["question"], choices), "reference": row["answerKey"], "choices": choices, "source": "allenai/ai2_arc:ARC-Easy:test"})
    boolq = stable_rows(load(config["sources"]["boolq"], "validation"), counts["boolq"], seed + 1)
    for i, row in enumerate(boolq):
        choices = ["Yes", "No"]
        rows.append({"sample_id": f"boolq-{i:04d}", "task": "text", "metric": "mcq", "prompt": mcq_prompt(row["question"], choices, row["passage"]), "reference": "A" if row["answer"] else "B", "choices": choices, "source": "google/boolq:validation"})
    piqa = stable_rows(load(config["sources"]["piqa"], "validation"), counts["piqa"], seed + 2)
    for i, row in enumerate(piqa):
        choices = list(row["choices"])
        rows.append({"sample_id": f"piqa-{i:04d}", "task": "text", "metric": "mcq", "prompt": mcq_prompt(row["question"], choices), "reference": row.get("answer", chr(65 + int(row["answer_index"]))), "choices": choices, "source": "extraordinarylab/piqa:validation"})

    spoken_candidates = [row for row in rows if row["metric"] == "mcq" and len(row["prompt"].split()) <= 70]
    spoken_pool = []
    for base in spoken_candidates:
        i = len(spoken_pool)
        audio = synthesize(base["prompt"], media / "spoken_mcq" / f"{i:04d}.wav", "slt" if i % 2 == 0 else "rms")
        if duration_seconds(audio) > 30.0:
            continue
        spoken_pool.append((base, audio))
        if len(spoken_pool) == counts["spoken_mcq"]:
            break
    if len(spoken_pool) < counts["spoken_mcq"]:
        raise RuntimeError(f"Only {len(spoken_pool)} spoken MCQs fit the 30-second audio budget")
    for i, (base, audio) in enumerate(spoken_pool):
        rows.append({**base, "sample_id": f"spoken-{i:04d}", "task": "audio", "prompt": "", "audio": audio, "metadata": {"transcript_prompt": base["prompt"], "require_audio": True}, "source": base["source"] + "+flite"})

    if not args.skip_audio:
        for subset, key, offset in (("clean", "librispeech_clean", 10), ("other", "librispeech_other", 11)):
            dataset = load(config["sources"]["librispeech"], "test", subset)
            indices = list(range(len(dataset)))
            random.Random(seed + offset).shuffle(indices)
            accepted = 0
            for index in indices:
                row = dataset[index]
                audio_value = row.get("audio") or row.get("file")
                path = save_audio(audio_value, media / "librispeech" / subset / f"{accepted:04d}.wav")
                if duration_seconds(path) > 30.0:
                    continue
                reference = row.get("text") or row.get("transcription") or row.get("sentence")
                rows.append({"sample_id": f"librispeech-{subset}-{accepted:04d}", "task": "audio", "metric": "wer", "prompt": "Transcribe the speech exactly in English.", "reference": reference, "audio": path, "metadata": {"require_audio": False}, "source": f"openslr/librispeech_asr:{subset}:test"})
                accepted += 1
                if accepted == counts[key]:
                    break
            if accepted < counts[key]:
                raise RuntimeError(f"LibriSpeech {subset} only supplied {accepted} samples within 30 seconds")

    visual_specs = [
        (config["sources"]["mmbench_en"], "dev", counts["mmbench_en"], "mmbench", seed + 20),
        (config["sources"]["pope"], "test", counts["pope"], "pope", seed + 21),
        (config["sources"]["textvqa"], "train", counts["textvqa"], "textvqa", seed + 22),
    ]
    for dataset_id, split, count, kind, local_seed in visual_specs:
        if count <= 0:
            continue
        dataset = load(dataset_id, split)
        if kind == "pope":
            flattened = []
            # Each POPE row owns one image and six questions in each sampling strategy.
            for source_row in dataset:
                for item in source_row.get("adversarial", []):
                    flattened.append({"image": source_row["image"], **item})
            candidates = stable_rows(flattened, len(flattened), local_seed)
        elif kind == "mmbench":
            eligible = []
            for source_row in dataset:
                candidate_choices = [str(source_row.get(letter)) for letter in "ABCD" if str(source_row.get(letter, "nan")) != "nan"]
                candidate_prompt = mcq_prompt(str(source_row["question"]), candidate_choices, str(source_row.get("hint", "")))
                if len(candidate_prompt.split()) <= 70:
                    eligible.append(source_row)
            candidates = stable_rows(eligible, len(eligible), local_seed)
        else:
            candidates = stable_rows(dataset, len(dataset), local_seed)
        accepted = 0
        for row in candidates:
            if kind == "mmbench":
                choices = [str(row.get(letter)) for letter in "ABCD" if str(row.get(letter, "nan")) != "nan"]
                prompt = mcq_prompt(str(row["question"]), choices, str(row.get("hint", "")))
                reference, metric = row["answer"], "mcq"
            elif kind == "pope":
                choices = ["Yes", "No"]
                prompt = mcq_prompt(str(row.get("question") or row.get("text")), choices)
                raw = str(row.get("answer") or row.get("label")).lower()
                reference, metric = ("A" if raw in {"yes", "1", "true"} else "B"), "mcq"
            else:
                choices = []
                prompt = str(row.get("question")) + "\nAnswer using a short phrase only."
                reference, metric = row.get("answers") or row.get("answer"), "vqa"
            question_audio = synthesize(prompt, media / kind / f"{accepted:04d}.wav", "slt" if accepted % 2 == 0 else "rms")
            if duration_seconds(question_audio) > 30.0:
                continue
            image_value = row.get("image") or row.get("image_path")
            image = save_image(image_value, media / kind / f"{accepted:04d}.jpg")
            rows.append({"sample_id": f"{kind}-{accepted:04d}", "task": "audio_vision" if kind == "pope" else "vision", "metric": metric, "prompt": "", "reference": reference, "choices": choices, "image": image, "audio": question_audio, "metadata": {"transcript_prompt": prompt, "require_audio": True}, "source": f"{dataset_id}:{split}"})
            accepted += 1
            if accepted == count:
                break
        if accepted < count:
            raise RuntimeError(f"{kind} only supplied {accepted} of {count} samples within 30 seconds")

    # Fixed programmatic instruction following contributes to the text category.
    for i in range(counts["instruction"]):
        color = ["red", "blue", "green", "yellow"][i % 4]
        rows.append({"sample_id": f"instruction-{i:04d}", "task": "text", "metric": "constraint", "prompt": f"Reply in at most five words and include the word {color}.", "reference": {"max_words": 5, "must_include": [color]}, "source": "generated:instruction-v1"})

    # Gaussian-noise preservation set, paired to the first spoken MCQs.
    for i, (base, _) in enumerate(spoken_pool[: counts["robustness"]]):
        clean = media / "spoken_mcq" / f"{i:04d}.wav"
        noisy = add_noise(str(clean), media / "robustness" / f"{i:04d}-snr10.wav", seed + i)
        rows.append({**base, "sample_id": f"robust-audio-{i:04d}", "task": "robustness", "prompt": "", "audio": noisy, "perturbation": "white-noise-10db", "metadata": {"base_id": f"spoken-{i:04d}", "transcript_prompt": base["prompt"], "require_audio": True}, "source": base["source"] + "+noise10db"})

    defaults = {"choices": [], "image": None, "audio": None, "perturbation": "clean", "metadata": {}}
    training_paths = args.training_parquet or [str(PROJECT_ROOT / "dataset" / name) for name in ("sft_t2a.parquet", "sft_a2a.parquet", "sft_i2t.parquet")]
    seen = set() if args.skip_decontamination else training_hashes(training_paths)
    clean_rows, rejected = [], []
    for row in rows:
        complete = {**defaults, **row}
        complete["metadata"] = dict(complete.get("metadata") or {})
        complete["metadata"].setdefault("require_audio", complete["metric"] != "wer")
        decontamination_text = complete["metadata"].get("transcript_prompt", complete["prompt"])
        if seen and content_hash(decontamination_text) in seen:
            rejected.append(complete["sample_id"])
        else:
            clean_rows.append(complete)
    manifest = output / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean_rows), encoding="utf-8")
    source_counts = Counter(row["source"] for row in clean_rows)
    metadata = {
        "name": config["name"], "seed": seed, "samples": len(clean_rows),
        "source_counts": source_counts, "rejected_training_overlap": rejected,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "tts": {"engine": "FFmpeg flite", "voices": ["slt", "rms"], "sample_rate": 16000},
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
