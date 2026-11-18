#!/usr/bin/env python3
"""Apply one shared ASR and optional UTMOS scorer to generated speech."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import (
    aggregate, character_error_rate, extract_spoken_choice, score_sample,
    word_edit_similarity, word_error_rate,
)


def load_asr(path: str, device: str = "cpu"):
    from funasr import AutoModel
    return AutoModel(model=path, trust_remote_code=True, disable_update=True, device=device, disable_log=True)


def transcribe(model, path: str) -> str:
    import soundfile as sf
    import numpy as np
    from scipy.signal import resample_poly
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = np.asarray(waveform).mean(axis=1)
    if sample_rate != 16000:
        waveform = resample_poly(waveform, 16000, sample_rate).astype("float32")
    result = model.generate(input=waveform, cache={}, language="en", use_itn=True, disable_pbar=True, disable_log=True)
    return rich_transcription_postprocess(result[0]["text"]).strip() if result else ""


def load_utmos(repo: str | None):
    if not repo:
        return None
    import torch
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*weight_norm.*deprecated.*", category=FutureWarning)
        if Path(repo).exists():
            return torch.hub.load(repo, "utmos22_strong", source="local").eval()
        return torch.hub.load(repo, "utmos22_strong", trust_repo=True).eval()


def score_utmos(model, path: str) -> float | None:
    if model is None:
        return None
    import numpy as np
    import soundfile as sf
    import torch
    waveform, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = np.asarray(waveform).mean(axis=1)
    with torch.inference_mode():
        return float(model(torch.from_numpy(waveform).unsqueeze(0), sample_rate).squeeze())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--asr", required=True)
    parser.add_argument("--device", default="cpu", help="SenseVoice device, for example cpu or cuda:0")
    parser.add_argument("--utmos-repo", default=None, help="Optional local SpeechMOS repo or torch.hub spec")
    parser.add_argument("--output", default=None)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--shard-label", default="")
    parser.add_argument("--reference-cache", default=None)
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output) if args.output else source.with_name("per_sample_scored.jsonl")
    asr, utmos = load_asr(args.asr, args.device), load_utmos(args.utmos_repo)
    reference_cache = Path(args.reference_cache).resolve() if args.reference_cache else None
    if reference_cache:
        reference_cache.mkdir(parents=True, exist_ok=True)

    def cached_transcript(path: str) -> str:
        if reference_cache is None:
            return transcribe(asr, path)
        source_path = Path(path).resolve()
        stat = source_path.stat()
        key = hashlib.sha256(f"{source_path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()
        destination_path = reference_cache / f"{key}.txt"
        if destination_path.is_file():
            return destination_path.read_text(encoding="utf-8").strip()
        value = transcribe(asr, path)
        temporary = destination_path.with_suffix(".tmp")
        temporary.write_text(value + "\n", encoding="utf-8")
        os.replace(temporary, destination_path)
        return value

    def cached_utmos(path: str) -> float | None:
        if utmos is None:
            return None
        if reference_cache is None:
            return score_utmos(utmos, path)
        source_path = Path(path).resolve()
        stat = source_path.stat()
        key = hashlib.sha256(f"{source_path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()).hexdigest()
        destination_path = reference_cache / f"{key}.utmos.txt"
        if destination_path.is_file():
            return float(destination_path.read_text(encoding="utf-8").strip())
        value = score_utmos(utmos, path)
        temporary = destination_path.with_suffix(".tmp")
        temporary.write_text(f"{value}\n", encoding="utf-8")
        os.replace(temporary, destination_path)
        return value
    rows = []
    source_lines = [line for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Write each completed row immediately so the parent runner can report one
    # concise aggregate progress line across all scoring shards.
    with destination.open("w", encoding="utf-8") as output_handle:
        for index, line in enumerate(source_lines, 1):
            row = json.loads(line)
            audio = row.get("generated_audio")
            if audio and Path(audio).is_file():
                transcript = transcribe(asr, audio)
                row["generated_audio_text"] = transcript
                if transcript.strip() and str(row.get("generated_text", "")).strip():
                    consistency = word_edit_similarity(transcript, row["generated_text"])
                else:
                    consistency = 0.0
                row.setdefault("scores", {})["speech_semantic"] = consistency
                if row.get("task") == "spoken_qa_a2a":
                    text_primary = float(row["scores"].get("primary", 0.0))
                    spoken_choice = extract_spoken_choice(transcript, row.get("choices", []))
                    spoken_result = {
                        **row,
                        "generated_text": spoken_choice or transcript,
                        "status": "ok",
                    }
                    spoken_scores = score_sample(spoken_result, spoken_result)
                    row["scores"].update(spoken_scores)
                    row["scores"]["text_primary"] = text_primary
                    row["scores"]["spoken_primary"] = float(spoken_scores.get("primary", 0.0))
                elif row.get("task") == "tts":
                    raw_wer = word_error_rate(transcript, row.get("reference"))
                    row["tts_wer"] = raw_wer
                    row["tts_cer"] = character_error_rate(transcript, row.get("reference"))
                    row["scores"]["primary"] = word_edit_similarity(transcript, row.get("reference"))
                    label_audio = row.get("metadata", {}).get("label_audio")
                    if label_audio and Path(label_audio).is_file():
                        label_text = cached_transcript(label_audio)
                        row["natural_label_audio_text"] = label_text
                        row["natural_label_wer"] = word_error_rate(label_text, row.get("reference"))
                        row["natural_label_utmos"] = cached_utmos(label_audio)
                        import soundfile as sf
                        label_duration = float(sf.info(label_audio).duration)
                        generated_duration = float(row.get("audio_diagnostics", {}).get("duration_seconds", 0.0))
                        row["duration_ratio"] = generated_duration / max(label_duration, 1e-9)
                if utmos is not None:
                    mos = score_utmos(utmos, audio)
                    row["utmos"] = mos
                    row["scores"]["speech_quality"] = row["audio_diagnostics"]["audio_valid"] * max(
                        0.0, min((mos - 1.0) / 4.0, 1.0)
                    )
            elif row.get("metadata", {}).get("require_audio"):
                row.setdefault("scores", {})["speech_semantic"] = 0.0
                row["scores"]["speech_quality"] = 0.0
                row["scores"]["primary"] = 0.0
            rows.append(row)
            output_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            output_handle.flush()
            if index == 1 or index % 10 == 0 or index == len(source_lines):
                label = f" {args.shard_label}" if args.shard_label else ""
                print(f"[{index}/{len(source_lines)}]{label} speech scored", flush=True)
    if args.skip_metrics:
        print(f"Scored {len(rows)} rows -> {destination}")
        return
    metrics = aggregate(rows)
    raw_metrics = destination.parent / "metrics.json"
    if raw_metrics.exists():
        environment = json.loads(raw_metrics.read_text(encoding="utf-8")).get("environment")
        if environment:
            metrics["environment"] = environment
    (destination.parent / "metrics_scored.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Scored {len(rows)} rows -> {destination}")


if __name__ == "__main__":
    main()
