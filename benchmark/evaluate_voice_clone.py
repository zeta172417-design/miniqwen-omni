#!/usr/bin/env python3
"""Score reference voice cloning with speaker similarity and ASR content accuracy."""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.evaluate_audio import load_asr, load_utmos, score_utmos, transcribe
from benchmark.metrics import (
    audio_diagnostics, bootstrap_ci, word_edit_similarity, word_error_rate,
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def stat(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "ci95": [None, None], "samples": 0}
    low, high = bootstrap_ci(values)
    return {"mean": statistics.fmean(values), "ci95": [low, high], "samples": len(values)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--models", default="miniqwen-v0,miniqwen-v0.1,minimind-3o-moe")
    parser.add_argument("--asr", default=str(PROJECT_ROOT / "model/SenseVoiceSmall"))
    parser.add_argument("--speaker-model", default=str(PROJECT_ROOT / "model/eres2net_sv_en_voxceleb_16k"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--utmos-repo", default=None)
    args = parser.parse_args()

    import librosa
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    logging.getLogger("modelscope").setLevel(logging.ERROR)
    run_dir = Path(args.run_dir).resolve()
    manifest = read_jsonl(Path(args.manifest))
    by_speaker = {}
    for sample in manifest:
        by_speaker.setdefault(sample["speaker_id"], sample["ref_audio"])
    speaker_ids = sorted(by_speaker)
    wrong_ref = {
        speaker: by_speaker[speaker_ids[(index + 1) % len(speaker_ids)]]
        for index, speaker in enumerate(speaker_ids)
    }

    print("Loading SenseVoice and independent ERes2Net verifier", flush=True)
    asr = load_asr(args.asr, args.device)
    utmos = load_utmos(args.utmos_repo)
    verifier = pipeline(task=Tasks.speaker_verification, model=args.speaker_model, device=args.device)
    waveform_cache = {}
    similarity_cache = {}

    def waveform(path: str):
        if path not in waveform_cache:
            waveform_cache[path] = librosa.load(path, sr=16000, mono=True)[0]
        return waveform_cache[path]

    def similarity(left: str, right: str) -> float:
        key = (left, right)
        if key not in similarity_cache:
            similarity_cache[key] = float(verifier([waveform(left), waveform(right)])["score"])
        return similarity_cache[key]

    reference_rows = []
    for sample in manifest:
        reference_rows.append({
            "sample_id": sample["sample_id"],
            "real_similarity": similarity(sample["ref_audio"], sample["label_audio"]),
            "wrong_similarity": similarity(wrong_ref[sample["speaker_id"]], sample["label_audio"]),
            "natural_label_wer": word_error_rate(
                transcribe(asr, sample["label_audio"]), sample["target_text"],
            ),
            "natural_label_utmos": score_utmos(utmos, sample["label_audio"]),
        })
    reference_by_id = {row["sample_id"]: row for row in reference_rows}
    summary = {
        "name": "omni-voice-clone-v2.3",
        "samples": len(manifest),
        "speakers": len(speaker_ids),
        "reference": {
            "real_similarity": stat([row["real_similarity"] for row in reference_rows]),
            "wrong_similarity": stat([row["wrong_similarity"] for row in reference_rows]),
            "natural_label_wer": stat([row["natural_label_wer"] for row in reference_rows]),
            "natural_label_utmos": stat([
                row["natural_label_utmos"] for row in reference_rows
                if row["natural_label_utmos"] is not None
            ]),
        },
        "models": {},
    }
    scored_by_model = {}

    for model_id in args.models.split(","):
        source = run_dir / "voice_clone" / model_id / "per_sample.jsonl"
        if not source.exists():
            continue
        rows = read_jsonl(source)
        scored = []
        for index, row in enumerate(rows, 1):
            diag = audio_diagnostics(row.get("generated_audio"))
            valid = float(diag["audio_valid"] > 0 and row.get("status") == "ok")
            transcript, speaker_similarity, mos = "", 0.0, None
            if valid:
                transcript = transcribe(asr, row["generated_audio"])
                speaker_similarity = similarity(row["generated_audio"], row["ref_audio"])
                if utmos is not None:
                    mos = score_utmos(utmos, row["generated_audio"])
            content_wer = word_error_rate(transcript, row["target_text"]) if valid else 1.0
            content_score = valid * word_edit_similarity(transcript, row["target_text"])
            scored_row = {
                **row,
                **reference_by_id[row["sample_id"]],
                "generated_audio_text": transcript,
                "speaker_similarity": speaker_similarity,
                "utmos": mos,
                "content_score": content_score,
                "content_wer": content_wer,
                "audio_valid": valid,
                "audio_diagnostics": diag,
            }
            scored.append(scored_row)
            if index == 1 or index % 8 == 0 or index == len(rows):
                print(f"[{index}/{len(rows)}] scored {model_id}", flush=True)

        destination = source.with_name("per_sample_scored.jsonl")
        destination.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in scored), encoding="utf-8",
        )
        summary["models"][model_id] = {
            "speaker_similarity": stat([row["speaker_similarity"] for row in scored]),
            "content_score": stat([row["content_score"] for row in scored]),
            "audio_valid_rate": statistics.fmean(row["audio_valid"] for row in scored),
            "utmos": stat([row["utmos"] for row in scored if row.get("utmos") is not None]),
            "mean_latency_seconds": statistics.fmean(float(row["latency_seconds"]) for row in scored),
            "samples": len(scored),
        }
        scored_by_model[model_id] = {row["sample_id"]: row for row in scored}

    if "miniqwen-v0" in scored_by_model and "miniqwen-v0.1" in scored_by_model:
        left, right = scored_by_model["miniqwen-v0"], scored_by_model["miniqwen-v0.1"]
        common = sorted(left.keys() & right.keys())
        summary["v01_minus_v0"] = {
            "speaker_similarity": stat([
                right[key]["speaker_similarity"] - left[key]["speaker_similarity"] for key in common
            ]),
            "content_score": stat([
                right[key]["content_score"] - left[key]["content_score"] for key in common
            ]),
            "samples": len(common),
        }

    output = run_dir / "voice_clone" / "summary.json"
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
