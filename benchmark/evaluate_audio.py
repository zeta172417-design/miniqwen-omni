#!/usr/bin/env python3
"""Apply one shared ASR and optional UTMOS scorer to generated speech."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import aggregate, word_error_rate


def load_asr(path: str, device: str = "cpu"):
    from funasr import AutoModel
    return AutoModel(model=path, trust_remote_code=True, disable_update=True, device=device, disable_log=True)


def transcribe(model, path: str) -> str:
    import librosa
    from funasr.utils.postprocess_utils import rich_transcription_postprocess
    waveform, _ = librosa.load(path, sr=16000, mono=True)
    result = model.generate(input=waveform, cache={}, language="en", use_itn=True, disable_pbar=True, disable_log=True)
    return rich_transcription_postprocess(result[0]["text"]).strip() if result else ""


def load_utmos(repo: str | None):
    if not repo:
        return None
    import torch
    if Path(repo).exists():
        return torch.hub.load(repo, "utmos22_strong", source="local").eval()
    return torch.hub.load(repo, "utmos22_strong", trust_repo=True).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--asr", required=True)
    parser.add_argument("--device", default="cpu", help="SenseVoice device, for example cpu or cuda:0")
    parser.add_argument("--utmos-repo", default=None, help="Optional local SpeechMOS repo or torch.hub spec")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    source = Path(args.input)
    destination = Path(args.output) if args.output else source.with_name("per_sample_scored.jsonl")
    asr, utmos = load_asr(args.asr, args.device), load_utmos(args.utmos_repo)
    rows = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        audio = row.get("generated_audio")
        if audio and Path(audio).is_file():
            transcript = transcribe(asr, audio)
            row["generated_audio_text"] = transcript
            if transcript.strip() and str(row.get("generated_text", "")).strip():
                consistency = max(0.0, 1.0 - min(word_error_rate(transcript, row["generated_text"]), 1.0))
            else:
                consistency = 0.0
            row.setdefault("scores", {})["speech_semantic"] = consistency
            if utmos is not None:
                import librosa
                import torch
                waveform, _ = librosa.load(audio, sr=16000, mono=True)
                with torch.inference_mode():
                    mos = float(utmos(torch.tensor(waveform).unsqueeze(0), 16000).squeeze())
                row["utmos"] = mos
                row["scores"]["speech_quality"] = row["audio_diagnostics"]["audio_valid"] * max(
                    0.0, min((mos - 1.0) / 4.0, 1.0)
                )
        elif row.get("metadata", {}).get("require_audio"):
            row.setdefault("scores", {})["speech_semantic"] = 0.0
            row["scores"]["speech_quality"] = 0.0
        rows.append(row)
    destination.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
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
