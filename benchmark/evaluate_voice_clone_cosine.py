#!/usr/bin/env python3
"""Score only speaker cosine for reference-conditioned voice cloning."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def rows(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bootstrap(values: list[float], seed: int = 20261115) -> list[float]:
    import random
    rng = random.Random(seed)
    means = []
    for _ in range(10000):
        means.append(statistics.fmean(rng.choice(values) for _ in values))
    means.sort()
    return [means[250], means[9750]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models", required=True)
    parser.add_argument("--speaker-model", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    import librosa
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    manifest = rows(args.manifest)
    references = {row["sample_id"]: row["ref_audio"] for row in manifest}
    verifier = pipeline(task=Tasks.speaker_verification, model=str(args.speaker_model), device=args.device)
    cache: dict[str, object] = {}

    def wave(path: str):
        if path not in cache:
            cache[path] = librosa.load(path, sr=16000, mono=True)[0]
        return cache[path]

    summary = {"metric": "speaker_cosine_only", "samples": len(manifest), "models": {}}
    for model in args.models.split(","):
        source = args.run_dir / "voice_clone" / model / "per_sample.jsonl"
        if not source.exists():
            continue
        values = []
        for row in rows(source):
            generated = row.get("generated_audio")
            reference = references.get(row["sample_id"], row.get("ref_audio"))
            if not generated or not reference:
                continue
            values.append(float(verifier([wave(generated), wave(reference)])['score']))
        if values:
            summary["models"][model] = {
                "speaker_cosine": statistics.fmean(values),
                "ci95": bootstrap(values),
                "samples": len(values),
            }
    output = args.run_dir / "voice_clone" / "speaker_cosine_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
