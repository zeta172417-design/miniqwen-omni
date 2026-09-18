#!/usr/bin/env python3
"""Build a small, fixed LibriSpeech voice-cloning evaluation set."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import naturalize_tts_text, normalize_asr_text


DEFAULT_ARROW = (
    "/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/modelscope-cache/"
    "openslr___librispeech_asr/clean-e26c7aa5cc2ad4f5/0.0.0/master/"
    "librispeech_asr-test.arrow"
)
DEFAULT_OUTPUT = "/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/voice-clone-v2.3-standard"


def duration(audio_bytes: bytes) -> float:
    return float(sf.info(io.BytesIO(audio_bytes)).duration)


def select_rows(rows: list[dict], speakers: int, targets: int, seed: int) -> list[dict]:
    """Select one reference and ``targets`` different utterances per speaker."""
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        audio_bytes = (row.get("audio") or {}).get("bytes")
        text = " ".join(str(row.get("text") or "").split())
        if not audio_bytes or not text:
            continue
        seconds = duration(audio_bytes)
        if 2.0 <= seconds <= 10.0 and 4 <= len(text.split()) <= 28:
            grouped[str(row["speaker_id"])].append({**row, "text": text, "duration": seconds})

    for speaker, items in grouped.items():
        unique = {}
        for item in items:
            unique.setdefault(normalize_asr_text(item["text"]), item)
        grouped[speaker] = list(unique.values())

    eligible = [speaker for speaker, items in grouped.items() if len(items) >= targets + 1]
    random.Random(seed).shuffle(eligible)
    if len(eligible) < speakers:
        raise RuntimeError(f"Only {len(eligible)} eligible speakers; requested {speakers}")

    selected = []
    for speaker in eligible[:speakers]:
        items = sorted(grouped[speaker], key=lambda row: row["id"])
        random.Random(seed + int(speaker)).shuffle(items)
        reference, *target_rows = items[: targets + 1]
        selected.append({"speaker_id": speaker, "reference": reference, "targets": target_rows})
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arrow", default=DEFAULT_ARROW)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--speakers", type=int, default=8)
    parser.add_argument("--targets-per-speaker", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260914)
    parser.add_argument("--exclude-manifest", default=None)
    args = parser.parse_args()

    arrow_path = Path(args.arrow).resolve()
    output = Path(args.output).resolve()
    if not arrow_path.is_file():
        raise FileNotFoundError(f"LibriSpeech Arrow cache not found: {arrow_path}")

    rows = []
    with pa.memory_map(str(arrow_path), "r") as source:
        for batch in ipc.open_stream(source):
            rows.extend(batch.to_pylist())
    if args.exclude_manifest:
        excluded = set()
        for line in Path(args.exclude_manifest).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            for value in (
                sample.get("reference"), sample.get("prompt"),
                sample.get("metadata", {}).get("transcript_prompt"),
                sample.get("metadata", {}).get("target_text"),
            ):
                if isinstance(value, str) and value.strip():
                    excluded.add(normalize_asr_text(value))
        rows = [row for row in rows if normalize_asr_text(row.get("text")) not in excluded]
    selected = select_rows(rows, args.speakers, args.targets_per_speaker, args.seed)

    manifest_rows = []
    for group in selected:
        speaker = group["speaker_id"]
        speaker_dir = output / "audio" / speaker
        speaker_dir.mkdir(parents=True, exist_ok=True)
        reference = group["reference"]
        reference_path = speaker_dir / f"ref-{reference['id']}.flac"
        reference_path.write_bytes(reference["audio"]["bytes"])
        for index, target in enumerate(group["targets"]):
            label_path = speaker_dir / f"label-{target['id']}.flac"
            label_path.write_bytes(target["audio"]["bytes"])
            manifest_rows.append({
                "sample_id": f"clone-{speaker}-{index:02d}",
                "speaker_id": speaker,
                "ref_audio": str(reference_path),
                "ref_text": reference["text"],
                "ref_duration_seconds": reference["duration"],
                "target_text": naturalize_tts_text(target["text"]),
                "label_audio": str(label_path),
                "label_duration_seconds": target["duration"],
                "source_id": target["id"],
                "source": "openslr/librispeech_asr:clean:test",
            })

    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in manifest_rows)
    manifest.write_text(payload, encoding="utf-8")
    metadata = {
        "name": "omni-voice-clone-v2.3",
        "source": "openslr/librispeech_asr:clean:test",
        "seed": args.seed,
        "speakers": args.speakers,
        "targets_per_speaker": args.targets_per_speaker,
        "samples": len(manifest_rows),
        "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
    }
    (output / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
