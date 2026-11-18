#!/usr/bin/env python3
"""Stage download-ready ModelScope dataset repositories without mutating data.

Parquet files are hard-linked by default to avoid a second 85 GiB local copy.
The SFT stage also collects only media referenced by its path columns and keeps
the original ``dataset/raw/...`` relative layout, so a download to ``dataset/``
works with the existing readers.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]


def install(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def copy_tree(source_root: Path, target_root: Path) -> int:
    count = 0
    for source in source_root.rglob("*.parquet"):
        install(source, target_root / source.relative_to(source_root))
        count += 1
    return count


def collect_sft_media() -> list[Path]:
    media: set[Path] = set()
    for parquet in (ROOT / "dataset/sft").rglob("*.parquet"):
        names = pq.ParquetFile(parquet).schema_arrow.names
        columns = [name for name in ("question_audio_paths", "image_paths") if name in names]
        if not columns:
            continue
        for row in pq.ParquetFile(parquet).read(columns=columns).to_pylist():
            for name in columns:
                values = row.get(name) or []
                if isinstance(values, str):
                    values = [values]
                for value in values:
                    if isinstance(value, str):
                        path = ROOT / value
                        if path.is_file():
                            media.add(path)
                        else:
                            raise FileNotFoundError(f"SFT media reference is missing: {value}")
    return sorted(media)


def stage_pretrain(destination: Path) -> dict:
    count = copy_tree(ROOT / "dataset/pretrain", destination / "pretrain")
    source = ROOT / "dataset/raw/manifests/pretrain/plan.json"
    install(source, destination / "raw/manifests/pretrain/plan.json")
    return {"parquet_files": count}


def stage_sft(destination: Path) -> dict:
    count = copy_tree(ROOT / "dataset/sft", destination / "sft")
    manifest = ROOT / "dataset/sft/manifest.json"
    if manifest.is_file():
        install(manifest, destination / "sft/manifest.json")
    plan = ROOT / "dataset/raw/manifests/sft/plan.json"
    install(plan, destination / "raw/manifests/sft/plan.json")
    # Keep the original relative media tree. This is intentionally broader
    # than a row-by-row reference scan: it guarantees that every historical
    # path remains valid after downloading the dataset into ``dataset/``.
    media_root = ROOT / "dataset/raw/external"
    media_files = [p for p in media_root.rglob("*") if p.is_file()]
    for source in media_files:
        install(source, destination / source.relative_to(ROOT))
    return {"parquet_files": count, "media_files": len(media_files),
            "media_policy": "preserve dataset/raw/external relative tree"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pretrain-output", type=Path, required=True)
    parser.add_argument("--sft-output", type=Path, required=True)
    args = parser.parse_args()
    args.pretrain_output.mkdir(parents=True, exist_ok=True)
    args.sft_output.mkdir(parents=True, exist_ok=True)
    result = {"pretrain": stage_pretrain(args.pretrain_output.resolve()),
              "sft": stage_sft(args.sft_output.resolve())}
    for path in [args.pretrain_output, args.sft_output]:
        (path / "DATASET_CARD.md").write_text(
            "# MiniQwen-Omni prepared training data\n\n"
            "This repository is downloaded into the project `dataset/` directory. "
            "Keep the relative `raw/` media paths unchanged.\n",
            encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
