#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import aggregate


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge deterministic benchmark shards")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--num-shards", required=True, type=int)
    parser.add_argument("--stem", default="per_sample", choices=["per_sample", "per_sample_scored"])
    parser.add_argument("--skip-metrics", action="store_true")
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num-shards must be positive")

    run_dir = args.run_dir.resolve()
    rows = []
    shard_paths = []
    for index in range(args.num_shards):
        path = run_dir / f"{args.stem}.shard-{index:02d}-of-{args.num_shards:02d}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(f"missing shard: {path}")
        shard_paths.append(path)
        rows.extend(read_jsonl(path))
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate sample_id across shards")
    rows.sort(key=lambda row: (int(row.get("benchmark_index", 1 << 60)), row["sample_id"]))
    destination = run_dir / f"{args.stem}.jsonl"
    atomic_write(destination, "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))

    if not args.skip_metrics and rows and "task" in rows[0]:
        metrics = aggregate(rows)
        raw_metrics = run_dir / "metrics.json"
        shard_metrics = run_dir / f"metrics.shard-00-of-{args.num_shards:02d}.json"
        source_metrics = shard_metrics if shard_metrics.exists() else raw_metrics
        if source_metrics.exists():
            source = json.loads(source_metrics.read_text(encoding="utf-8"))
            for key in ("model_id", "run_id", "manifest", "environment"):
                if key in source:
                    metrics[key] = source[key]
        metrics_name = "metrics_scored.json" if args.stem == "per_sample_scored" else "metrics.json"
        atomic_write(run_dir / metrics_name, json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    print(f"Merged {len(rows)} rows from {len(shard_paths)} shards -> {destination}")


if __name__ == "__main__":
    main()
