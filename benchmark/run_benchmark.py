#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import random
import platform
import sys
import time
import traceback
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import SCORER_VERSION, aggregate, audio_diagnostics, score_sample
from benchmark.schema import BenchmarkSample, GenerationResult


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_adapter(spec: dict, output_dir: Path):
    module_name, class_name = spec["adapter"].rsplit(":", 1)
    cls = getattr(importlib.import_module(module_name), class_name)
    kwargs = dict(spec.get("kwargs", {}))
    return cls(model_id=spec["id"], output_dir=output_dir, **kwargs)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def path_fingerprint(value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        return f"missing:{path}"
    digest = hashlib.sha256()
    if path.is_file():
        digest.update(path.read_bytes())
    else:
        candidates = []
        for pattern in ("config.json", "generation_config.json", "*.safetensors", "*.bin"):
            candidates.extend(path.glob(pattern))
        for item in sorted(set(candidates)):
            stat = item.stat()
            digest.update(f"{item.name}\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode())
            if item.name.endswith(".json"):
                digest.update(item.read_bytes())
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the objective Omni benchmark")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, help="Model id from models.json")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("--shard-index must be in [0, --num-shards)")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    model_spec = next((model for model in config["models"] if model["id"] == args.model), None)
    if model_spec is None:
        raise SystemExit(f"Unknown model id: {args.model}")
    manifest = Path(args.manifest or config["manifest"]).expanduser().resolve()
    output_root = Path(args.output_root or config["output_root"]).expanduser().resolve()
    run_id = args.run_id or config["run_id"]
    run_dir = output_root / run_id / args.model
    run_dir.mkdir(parents=True, exist_ok=True)
    shard_suffix = f".shard-{args.shard_index:02d}-of-{args.num_shards:02d}" if args.num_shards > 1 else ""
    output_jsonl = run_dir / f"per_sample{shard_suffix}.jsonl"
    metadata_path = run_dir / f"run_metadata{shard_suffix}.json"
    metrics_path = run_dir / f"metrics{shard_suffix}.json"
    adapter_module = model_spec["adapter"].split(":", 1)[0]
    adapter_origin = getattr(importlib.util.find_spec(adapter_module), "origin", None)
    fingerprint_payload = {
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_fingerprint": path_fingerprint(model_spec.get("kwargs", {}).get("checkpoint")),
        "adapter_sha256": (
            hashlib.sha256(Path(adapter_origin).read_bytes()).hexdigest() if adapter_origin else None
        ),
        "seed": int(config.get("seed", 20260913)),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "scorer_version": SCORER_VERSION,
        "text_decoding": "greedy",
    }
    run_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    if args.resume and metadata_path.is_file():
        previous = json.loads(metadata_path.read_text(encoding="utf-8"))
        if previous.get("run_fingerprint") != run_fingerprint:
            raise RuntimeError(
                f"refusing stale resume for {args.model}: manifest/model/adapter/decoding settings changed"
            )
    elif args.resume and (output_jsonl.exists() or (run_dir / "per_sample.jsonl").exists()):
        raise RuntimeError(f"refusing resume without fingerprint metadata: {metadata_path}")
    samples = [BenchmarkSample.from_dict(row, manifest.parent) for row in read_jsonl(manifest)]
    enabled = set(config.get("tasks", []))
    if enabled:
        samples = [sample for sample in samples if sample.task in enabled]
    if args.limit:
        samples = samples[: args.limit]
    indexed_samples = [
        (index, sample) for index, sample in enumerate(samples)
        if index % args.num_shards == args.shard_index
    ]

    existing_by_id = {}
    if args.resume:
        for candidate in (output_jsonl, run_dir / "per_sample.jsonl"):
            if candidate.exists():
                existing_by_id.update({row["sample_id"]: row for row in read_jsonl(candidate)})
    seeded_rows = []
    for index, sample in indexed_samples:
        row = existing_by_id.get(sample.sample_id)
        if row is not None:
            row = dict(row)
            row["benchmark_index"] = index
            seeded_rows.append(row)
    completed = {row["sample_id"] for row in seeded_rows}
    if seeded_rows:
        output_jsonl.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in seeded_rows),
            encoding="utf-8",
        )

    if len(completed) == len(indexed_samples):
        output_jsonl.touch(exist_ok=True)
        print(
            f"Shard already complete: {args.model} "
            f"{args.shard_index + 1}/{args.num_shards} ({len(indexed_samples)} samples)",
            flush=True,
        )
        return

    audio_dir = run_dir / "audio"
    if args.num_shards > 1:
        audio_dir = audio_dir / f"shard-{args.shard_index:02d}"
    adapter = load_adapter(model_spec, audio_dir)
    adapter.load()
    environment = {
        "model_id": args.model,
        "role": model_spec.get("role"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "adapter": model_spec["adapter"],
        "checkpoint": model_spec.get("kwargs", {}).get("checkpoint"),
    }
    try:
        import torch
        environment["torch"] = torch.__version__
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cuda_available = torch.cuda.is_available()
        environment["accelerator"] = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    except Exception as exc:
        environment["accelerator"] = f"unknown: {exc}"
    environment.update({
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "batch_size": args.batch_size,
        "run_fingerprint": run_fingerprint,
        "fingerprint_payload": fingerprint_payload,
    })
    atomic_json(metadata_path, environment)
    try:
        with output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as handle:
            shard_total = len(indexed_samples)
            pending = [
                (benchmark_index, sample) for benchmark_index, sample in indexed_samples
                if sample.sample_id not in completed
            ]
            processed = len(completed)
            for batch_start in range(0, len(pending), args.batch_size):
                batch_entries = pending[batch_start:batch_start + args.batch_size]
                batch_samples = [sample for _, sample in batch_entries]
                seed = int(config.get("seed", 20260913))
                random.seed(seed)
                started = time.perf_counter()
                try:
                    batch_results = adapter.generate_batch(batch_samples, [seed] * len(batch_samples))
                    if len(batch_results) != len(batch_samples):
                        raise RuntimeError(
                            f"adapter returned {len(batch_results)} results for {len(batch_samples)} samples"
                        )
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    trace = traceback.format_exc(limit=8)
                    elapsed = time.perf_counter() - started
                    batch_results = [
                        GenerationResult(
                            sample.sample_id,
                            args.model,
                            status="error",
                            error=error,
                            latency_seconds=elapsed,
                            metadata={"traceback": trace},
                        )
                        for sample in batch_samples
                    ]
                batch_elapsed = time.perf_counter() - started
                for (benchmark_index, sample), result in zip(batch_entries, batch_results):
                    processed += 1
                    if result.latency_seconds is None:
                        result.latency_seconds = batch_elapsed
                    result_payload = result.to_dict()
                    # Adapter diagnostics augment sample metadata; they must not
                    # erase manifest fields such as base_id and require_audio.
                    result_payload["metadata"] = {
                        **sample.metadata,
                        **result_payload.get("metadata", {}),
                    }
                    row = {**sample.__dict__, **result_payload}
                    row["benchmark_index"] = benchmark_index
                    row["audio_diagnostics"] = audio_diagnostics(result.generated_audio)
                    row["scores"] = score_sample(row, row)
                    if sample.metadata.get("require_audio"):
                        diag = row["audio_diagnostics"]
                        row["scores"]["speech_quality"] = diag["audio_valid"] * max(
                            0.0, 1.0 - min(diag["clipping_rate"] * 10.0 + max(diag["silence_rate"] - 0.8, 0.0), 1.0)
                        )
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    if processed == 1 or processed == shard_total or processed % args.log_every == 0 or result.status != "ok":
                        print(
                            f"[{processed}/{shard_total}] shard "
                            f"{args.shard_index + 1}/{args.num_shards} {sample.sample_id}: {result.status}",
                            flush=True,
                        )
    finally:
        adapter.close()

    rows = list(read_jsonl(output_jsonl))
    summary = aggregate(rows)
    summary.update({"model_id": args.model, "run_id": run_id, "manifest": str(manifest)})
    summary["environment"] = environment
    atomic_json(metrics_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
