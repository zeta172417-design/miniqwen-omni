#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
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

from benchmark.metrics import aggregate, audio_diagnostics, score_sample
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
    args = parser.parse_args()

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
    output_jsonl = run_dir / "per_sample.jsonl"
    completed = {row["sample_id"] for row in read_jsonl(output_jsonl)} if args.resume and output_jsonl.exists() else set()
    samples = [BenchmarkSample.from_dict(row, manifest.parent) for row in read_jsonl(manifest)]
    enabled = set(config.get("tasks", []))
    if enabled:
        samples = [sample for sample in samples if sample.task in enabled]
    if args.limit:
        samples = samples[: args.limit]

    adapter = load_adapter(model_spec, run_dir / "audio")
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
    atomic_json(run_dir / "run_metadata.json", environment)
    try:
        with output_jsonl.open("a" if args.resume else "w", encoding="utf-8") as handle:
            for index, sample in enumerate(samples, 1):
                if sample.sample_id in completed:
                    continue
                seed = int(config.get("seed", 20260913))
                random.seed(seed)
                started = time.perf_counter()
                try:
                    result = adapter.generate(sample, seed)
                    if result.latency_seconds is None:
                        result.latency_seconds = time.perf_counter() - started
                except Exception as exc:
                    result = GenerationResult(
                        sample.sample_id,
                        args.model,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                        latency_seconds=time.perf_counter() - started,
                        metadata={"traceback": traceback.format_exc(limit=8)},
                    )
                result_payload = result.to_dict()
                # Adapter diagnostics augment sample metadata; they must not
                # erase manifest fields such as base_id and require_audio.
                result_payload["metadata"] = {
                    **sample.metadata,
                    **result_payload.get("metadata", {}),
                }
                row = {**sample.__dict__, **result_payload}
                row["audio_diagnostics"] = audio_diagnostics(result.generated_audio)
                row["scores"] = score_sample(row, row)
                if sample.metadata.get("require_audio"):
                    diag = row["audio_diagnostics"]
                    row["scores"]["speech_quality"] = diag["audio_valid"] * max(
                        0.0, 1.0 - min(diag["clipping_rate"] * 10.0 + max(diag["silence_rate"] - 0.8, 0.0), 1.0)
                    )
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index == 1 or index == len(samples) or index % args.log_every == 0 or result.status != "ok":
                    print(f"[{index}/{len(samples)}] {sample.sample_id}: {result.status}", flush=True)
    finally:
        adapter.close()

    rows = list(read_jsonl(output_jsonl))
    summary = aggregate(rows)
    summary.update({"model_id": args.model, "run_id": run_id, "manifest": str(manifest)})
    summary["environment"] = environment
    atomic_json(run_dir / "metrics.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
