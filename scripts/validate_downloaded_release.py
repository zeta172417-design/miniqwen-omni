#!/usr/bin/env python3
"""Validate the directory layout expected after a ModelScope download."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

required_data = [
    ROOT / "dataset/pretrain/AIA",
    ROOT / "dataset/pretrain/IIA",
    ROOT / "dataset/pretrain/AOA",
    ROOT / "dataset/sft/manifest.json",
]
required_models = [
    ROOT / "out/miniqwen_sft/last/checkpoint/config.json",
    ROOT / "out/miniqwen_sft/last/checkpoint/model.safetensors",
    ROOT / "out/miniqwen_pretrain/tts/last/checkpoint/config.json",
    ROOT / "out/miniqwen_pretrain/tts/last/checkpoint/model.safetensors",
]
missing = [str(path.relative_to(ROOT)) for path in required_data + required_models if not path.exists()]
if missing:
    raise SystemExit("Missing release files:\n" + "\n".join(missing))

for path in sorted((ROOT / "dataset/pretrain").rglob("*.parquet")):
    if path.stat().st_size == 0:
        raise SystemExit(f"Empty Parquet: {path}")
print("Downloaded data and model layout is ready")
print(json.dumps({
    "data_root": str(ROOT / "dataset"),
    "v1_model": str(ROOT / "out/miniqwen_sft/last/checkpoint"),
    "base_model": str(ROOT / "out/miniqwen_pretrain/tts/last/checkpoint"),
}, indent=2))
