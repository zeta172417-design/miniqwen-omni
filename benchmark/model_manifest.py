#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def files(root: Path):
    return [
        {"path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": sha256(path)}
        for path in sorted(root.iterdir())
        if path.is_file() and path.name not in {".msc", ".mv"}
    ]


def git_value(root: Path, *arguments: str) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(root), *arguments], text=True).strip()
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="/mnt/workspace/zhaozetao/multimodel/Omni")
    args = parser.parse_args()
    root = Path(args.workspace).resolve()
    mini_source = root / "mini-omni2/source"
    miniqwen = root / "miniqwen-omni"
    payload = {
        "schema_version": 1,
        "models": {
            "Qwen2.5-Omni-3B": {
                "source": "ModelScope:Qwen/Qwen2.5-Omni-3B",
                "revision": "master",
                "files": files(root / "Qwen2.5-Omni-3B"),
            },
            "Mini-Omni2": {
                "source": "HuggingFace:gpt-omni/mini-omni2",
                "revision": "49f474f4c38f80cf716859bb1b1442df2a8dea46",
                "code_commit": git_value(mini_source, "rev-parse", "HEAD"),
                "code_dirty": bool(git_value(mini_source, "status", "--porcelain")),
                "code_remote": git_value(mini_source, "remote", "get-url", "origin"),
                "files": files(root / "mini-omni2/checkpoint"),
            },
            "MiniQwen-Omni": {
                "source": str(miniqwen.resolve()),
                "code_commit": git_value(miniqwen, "rev-parse", "HEAD"),
                "code_dirty": bool(git_value(miniqwen, "status", "--porcelain")),
                "checkpoints": {
                    "v0.1": {
                        "path": str(miniqwen / "out/miniqwen_omni_full_main_codec_cp_v5/checkpoint"),
                        "files": files(miniqwen / "out/miniqwen_omni_full_main_codec_cp_v5/checkpoint"),
                    },
                },
            },
        },
    }
    destination = root / "models-manifest.json"
    destination.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
