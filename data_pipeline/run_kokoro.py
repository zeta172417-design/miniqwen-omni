#!/usr/bin/env python3
"""Launch one Kokoro worker per visible GPU device."""
from __future__ import annotations
import os, subprocess, sys

root = os.path.dirname(os.path.dirname(__file__))
n = int(os.environ.get("KOKORO_SHARDS", "16"))
procs = []
for shard in range(n):
    env = os.environ.copy()
    env["KOKORO_SHARD"] = str(shard)
    env["KOKORO_SHARDS"] = str(n)
    env["HF_HUB_OFFLINE"] = "1"
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["CUDA_VISIBLE_DEVICES"] = str(shard)
    log = open(os.path.join(root, ".runtime", "kokoro-a2a", f"gpu-{shard:02d}.log"), "w", encoding="utf-8")
    procs.append(subprocess.Popen([sys.executable, "data_pipeline/kokoro_worker.py"], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT))
codes = [p.wait() for p in procs]
raise SystemExit(max(codes))
