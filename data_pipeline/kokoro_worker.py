#!/usr/bin/env python3
"""Generate one Kokoro shard on the GPU device selected by CUDA_VISIBLE_DEVICES."""
from __future__ import annotations
import importlib.machinery, json, os, sys, types
from pathlib import Path

import torch  # must be imported before adding the CPU venv site-packages

# The shared GPU image has a torchvision extension built for another torch.
tv = types.ModuleType("torchvision"); tv.__path__ = []
tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", loader=None, is_package=True)
tvt = types.ModuleType("torchvision.transforms")
tvt.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", loader=None)
class _InterpolationMode:
    NEAREST = 0; NEAREST_EXACT = 0; BILINEAR = 2; BICUBIC = 3; BOX = 4; HAMMING = 5; LANCZOS = 1
tvt.InterpolationMode = _InterpolationMode; tv.transforms = tvt
sys.modules["torchvision"] = tv; sys.modules["torchvision.transforms"] = tvt

sys.path.insert(0, "/tmp/kokoro_gpu_pkgs")
from kokoro import KPipeline
import numpy as np
import soundfile as sf
from prepare_speech_mcq_kokoro import make_sources, RAW

def synth(pipe: KPipeline, text: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1024:
        return
    chunks = []
    for _, _, audio in pipe(text, voice="af_heart", speed=1.0):
        chunks.append(audio.detach().cpu().numpy())
    if not chunks:
        raise RuntimeError(f"empty Kokoro output: {path}")
    x = np.concatenate(chunks).astype(np.float32)
    if len(x) < 240 or not np.isfinite(x).all():
        raise RuntimeError(f"invalid Kokoro output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, x, 24_000, subtype="PCM_16")

def main() -> None:
    shard = int(os.environ["KOKORO_SHARD"]); nshards = int(os.environ.get("KOKORO_SHARDS", "16"))
    torch.set_num_threads(2)
    pipe = KPipeline(lang_code="a")
    sources = make_sources()
    done = 0
    out = RAW / f"records-{shard:02d}.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i in range(shard, len(sources), nshards):
            source, prompt, answer = sources[i]
            q = RAW / "questions" / f"{i:05d}.wav"; a = RAW / "answers" / f"{i:05d}.wav"
            synth(pipe, prompt, q); synth(pipe, answer, a)
            f.write(json.dumps({"i": i, "source": source, "prompt": prompt, "answer": answer,
                                "q": str(q), "a": str(a), "voice": "af_heart", "tts_engine": "kokoro-82m"}, ensure_ascii=False) + "\n")
            done += 1
            if done % 25 == 0: print(f"SHARD {shard} READY {done}", flush=True)
    print(f"SHARD {shard} COMPLETE {done}", flush=True)

if __name__ == "__main__": main()
