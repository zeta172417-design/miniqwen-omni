#!/usr/bin/env python3
"""Generate the 10k speech-MCQ waveforms with Kokoro-82M.

This intentionally writes into a new raw derivation. The previous Flite
derivation remains available for audit and is never mixed into the new pool.
Run with the isolated Kokoro CPU environment, for example:
``/tmp/cosy-cpu/bin/python data_pipeline/prepare_speech_mcq_kokoro.py``.
"""
from __future__ import annotations

import hashlib
import json
import concurrent.futures
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
from kokoro import KPipeline

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "dataset/raw/external/speech_mcq/source"
RAW = ROOT / "dataset/raw/external/speech_mcq/10k_audio_kokoro"


def make_sources() -> list[tuple[str, str, str]]:
    sources: list[tuple[str, str, str]] = []
    for r in pq.read_table(SRC / "arc.parquet").to_pylist()[:2251]:
        choices = list(r["choices"]["text"])
        labels = list(r["choices"]["label"])
        prompt = r["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(labels, choices))
        sources.append(("arc", prompt, r["answerKey"]))
    for r in pq.read_table(SRC / "boolq.parquet").to_pylist()[:3875]:
        passage = " ".join(str(r["passage"]).split()[:60])
        prompt = passage + "\n\n" + r["question"] + "\nA. Yes\nB. No"
        sources.append(("boolq", prompt, "A" if r["answer"] else "B"))
    for r in pq.read_table(SRC / "piqa.parquet").to_pylist()[:3874]:
        goal = " ".join(str(r["goal"]).split()[:30])
        sol1 = " ".join(str(r["sol1"]).split()[:30])
        sol2 = " ".join(str(r["sol2"]).split()[:30])
        sources.append(("piqa", goal + "\nA. " + sol1 + "\nB. " + sol2, "A" if int(r["label"]) == 0 else "B"))
    assert len(sources) == 10_000
    return sources


def synth(pipeline: KPipeline, text: str, path: Path, voice: str) -> None:
    if path.exists() and path.stat().st_size > 1024:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks = []
    for _, _, audio in pipeline(text, voice=voice, speed=1.0):
        chunks.append(audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio))
    if not chunks:
        raise RuntimeError(f"Kokoro produced no audio for {path}: {text[:80]}")
    audio = np.concatenate(chunks).astype(np.float32)
    if not np.isfinite(audio).all() or len(audio) < 240:
        raise RuntimeError(f"invalid Kokoro audio for {path}")
    sf.write(path, audio, 24_000, subtype="PCM_16")


_PIPELINE = None


def init_worker() -> None:
    global _PIPELINE
    import torch
    torch.set_num_threads(2)
    _PIPELINE = KPipeline(lang_code="a")


def make_one(item: tuple[int, str, str, str]) -> dict:
    i, source, prompt, answer = item
    q = RAW / "questions" / f"{i:05d}.wav"
    a = RAW / "answers" / f"{i:05d}.wav"
    synth(_PIPELINE, prompt, q, "af_heart")
    synth(_PIPELINE, answer, a, "af_heart")
    return {"i": i, "source": source, "prompt": prompt, "answer": answer,
            "q": str(q), "a": str(a), "voice": "af_heart", "tts_engine": "kokoro-82m"}


def main() -> None:
    records_path = RAW / "records.jsonl"
    RAW.mkdir(parents=True, exist_ok=True)
    sources = make_sources()
    items = [(i, source, prompt, answer) for i, (source, prompt, answer) in enumerate(sources)]
    records = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=16, initializer=init_worker) as ex:
        for n, rec in enumerate(ex.map(make_one, items), 1):
            records.append(rec)
            if n % 50 == 0:
                print(f"SYNTH_READY {n}/10000", flush=True)
    with records_path.open("w", encoding="utf-8") as f:
        for rec in sorted(records, key=lambda x: x["i"]):
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("SYNTH_COMPLETE 10000", flush=True)


if __name__ == "__main__":
    main()
