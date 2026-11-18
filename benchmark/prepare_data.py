#!/usr/bin/env python3
"""Prepare the fixed English Omni Benchmark v2.4 manifests."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import naturalize_tts_text, normalize_text


BUILDER_VERSION = "2.4.0"


def stable_rows(dataset, count: int, seed: int):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return [dataset[index] for index in indices[:count]]


def stable_iter(dataset, seed: int):
    """Shuffle indices without decoding every image into a giant Python list."""
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    for index in indices:
        yield dataset[index]


def duration_seconds(path: str | Path) -> float:
    import soundfile as sf
    return float(sf.info(path).duration)


def mcq_prompt(question: str, choices: list[str], context: str = "") -> str:
    body = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    prefix = f"{context.strip()}\n\n" if context and context != "nan" else ""
    return f"{prefix}{question.strip()}\n{body}\nAnswer with only the option letter."


def save_image(value: Any, path: Path) -> str:
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, Image.Image):
        image = value
    elif isinstance(value, dict) and value.get("bytes"):
        image = Image.open(io.BytesIO(value["bytes"]))
    elif isinstance(value, dict) and value.get("path"):
        image = Image.open(value["path"])
    else:
        image = Image.open(value)
    image.convert("RGB").save(path, quality=92)
    return str(path)


def save_audio(value: Any, path: Path) -> str:
    import numpy as np
    import soundfile as sf
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict) and "array" in value:
        sf.write(path, np.asarray(value["array"]), int(value["sampling_rate"]))
    elif isinstance(value, dict) and value.get("bytes"):
        path.write_bytes(value["bytes"])
    elif isinstance(value, dict) and value.get("path"):
        shutil.copyfile(value["path"], path)
    else:
        shutil.copyfile(value, path)
    return str(path)


def synthesize(text: str, path: Path, voice: str, speed: float = 1.0) -> str:
    """Create a deterministic spoken prompt without depending on a benchmarked model."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(f"flite-v2.1\0{voice}\0{speed}\0{text}".encode()).hexdigest()
    stamp = path.with_suffix(path.suffix + ".sha256")
    if path.is_file() and stamp.is_file() and stamp.read_text().strip() == digest:
        return str(path)
    text_path = path.with_suffix(".txt")
    text_path.write_text(text, encoding="utf-8")
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
        f"flite=textfile='{text_path}':voice={voice}",
    ]
    if speed != 1.0:
        command += ["-filter:a", f"atempo={speed:.3f}"]
    command += ["-ar", "16000", "-ac", "1", str(path)]
    subprocess.run(command, check=True)
    text_path.unlink(missing_ok=True)
    stamp.write_text(digest + "\n")
    return str(path)


def canonical_decontamination_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"\b(answer|respond) with only the option letter\b.*$", " ", text, flags=re.I | re.S)
    text = re.sub(r"\banswer using a short phrase only\b.*$", " ", text, flags=re.I | re.S)
    return normalize_text(text)


def content_hash(text: str) -> bytes:
    # 128 digest bits are sufficient for local overlap detection and reduce a
    # multi-million-row training hash cache to a manageable size.
    return hashlib.sha256(normalize_text(text).encode()).digest()[:16]


def content_hashes(text: str) -> set[bytes]:
    variants = {normalize_text(text), canonical_decontamination_text(text)}
    return {hashlib.sha256(value.encode()).digest()[:16] for value in variants if value}


def training_hashes(paths: list[str], cache_path: Path | None = None) -> tuple[set[bytes], dict[str, Any]]:
    resolved = [Path(path).resolve() for path in paths if Path(path).exists()]
    fingerprint = [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in resolved]
    if cache_path and cache_path.is_file():
        try:
            cached = pickle.loads(cache_path.read_bytes())
            if cached.get("fingerprint") == fingerprint:
                print(f"loaded {len(cached['hashes'])} cached training-text hashes", flush=True)
                return cached["hashes"], cached.get("scan", {})
        except Exception:
            pass

    hashes: set[bytes] = set()
    import pyarrow.parquet as pq
    scanned_rows = 0
    for path in resolved:
        print(f"scanning training text for overlap: {path.name}", flush=True)
        parquet = pq.ParquetFile(path)
        if "conversations" not in parquet.schema.names:
            raise ValueError(f"training parquet has no conversations column: {path}")
        for batch in parquet.iter_batches(batch_size=2048, columns=["conversations"]):
            scanned_rows += batch.num_rows
            for row in batch.to_pylist():
                conversations = row.get("conversations") or []
                if isinstance(conversations, str):
                    conversations = json.loads(conversations)
                for turn in conversations:
                    if isinstance(turn, dict) and turn.get("role") in {"user", "human"}:
                        hashes.update(content_hashes(str(turn.get("content") or turn.get("value") or "")))
        print(
            f"decontamination scan: {scanned_rows:,} rows, {len(hashes):,} unique hashes",
            flush=True,
        )
    scan = {"files": [str(path) for path in resolved], "rows": scanned_rows, "hashes": len(hashes)}
    if not resolved or scanned_rows == 0:
        raise RuntimeError("training decontamination found no readable training rows")
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
        temporary.write_bytes(pickle.dumps({"fingerprint": fingerprint, "hashes": hashes, "scan": scan}, protocol=5))
        os.replace(temporary, cache_path)
    return hashes, scan


def canonical_arc_answer(value: Any, choices: list[str]) -> str:
    text = str(value).strip().upper()
    if text in [chr(65 + index) for index in range(len(choices))]:
        return text
    if text.isdigit() and 1 <= int(text) <= len(choices):
        return chr(64 + int(text))
    raise ValueError(f"invalid ARC answer {value!r} for {len(choices)} choices")


def canonical_zero_based_answer(value: Any, choices: list[str]) -> str:
    text = str(value).strip().upper()
    if text in [chr(65 + index) for index in range(len(choices))]:
        return text
    if text.isdigit() and 0 <= int(text) < len(choices):
        return chr(65 + int(text))
    raise ValueError(f"invalid zero-based answer {value!r} for {len(choices)} choices")


def textvqa_rows(parquet_dir: Path, count: int, seed: int):
    """Stream a deterministic subset from local, fully annotated validation shards."""
    from datasets import load_dataset

    files = sorted(str(path) for path in parquet_dir.glob("data/validation-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"TextVQA validation parquet is missing under {parquet_dir}; "
            "download lmms-lab-encoder/textvqa validation shards first"
        )
    dataset = load_dataset("parquet", data_files=files, split="train", streaming=True)
    iterator = dataset.shuffle(seed=seed, buffer_size=max(512, count * 8))
    accepted = []
    for row in iterator:
        answers = [str(answer).strip() for answer in (row.get("answers") or []) if str(answer).strip()]
        question = str(row.get("question") or "").strip()
        if question and answers and row.get("image") is not None:
            accepted.append(row)
        if len(accepted) == count:
            break
    if len(accepted) < count:
        raise RuntimeError(f"TextVQA supplied only {len(accepted)}/{count} valid rows")
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "benchmark/configs/standard.json"))
    parser.add_argument("--output", default=None)
    parser.add_argument("--cache", default="/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/modelscope-cache")
    parser.add_argument("--skip-decontamination", action="store_true")
    parser.add_argument("--training-parquet", action="append", default=[])
    parser.add_argument(
        "--training-hash-cache",
        default=str(PROJECT_ROOT / ".runtime/benchmark_cache/training-text-hashes-v2.1.pkl"),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    counts, seed = config["counts"], int(config["seed"])
    default_output = f"/mnt/workspace/zhaozetao/multimodel/Omni/benchmark-data/{config['name']}"
    output = Path(args.output or default_output).resolve()
    media = output / "media"
    output.mkdir(parents=True, exist_ok=True)

    from modelscope.msdatasets import MsDataset

    def load(name: str, split: str, subset: str | None = None):
        # Prefer already materialized Arrow data. MsDataset otherwise performs
        # a Hub metadata request even when every byte is present locally,
        # which makes an offline/restricted benchmark unnecessarily brittle.
        cache_root = Path(args.cache)
        namespace, dataset_name = name.split("/", 1)
        normalize = lambda value: re.sub(r"[^a-z0-9]", "", value.lower())
        roots = [
            path for path in cache_root.iterdir() if path.is_dir()
            and normalize(namespace) in normalize(path.name)
            and normalize(dataset_name) in normalize(path.name)
        ] if cache_root.is_dir() else []
        arrows = [
            path for root in roots for path in root.rglob(f"*-{split}.arrow")
            if not subset or normalize(subset) in normalize(str(path.parent))
        ]
        if len(arrows) == 1:
            from datasets import Dataset
            return Dataset.from_file(str(arrows[0]))
        return MsDataset.load(name, subset_name=subset, split=split, cache_dir=args.cache)

    def metadata(capability: str, output_modality: str, **extra):
        return {
            "capability": capability,
            "output_modality": output_modality,
            "require_audio": output_modality in {"audio", "text_audio"},
            **extra,
        }

    generation_limits = {
        "mcq": 32, "binary_mcq": 32, "vqa": 64, "wer": 256,
        "tts": 256, "em": 64, "f1": 64, "constraint": 64,
    }

    def task_metadata(capability: str, output_modality: str, metric: str, **extra):
        return metadata(
            capability, output_modality,
            max_new_tokens=generation_limits[metric], **extra,
        )

    rows: list[dict[str, Any]] = []

    # Use ARC rows only as a source pool for spoken QA; pure text regression is not benchmarked.
    arc_count = max(counts["spoken_qa"] * 3, counts["spoken_qa"] + 20)
    arc_all = stable_rows(load(config["sources"]["arc_easy"], "test", "ARC-Easy"), arc_count, seed)

    spoken = []
    for base in arc_all:
        choices = list(base["choices"]["text"])
        prompt = mcq_prompt(base["question"], choices)
        if len(prompt.split()) > 70:
            continue
        index = len(spoken)
        voice, speed = (("slt", 0.92), ("slt", 1.08), ("rms", 0.92), ("rms", 1.08))[index % 4]
        audio = synthesize(prompt, media / "spoken_qa" / f"{index:04d}.wav", voice, speed)
        if duration_seconds(audio) <= 30:
            spoken.append((prompt, choices, canonical_arc_answer(base["answerKey"], choices), audio))
        if len(spoken) == counts["spoken_qa"]:
            break
    if len(spoken) < counts["spoken_qa"]:
        raise RuntimeError(f"only {len(spoken)} spoken questions fit the duration limit")
    for i, (prompt, choices, answer, audio) in enumerate(spoken):
        common = {
            "metric": "mcq", "prompt": "", "reference": answer, "choices": choices, "audio": audio,
            "source": "allenai/ai2_arc:ARC-Easy:test+flite",
        }
        rows.append({
            **common, "sample_id": f"spoken-a2t-{i:04d}", "task": "spoken_qa_a2t",
            "metadata": task_metadata("speech_understanding", "text", "mcq", transcript_prompt=prompt),
        })

    # Natural speech for ASR and disjoint target sentences/natural labels for TTS.
    tts_remaining = counts["tts"]
    for subset, count, offset in (
        ("clean", counts["librispeech_clean"], 10),
        ("other", counts["librispeech_other"], 11),
    ):
        dataset = load(config["sources"]["librispeech"], "test", subset)
        extra = tts_remaining if subset == "clean" else 0
        candidates = stable_rows(dataset, count + extra + 100, seed + offset)
        accepted = 0
        tts_candidates = []
        for source_row in candidates:
            audio_value = source_row.get("audio") or source_row.get("file")
            path = save_audio(audio_value, media / "librispeech" / subset / f"{accepted:04d}.flac")
            if duration_seconds(path) > 30:
                Path(path).unlink(missing_ok=True)
                continue
            text = source_row.get("text") or source_row.get("transcription") or source_row.get("sentence")
            if accepted < count:
                rows.append({
                    "sample_id": f"asr-{subset}-{accepted:04d}", "task": f"asr_{subset}",
                    "metric": "wer", "prompt": "Transcribe the speech exactly in English.",
                    "reference": text, "audio": path,
                    "source": f"openslr/librispeech_asr:{subset}:test",
                    "metadata": task_metadata(
                        "asr", "text", "wer", input_duration_seconds=duration_seconds(path),
                        decontamination_text=text,
                    ),
                })
            elif subset == "clean" and len(tts_candidates) < tts_remaining:
                tts_candidates.append((text, path))
            accepted += 1
            if accepted >= count and (subset != "clean" or len(tts_candidates) >= tts_remaining):
                break
        if accepted < count:
            raise RuntimeError(f"LibriSpeech {subset} supplied only {accepted}/{count} samples")
        if subset == "clean":
            for i, (text, label_audio) in enumerate(tts_candidates):
                spoken_text = naturalize_tts_text(text)
                rows.append({
                    "sample_id": f"tts-{i:04d}", "task": "tts", "metric": "tts",
                    "prompt": f"Read the following text exactly and do not add commentary:\n{spoken_text}",
                    "reference": spoken_text, "source": "openslr/librispeech_asr:clean:test",
                    "metadata": task_metadata(
                        "speech_generation", "audio", "tts", target_text=spoken_text,
                        label_audio=label_audio, decontamination_text=text,
                    ),
                })
            tts_remaining -= len(tts_candidates)
    if tts_remaining:
        raise RuntimeError(f"missing {tts_remaining} TTS target sentences")

    visual_for_av: list[dict[str, Any]] = []

    # MMBench-EN: general image understanding.
    dataset_id, split = config["sources"]["mmbench_en"], "dev"
    dataset = load(dataset_id, split)
    accepted = 0
    for row in stable_iter(dataset, seed + 20):
        choices = [str(row.get(letter)) for letter in "ABCD" if str(row.get(letter, "nan")) != "nan"]
        prompt = mcq_prompt(str(row["question"]), choices, str(row.get("hint", "")))
        if len(prompt.split()) > 90:
            continue
        image = save_image(row.get("image") or row.get("image_path"), media / "mmbench" / f"{accepted:04d}.jpg")
        visual_row = {
            "sample_id": f"vision-mmbench-{accepted:04d}", "task": "vision_mmbench",
            "metric": "mcq", "prompt": prompt, "reference": str(row["answer"]).strip().upper(),
            "choices": choices, "image": image, "source": f"{dataset_id}:{split}",
            "metadata": task_metadata(
                "vision", "text", "mcq", source_id=str(row.get("index") or row.get("id") or accepted),
            ),
        }
        rows.append(visual_row)
        visual_for_av.append(visual_row)
        accepted += 1
        if accepted == counts["mmbench_en"]:
            break
    if accepted < counts["mmbench_en"]:
        raise RuntimeError(f"mmbench supplied only {accepted}/{counts['mmbench_en']} samples")
    del dataset

    # POPE: balance adversarial/popular/random and Yes/No rather than scoring
    # only the adversarial subset.
    dataset_id, split = config["sources"]["pope"], "test"
    dataset = load(dataset_id, split)
    strategies = ("adversarial", "popular", "random")
    desired = Counter()
    for index in range(counts["pope"]):
        # Alternating labels while cycling strategies keeps both dimensions
        # balanced even when the requested count is not divisible by six.
        desired[(strategies[index % len(strategies)], "yes" if index % 2 == 0 else "no")] += 1
    accepted = 0
    for source_index, source_row in enumerate(stable_iter(dataset, seed + 21)):
        for strategy in strategies:
            items = list(source_row.get(strategy, []))
            random.Random(seed + source_index * 17 + len(strategy)).shuffle(items)
            for item in items:
                raw = str(item.get("answer") or item.get("label")).lower()
                label = "yes" if raw in {"yes", "1", "true"} else "no"
                key = (strategy, label)
                if desired[key] <= 0:
                    continue
                image = save_image(source_row["image"], media / "pope" / f"{accepted:04d}.jpg")
                choices = ["Yes", "No"]
                visual_row = {
                    "sample_id": f"vision-pope-{accepted:04d}", "task": "vision_pope",
                    "metric": "binary_mcq", "prompt": mcq_prompt(str(item["question"]), choices),
                    "reference": "A" if label == "yes" else "B", "choices": choices,
                    "image": image, "source": f"{dataset_id}:{split}:{strategy}",
                    "metadata": task_metadata(
                        "vision", "text", "binary_mcq", pope_strategy=strategy,
                        source_id=f"{source_index}:{strategy}:{item['question']}",
                    ),
                }
                rows.append(visual_row)
                visual_for_av.append(visual_row)
                desired[key] -= 1
                accepted += 1
                break
        if accepted == counts["pope"]:
            break
    if accepted < counts["pope"]:
        raise RuntimeError(f"POPE supplied only {accepted}/{counts['pope']} balanced samples")
    del dataset

    # TextVQA: use fully annotated validation parquet. The former lite source
    # contained images only and is intentionally rejected by the validator.
    textvqa_dir = Path(config["sources"]["textvqa_local_dir"]).expanduser().resolve()
    textvqa_candidate_count = counts["textvqa"] + (
        max(16, counts["textvqa"] // 4) if not args.skip_decontamination else 0
    )
    for accepted, row in enumerate(textvqa_rows(textvqa_dir, textvqa_candidate_count, seed + 22)):
        answers = [str(answer).strip() for answer in row["answers"] if str(answer).strip()]
        prompt = str(row["question"]).strip() + "\nAnswer using a short phrase only."
        image = save_image(row["image"], media / "textvqa" / f"{accepted:04d}.jpg")
        rows.append({
            "sample_id": f"vision-textvqa-{accepted:04d}", "task": "vision_textvqa",
            "metric": "vqa", "prompt": prompt, "reference": answers, "choices": [],
            "image": image, "source": "lmms-lab-encoder/textvqa:validation",
            "metadata": task_metadata(
                "vision", "text", "vqa",
                source_id=str(row.get("question_id") or f"textvqa-{accepted}"),
            ),
        })

    accepted = 0
    for base in visual_for_av:
        question_audio = synthesize(
            base["prompt"], media / "audio_vision" / f"{accepted:04d}.wav",
            ("slt", "rms")[accepted % 2], (0.95, 1.05)[accepted % 2],
        )
        input_duration = duration_seconds(question_audio)
        if input_duration > 30:
            Path(question_audio).unlink(missing_ok=True)
            Path(question_audio + ".sha256").unlink(missing_ok=True)
            continue
        rows.append({
            "sample_id": f"audio-vision-{accepted:04d}", "task": "audio_vision", "metric": base["metric"],
            "prompt": "", "reference": base["reference"], "choices": base["choices"],
            "image": base["image"], "audio": question_audio, "source": base["source"] + "+flite",
            "metadata": task_metadata(
                "cross_modal", "text", base["metric"], paired_id=base["sample_id"],
                transcript_prompt=base["prompt"], input_duration_seconds=input_duration,
            ),
        })
        accepted += 1
        if accepted == counts["audio_vision"]:
            break
    if accepted < counts["audio_vision"]:
        raise RuntimeError("not enough visual MCQs for the paired audio-vision set")

    nlvr_split = config.get("splits", {}).get("nlvr2", "balanced_dev")
    nlvr = load(config["sources"]["nlvr2"], nlvr_split)
    accepted = 0
    for row in stable_iter(nlvr, seed + 30):
        first = row.get("image0") or row.get("image_0") or row.get("left_image")
        second = row.get("image1") or row.get("image_1") or row.get("right_image")
        if first is None or second is None:
            continue
        images = [
            save_image(first, media / "nlvr2" / f"{accepted:04d}-0.jpg"),
            save_image(second, media / "nlvr2" / f"{accepted:04d}-1.jpg"),
        ]
        choices = ["True", "False"]
        raw = str(row.get("label") or row.get("answer")).lower()
        reference = "A" if raw in {"true", "1", "yes"} else "B"
        statement = str(row.get("sentence") or row.get("question") or row.get("text"))
        rows.append({
            "sample_id": f"multi-image-{accepted:04d}", "task": "multi_image", "metric": "mcq",
            "prompt": mcq_prompt(statement, choices), "reference": reference, "choices": choices,
            "images": images, "source": f"{config['sources']['nlvr2']}:{nlvr_split}",
            "metadata": task_metadata(
                "cross_modal", "text", "mcq",
                source_id=str(row.get("identifier") or row.get("id") or accepted),
            ),
        })
        accepted += 1
        if accepted == counts["nlvr2"]:
            break
    if accepted < counts["nlvr2"]:
        raise RuntimeError(f"NLVR2 supplied only {accepted}/{counts['nlvr2']} samples")

    defaults = {
        "choices": [], "image": None, "images": [], "audio": None,
        "perturbation": "clean", "metadata": {},
    }
    training_paths = args.training_parquet or [
        str(PROJECT_ROOT / "dataset" / "raw" / "legacy" / name)
        for name in ("sft_t2a.parquet", "sft_a2a.parquet", "sft_i2t.parquet")
    ]
    if args.skip_decontamination:
        seen, decontamination_scan = set(), {"skipped": True}
    else:
        seen, decontamination_scan = training_hashes(
            training_paths, Path(args.training_hash_cache).resolve(),
        )
    from benchmark.validate_manifest import expected_task_counts, validate_manifest

    desired_counts = expected_task_counts(config)
    selected_counts: Counter[str] = Counter()
    clean_rows, rejected = [], []
    for row in rows:
        complete = {**defaults, **row}
        decontamination_text = complete["metadata"].get(
            "decontamination_text",
            complete["metadata"].get("transcript_prompt", complete["prompt"]),
        )
        if not str(decontamination_text or "").strip():
            raise ValueError(f"sample {complete['sample_id']} has no decontamination text")
        if seen and content_hashes(str(decontamination_text)) & seen:
            rejected.append(complete["sample_id"])
        elif selected_counts[complete["task"]] < desired_counts[complete["task"]]:
            clean_rows.append(complete)
            selected_counts[complete["task"]] += 1

    manifest = output / "manifest.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in clean_rows)
    manifest.write_text(payload, encoding="utf-8")
    summary = {
        "name": config["name"], "profile": config.get("profile", "standard"), "seed": seed,
        "builder_version": BUILDER_VERSION,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "samples": len(clean_rows), "task_counts": Counter(row["task"] for row in clean_rows),
        "decontaminated_against_training": not args.skip_decontamination,
        "decontamination_scan": decontamination_scan,
        "rejected_training_overlap": rejected,
        "manifest_sha256": hashlib.sha256(payload.encode()).hexdigest(),
        "spoken_prompt_tts": {
            "engine": "FFmpeg flite", "voices": ["slt", "rms"],
            "speed_factors": [0.92, 0.95, 1.05, 1.08],
        },
    }
    (output / "manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    validate_manifest(
        manifest,
        metadata_path=output / "manifest.json",
        expected_builder=BUILDER_VERSION,
        config_path=config_path,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
