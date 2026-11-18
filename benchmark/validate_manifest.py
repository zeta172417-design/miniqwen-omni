#!/usr/bin/env python3
"""Fail fast on malformed, stale, or incomplete Omni benchmark manifests."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_task_counts(config: dict[str, Any]) -> dict[str, int]:
    counts = config["counts"]
    return {
        "spoken_qa_a2t": int(counts["spoken_qa"]),
        "asr_clean": int(counts["librispeech_clean"]),
        "asr_other": int(counts["librispeech_other"]),
        "tts": int(counts["tts"]),
        "vision_mmbench": int(counts["mmbench_en"]),
        "vision_pope": int(counts["pope"]),
        "vision_textvqa": int(counts["textvqa"]),
        "audio_vision": int(counts["audio_vision"]),
        "multi_image": int(counts["nlvr2"]),
    }


def validate_manifest(
    manifest_path: str | Path,
    metadata_path: str | Path | None = None,
    expected_builder: str | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    manifest = Path(manifest_path).resolve()
    metadata_file = Path(metadata_path).resolve() if metadata_path else manifest.with_name("manifest.json")
    rows = read_jsonl(manifest)
    errors: list[str] = []
    if not rows:
        errors.append("manifest contains no samples")
    ids = [str(row.get("sample_id") or "") for row in rows]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if not sample_id or count > 1]
    if duplicates:
        errors.append(f"empty or duplicate sample IDs: {duplicates[:8]}")
    by_id = {row.get("sample_id"): row for row in rows}

    for row in rows:
        sample_id = str(row.get("sample_id") or "<missing>")
        metric = row.get("metric")
        prompt = row.get("prompt")
        metadata = row.get("metadata") or {}
        effective_prompt = metadata.get("transcript_prompt") or prompt
        if not isinstance(prompt, str) or not str(effective_prompt or "").strip():
            errors.append(f"{sample_id}: empty prompt/transcript_prompt")
        reference = row.get("reference")
        if reference is None or reference == "" or reference == []:
            errors.append(f"{sample_id}: empty reference")
        if metric in {"mcq", "binary_mcq"}:
            choices = row.get("choices") or []
            valid = [chr(65 + index) for index in range(len(choices))]
            if len(choices) < 2 or str(reference).upper() not in valid:
                errors.append(f"{sample_id}: invalid MCQ reference {reference!r} for {len(choices)} choices")
        if metric == "vqa":
            if not isinstance(reference, list) or not any(str(value).strip() for value in reference):
                errors.append(f"{sample_id}: VQA requires a non-empty answer list")

        media_paths = []
        if row.get("image"):
            media_paths.append(row["image"])
        media_paths.extend(row.get("images") or [])
        if row.get("audio"):
            media_paths.append(row["audio"])
        if metadata.get("label_audio"):
            media_paths.append(metadata["label_audio"])
        for value in media_paths:
            path = Path(value)
            if not path.is_absolute():
                path = manifest.parent / path
            if not path.is_file():
                errors.append(f"{sample_id}: missing media {path}")
        if row.get("task") == "multi_image" and len(row.get("images") or []) < 2:
            errors.append(f"{sample_id}: multi_image requires at least two images")
        if row.get("audio"):
            try:
                import soundfile as sf
                audio_path = Path(row["audio"])
                if not audio_path.is_absolute():
                    audio_path = manifest.parent / audio_path
                seconds = float(sf.info(audio_path).duration)
                if seconds <= 0.1 or seconds > 30.0:
                    errors.append(f"{sample_id}: input audio duration {seconds:.2f}s is outside (0.1, 30]")
            except Exception as exc:
                errors.append(f"{sample_id}: unreadable audio: {exc}")
        paired_id = metadata.get("paired_id")
        if paired_id:
            base = by_id.get(paired_id)
            if base is None:
                errors.append(f"{sample_id}: missing paired sample {paired_id}")
            elif base.get("reference") != reference:
                errors.append(f"{sample_id}: paired sample reference mismatch")
        if metadata.get("max_new_tokens") is None:
            errors.append(f"{sample_id}: max_new_tokens is not declared")

    metadata: dict[str, Any] = {}
    if metadata_file.is_file():
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        if metadata.get("manifest_sha256") != digest:
            errors.append("manifest SHA256 does not match manifest.json")
        if int(metadata.get("samples", -1)) != len(rows):
            errors.append("manifest sample count does not match manifest.json")
        actual_counts = Counter(row.get("task") for row in rows)
        if dict(metadata.get("task_counts", {})) != dict(actual_counts):
            errors.append("task counts do not match manifest.json")
        if expected_builder and metadata.get("builder_version") != expected_builder:
            errors.append(
                f"stale builder version {metadata.get('builder_version')!r}; expected {expected_builder!r}"
            )
        if config_path:
            config_file = Path(config_path).resolve()
            config_digest = hashlib.sha256(config_file.read_bytes()).hexdigest()
            if metadata.get("config_sha256") != config_digest:
                errors.append("manifest was built from a different benchmark config")
            configured_counts = expected_task_counts(json.loads(config_file.read_text(encoding="utf-8")))
            actual_counts = dict(Counter(row.get("task") for row in rows))
            if actual_counts != configured_counts:
                errors.append(
                    f"task counts differ from config: actual={actual_counts}, expected={configured_counts}"
                )
        if metadata.get("profile") == "standard" and not metadata.get("decontaminated_against_training"):
            errors.append("standard manifest must be decontaminated against training data")
    elif expected_builder:
        errors.append(f"missing manifest metadata: {metadata_file}")

    if errors:
        preview = "\n".join(f"- {error}" for error in errors[:30])
        suffix = f"\n- ... and {len(errors) - 30} more" if len(errors) > 30 else ""
        raise ValueError(f"invalid benchmark manifest ({len(errors)} errors):\n{preview}{suffix}")
    return {
        "manifest": str(manifest), "samples": len(rows),
        "tasks": dict(Counter(row["task"] for row in rows)),
        "builder_version": metadata.get("builder_version"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--expected-builder", default="2.3.0")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    result = validate_manifest(args.manifest, args.metadata, args.expected_builder, args.config)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
