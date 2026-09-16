from __future__ import annotations

import math
import random
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


TASK_WEIGHTS = {
    "text": 0.10,
    "audio": 0.20,
    "vision": 0.20,
    "audio_vision": 0.15,
    "speech_semantic": 0.15,
    "speech_quality": 0.15,
    "robustness": 0.05,
}


def normalize_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: Any) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    return float(any(normalize_text(prediction) == normalize_text(ref) for ref in refs))


def token_f1(prediction: str, reference: Any) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    pred = normalize_text(prediction).split()
    scores = []
    for ref in refs:
        gold = normalize_text(ref).split()
        common = sum((Counter(pred) & Counter(gold)).values())
        if not pred or not gold:
            scores.append(float(pred == gold))
        elif not common:
            scores.append(0.0)
        else:
            precision, recall = common / len(pred), common / len(gold)
            scores.append(2 * precision * recall / (precision + recall))
    return max(scores, default=0.0)


def edit_distance(left: list[str], right: list[str]) -> int:
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1]


def word_error_rate(prediction: str, reference: Any) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    hyp = normalize_text(prediction).split()
    return min(
        (edit_distance(normalize_text(ref).split(), hyp) / max(len(normalize_text(ref).split()), 1) for ref in refs),
        default=1.0,
    )


def extract_choice(text: str, choices: list[str]) -> str | None:
    upper = text.upper().strip()
    valid = [chr(ord("A") + i) for i in range(len(choices))]
    patterns = [r"(?:ANSWER|OPTION|CHOICE)\s*(?:IS|:)?\s*([A-Z])\b", r"^\s*([A-Z])(?:[\).:\s]|$)"]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match and match.group(1) in valid:
            return match.group(1)
    normalized = normalize_text(text)
    exact = [valid[i] for i, choice in enumerate(choices) if normalize_text(choice) == normalized]
    return exact[0] if len(exact) == 1 else None


def multiple_choice_accuracy(text: str, reference: Any, choices: list[str]) -> float:
    predicted = extract_choice(text, choices)
    expected = str(reference).strip().upper()
    if expected.isdigit():
        expected = chr(ord("A") + int(expected))
    return float(predicted == expected)


def vqa_accuracy(text: str, reference: Any) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    prediction = normalize_text(text)
    count = sum(normalize_text(ref) == prediction for ref in refs)
    return min(count / 3.0, 1.0) if len(refs) >= 3 else float(count > 0)


def constraint_accuracy(text: str, specification: dict[str, Any]) -> float:
    checks = []
    words = text.split()
    if "max_words" in specification:
        checks.append(len(words) <= int(specification["max_words"]))
    if "min_words" in specification:
        checks.append(len(words) >= int(specification["min_words"]))
    for token in specification.get("must_include", []):
        checks.append(normalize_text(token) in normalize_text(text))
    for token in specification.get("must_not_include", []):
        checks.append(normalize_text(token) not in normalize_text(text))
    if specification.get("json"):
        import json
        try:
            json.loads(text)
            checks.append(True)
        except Exception:
            checks.append(False)
    return sum(checks) / max(len(checks), 1)


def audio_diagnostics(path: str | None) -> dict[str, float]:
    output = {"audio_valid": 0.0, "duration_seconds": 0.0, "clipping_rate": 1.0, "silence_rate": 1.0}
    if not path or not Path(path).is_file():
        return output
    try:
        import numpy as np
        import soundfile as sf
        audio, sample_rate = sf.read(path, always_2d=False)
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        duration = len(audio) / max(sample_rate, 1)
        finite = bool(audio.size and np.isfinite(audio).all())
        output.update({
            "audio_valid": float(finite and duration >= 0.1),
            "duration_seconds": float(duration),
            "clipping_rate": float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 1.0,
            "silence_rate": float(np.mean(np.abs(audio) < 1e-4)) if audio.size else 1.0,
        })
    except Exception:
        pass
    return output


def score_sample(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    if result.get("status") != "ok":
        return {"primary": 0.0}
    prediction = result.get("generated_text", "")
    metric = sample["metric"]
    if metric == "mcq":
        primary = multiple_choice_accuracy(prediction, sample["reference"], sample.get("choices", []))
    elif metric == "em":
        primary = exact_match(prediction, sample["reference"])
    elif metric == "f1":
        primary = token_f1(prediction, sample["reference"])
    elif metric == "wer":
        primary = 1.0 - min(word_error_rate(prediction, sample["reference"]), 1.0)
    elif metric == "vqa":
        primary = vqa_accuracy(prediction, sample["reference"])
    elif metric == "constraint":
        primary = constraint_accuracy(prediction, sample["reference"])
    else:
        raise ValueError(f"Unsupported metric: {metric}")
    return {"primary": primary}


def bootstrap_ci(values: Iterable[float], seed: int = 20260913, rounds: int = 2000) -> tuple[float, float]:
    values = list(values)
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choices(values, k=len(values))) for _ in range(rounds))
    return means[int(rounds * 0.025)], means[min(int(rounds * 0.975), rounds - 1)]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    latencies, memory, failures = [], [], 0
    speech_semantic, speech_quality = [], []
    for row in rows:
        score = float(row.get("scores", {}).get("primary", 0.0))
        groups[row["task"]].append(score)
        if row.get("latency_seconds") is not None:
            latencies.append(float(row["latency_seconds"]))
        if row.get("peak_memory_gb") is not None:
            memory.append(float(row["peak_memory_gb"]))
        failures += row.get("status") != "ok"
        if row.get("scores", {}).get("speech_semantic") is not None:
            speech_semantic.append(float(row["scores"]["speech_semantic"]))
        if row.get("scores", {}).get("speech_quality") is not None:
            speech_quality.append(float(row["scores"]["speech_quality"]))
    tasks = {}
    for task, values in sorted(groups.items()):
        low, high = bootstrap_ci(values)
        tasks[task] = {"score": statistics.fmean(values), "ci95": [low, high], "samples": len(values)}
    by_id = {row["sample_id"]: float(row.get("scores", {}).get("primary", 0.0)) for row in rows}
    paired = [
        (by_id.get(row.get("metadata", {}).get("base_id")), float(row.get("scores", {}).get("primary", 0.0)))
        for row in rows if row.get("task") == "robustness" and row.get("metadata", {}).get("base_id")
    ]
    paired = [(clean, noisy) for clean, noisy in paired if clean is not None]
    if paired:
        clean_score = statistics.fmean(clean for clean, _ in paired)
        noisy_score = statistics.fmean(noisy for _, noisy in paired)
        preservation = min(noisy_score / max(clean_score, 1e-12), 1.0) if clean_score else 0.0
        tasks["robustness"] = {
            "score": preservation, "ci95": list(bootstrap_ci([float(noisy >= clean) for clean, noisy in paired])),
            "samples": len(paired), "clean_score": clean_score, "perturbed_score": noisy_score,
        }
    for task, values in (("speech_semantic", speech_semantic), ("speech_quality", speech_quality)):
        if values:
            low, high = bootstrap_ci(values)
            tasks[task] = {"score": statistics.fmean(values), "ci95": [low, high], "samples": len(values)}
    weighted, available = 0.0, 0.0
    for task, weight in TASK_WEIGHTS.items():
        if task in tasks:
            weighted += tasks[task]["score"] * weight
            available += weight
    return {
        "objective_score": 100.0 * weighted / available if available else 0.0,
        "task_metrics": tasks,
        "failure_rate": failures / max(len(rows), 1),
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "peak_memory_gb": max(memory) if memory else None,
        "samples": len(rows),
    }
