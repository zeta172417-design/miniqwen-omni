from __future__ import annotations

import math
import random
import re
import statistics
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


CAPABILITY_WEIGHTS = {
    "asr": 0.20,
    "speech_understanding": 0.20,
    "speech_generation": 0.20,
    "vision": 0.20,
    "cross_modal": 0.20,
}

REQUIRED_TASKS = {
    "asr": ("asr_clean", "asr_other"),
    "speech_understanding": ("spoken_qa_a2t", "spoken_qa_a2a"),
    "speech_generation": ("tts",),
    "vision": ("vision_mmbench", "vision_pope", "vision_textvqa"),
    "cross_modal": ("audio_vision", "multi_image"),
}

LEGACY_CAPABILITIES = {
    "text": "text_retention",
    "audio": "speech_understanding",
    "vision": "vision",
    "audio_vision": "cross_modal",
    "robustness": "speech_understanding",
}

SCORER_VERSION = "2.3.0"


def capability_name(row: dict[str, Any]) -> str:
    configured = row.get("metadata", {}).get("capability")
    if configured:
        return str(configured)
    task = str(row.get("task", ""))
    if task.startswith("asr_"):
        return "asr"
    if task.startswith("spoken_qa_"):
        return "speech_understanding"
    if task == "tts":
        return "speech_generation"
    if task.startswith("vision_"):
        return "vision"
    if task in {"audio_vision", "multi_image"}:
        return "cross_modal"
    if task == "text_regression":
        return "text_retention"
    return LEGACY_CAPABILITIES.get(task, task)


def normalize_vqa_text(value: Any) -> str:
    text = str(value or "").lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    }
    text = " ".join(number_words.get(token, token) for token in text.split())
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


# Backward-compatible public name used by data decontamination and tests.
normalize_text = normalize_vqa_text


def normalize_asr_text(value: Any) -> str:
    """LibriSpeech-style normalization without deleting lexical articles."""
    text = str(value or "").lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    return " ".join(text.split())


def naturalize_tts_text(value: Any) -> str:
    """Turn corpus-style all-caps transcripts into ordinary written English.

    LibriSpeech distributes transcripts in uppercase.  Passing those strings
    verbatim to a speech generator can make some models pronounce every letter
    as an initialism, even though case is not an audible part of the target.
    """
    text = " ".join(str(value or "").strip().split())
    if text and text == text.upper() and any(character.isalpha() for character in text):
        text = text.lower()
        text = re.sub(r"\bi\b", "I", text)
        first_alpha = next((index for index, character in enumerate(text) if character.isalpha()), None)
        if first_alpha is not None:
            text = text[:first_alpha] + text[first_alpha].upper() + text[first_alpha + 1:]
    if text and text[-1] not in ".!?":
        text += "."
    return text


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
    hyp = normalize_asr_text(prediction).split()
    return min(
        (edit_distance(normalize_asr_text(ref).split(), hyp) / max(len(normalize_asr_text(ref).split()), 1) for ref in refs),
        default=1.0,
    )


def character_error_rate(prediction: str, reference: Any) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    hyp = list(normalize_asr_text(prediction).replace(" ", ""))
    return min(
        edit_distance(list(normalize_asr_text(ref).replace(" ", "")), hyp)
        / max(len(normalize_asr_text(ref).replace(" ", "")), 1)
        for ref in refs
    )


def word_edit_similarity(prediction: str, reference: Any) -> float:
    """Bounded edit similarity that remains discriminative when WER exceeds 100%."""
    refs = reference if isinstance(reference, list) else [reference]
    hyp = normalize_asr_text(prediction).split()
    return max((
        1.0 - edit_distance(normalize_asr_text(ref).split(), hyp)
        / max(len(normalize_asr_text(ref).split()), len(hyp), 1)
        for ref in refs
    ), default=0.0)


def _contains_token_span(container: list[str], candidate: list[str]) -> bool:
    return bool(candidate) and any(
        container[index:index + len(candidate)] == candidate
        for index in range(len(container) - len(candidate) + 1)
    )


def extract_explicit_choice(text: str, choices: list[str]) -> str | None:
    """Parse a letter only when the response clearly marks it as an answer."""
    valid = [chr(ord("A") + index) for index in range(len(choices))]
    patterns = (
        r"\b(?:final\s+)?(?:correct\s+)?(?:answer|option|choice|response|caption)"
        r"(?:\s+(?:to|for|from)\s+(?:the\s+)?(?:question|image|given\s+choices))?"
        r"\s*(?:is|would\s+be|:|=|-)?\s*[\(\[]?([A-Z])[\)\]]?\b",
        r"\b(?:choose|select|pick)\s*(?:option|choice)?\s*[\(\[]?([A-Z])[\)\]]?\b",
        r"\b([A-Z])\s+(?:is|would\s+be)\s+(?:the\s+)?(?:correct|best)\s*(?:answer|option|choice)?\b",
    )
    for pattern in patterns:
        match = re.search(pattern, str(text), flags=re.IGNORECASE)
        if match and match.group(1).upper() in valid:
            return match.group(1).upper()

    # Markdown, quotes, bullets and parentheses may precede a concise answer.
    leading = re.sub(r"^[\s>*_`#'\"-]+", "", str(text))
    match = re.match(
        r"[\(\[]?\s*([A-Z])\s*[\)\]]?(?:[\s).,:;\-]|$)",
        leading,
        flags=re.IGNORECASE,
    )
    if match and match.group(1).upper() in valid:
        return match.group(1).upper()
    return None


def _choice_from_unique_text(text: str, choices: list[str]) -> str | None:
    """Map a verbose answer to an option when exactly one choice is stated."""
    prediction = normalize_vqa_text(text).split()
    normalized_choices = [normalize_vqa_text(choice).split() for choice in choices]
    matched = [
        index for index, choice in enumerate(normalized_choices)
        if _contains_token_span(prediction, choice)
    ]
    # Drop a shorter choice wholly contained in another matched choice, e.g.
    # "cat" when the response uniquely states the option "black cat".
    maximal = []
    for index in matched:
        choice = normalized_choices[index]
        if any(
            index != other
            and len(normalized_choices[other]) > len(choice)
            and _contains_token_span(normalized_choices[other], choice)
            for other in matched
        ):
            continue
        maximal.append(index)
    return chr(65 + maximal[0]) if len(maximal) == 1 else None


def _choice_from_isolated_letter(text: str, choices: list[str]) -> str | None:
    """Conservative fallback for one standalone capital option letter."""
    valid = set(chr(65 + index) for index in range(len(choices)))
    matches = [
        match for match in re.finditer(r"(?<![A-Za-z0-9°-])([A-Z])(?![A-Za-z0-9-])", str(text))
        if match.group(1) in valid
    ]
    letters = {match.group(1) for match in matches}
    if len(letters) != 1:
        return None
    match = matches[0]
    # Scan from the beginning as a fallback, but only accept a letter near the
    # start of the response. A later isolated A/B/C/D is often an English
    # article, physical unit, quoted label, or the start of an option list.
    prefix = str(text)[:match.start()].strip()
    suffix = str(text)[match.end():]
    if str(text)[:match.start()].count('"') % 2:
        return None
    if match.group(1) == "A" and re.match(r"\s+[A-Za-z]", suffix):
        return None
    return match.group(1) if len(prefix) <= 24 else None


def extract_choice(text: str, choices: list[str]) -> str | None:
    """Extract MCQ intent without requiring letter-only model output.

    Precedence is explicit option syntax, one uniquely stated complete choice,
    then one unambiguous standalone uppercase option letter.
    """
    explicit = extract_explicit_choice(text, choices)
    if explicit is not None:
        return explicit
    choice_text = _choice_from_unique_text(text, choices)
    if choice_text is not None:
        return choice_text
    return _choice_from_isolated_letter(text, choices)


def extract_spoken_choice(text: str, choices: list[str]) -> str | None:
    """Parse an ASR transcript of a spoken option letter.

    Homophones are accepted only when the whole transcript is one word, so a
    normal sentence containing "see" or "be" cannot become an option answer.
    """
    parsed = extract_choice(text, choices)
    if parsed is not None:
        return parsed
    token = normalize_asr_text(text)
    spoken_letters = {
        "a": "A", "ay": "A",
        "b": "B", "be": "B", "bee": "B",
        "c": "C", "sea": "C", "see": "C",
        "d": "D", "dee": "D",
    }
    choice = spoken_letters.get(token)
    return choice if choice and ord(choice) - 65 < len(choices) else None


def multiple_choice_accuracy(text: str, reference: Any, choices: list[str]) -> float:
    predicted = extract_choice(text, choices)
    expected = str(reference).strip().upper()
    if expected.isdigit():
        numeric = int(expected)
        # Public MCQ datasets use both 0-based (PIQA) and 1-based (ARC)
        # labels. New manifests canonicalize labels to letters; this fallback
        # keeps legacy rows safe and never creates an out-of-range option.
        index = 0 if numeric == 0 else numeric - 1
        expected = chr(ord("A") + index)
    return float(predicted == expected)


def semantic_choice_accuracy(text: str, reference: Any, choices: list[str]) -> float:
    """Backward-compatible public name for semantic option accuracy."""
    return multiple_choice_accuracy(text, reference, choices)


def vqa_accuracy(text: str, reference: Any) -> float:
    refs = reference if isinstance(reference, list) else [reference]
    prediction = normalize_text(text).split()
    # Preserve the annotation-consensus rule while allowing a short answer to
    # occur as a complete token span inside a generated explanation.
    count = sum(
        _contains_token_span(prediction, normalize_text(ref).split())
        for ref in refs
    )
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
    output = {
        "audio_valid": 0.0, "duration_seconds": 0.0, "clipping_rate": 1.0,
        "silence_rate": 1.0, "rms": 0.0,
    }
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
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        silence = float(np.mean(np.abs(audio) < 1e-4)) if audio.size else 1.0
        output.update({
            "audio_valid": float(finite and duration >= 0.1 and rms >= 1e-4 and silence < 0.995),
            "duration_seconds": float(duration),
            "clipping_rate": float(np.mean(np.abs(audio) >= 0.999)) if audio.size else 1.0,
            "silence_rate": silence,
            "rms": rms,
        })
    except Exception:
        pass
    return output


def score_sample(sample: dict[str, Any], result: dict[str, Any]) -> dict[str, float]:
    if result.get("status") == "unsupported":
        return {}
    if result.get("status") != "ok":
        return {"primary": 0.0}
    prediction = result.get("generated_text", "")
    metric = sample["metric"]
    if metric in {"mcq", "binary_mcq"}:
        primary = multiple_choice_accuracy(prediction, sample["reference"], sample.get("choices", []))
        parsed = extract_choice(prediction, sample.get("choices", []))
        explicit = extract_explicit_choice(prediction, sample.get("choices", []))
        return {
            "primary": primary,
            # Strict instruction following is diagnostic only; primary accuracy
            # uses the robust semantic parser above.
            "format_compliance": float(explicit is not None),
            "strict_format_compliance": float(explicit is not None),
            "answer_parse_rate": float(parsed is not None),
            "semantic_choice": primary,
        }
    elif metric == "em":
        primary = exact_match(prediction, sample["reference"])
    elif metric == "f1":
        primary = token_f1(prediction, sample["reference"])
    elif metric == "wer":
        primary = word_edit_similarity(prediction, sample["reference"])
    elif metric == "vqa":
        primary = vqa_accuracy(prediction, sample["reference"])
    elif metric == "constraint":
        primary = constraint_accuracy(prediction, sample["reference"])
    elif metric == "tts":
        # Filled from an independent ASR pass in evaluate_audio.py.
        primary = 0.0
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


def _pope_f1(rows: list[dict[str, Any]]) -> float:
    tp = fp = fn = 0
    for row in rows:
        pred = extract_choice(row.get("generated_text", ""), row.get("choices", []))
        ref = str(row.get("reference", "")).upper()
        tp += pred == "A" and ref == "A"
        fp += pred == "A" and ref == "B"
        fn += pred != "A" and ref == "A"
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def paired_fixed_omni_comparison(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    seed: int = 20260913,
    rounds: int = 2000,
) -> dict[str, Any]:
    """Paired, task-stratified comparison using Fixed Omni weighting.

    The same sampled IDs are used for both models in every bootstrap draw.
    Text regression is excluded, task sample counts cannot change their
    weights, and POPE retains its standard F1 aggregation.
    """
    required = {task for tasks in REQUIRED_TASKS.values() for task in tasks}
    left = {row["sample_id"]: row for row in left_rows if row.get("task") in required}
    right = {row["sample_id"]: row for row in right_rows if row.get("task") in required}
    paired_by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for sample_id in sorted(left.keys() & right.keys()):
        if left[sample_id].get("task") == right[sample_id].get("task"):
            paired_by_task[str(left[sample_id]["task"])].append((left[sample_id], right[sample_id]))

    def row_score(row: dict[str, Any]) -> float:
        if row.get("status") == "unsupported":
            return 0.0
        return float(row.get("scores", {}).get("primary", 0.0))

    def model_score(selected: dict[str, list[dict[str, Any]]]) -> float:
        capabilities = []
        for capability, tasks in REQUIRED_TASKS.items():
            task_scores = []
            for task in tasks:
                rows = selected.get(task, [])
                if task == "vision_pope":
                    task_scores.append(_pope_f1(rows) if rows else 0.0)
                else:
                    task_scores.append(statistics.fmean(row_score(row) for row in rows) if rows else 0.0)
            capabilities.append(statistics.fmean(task_scores) * CAPABILITY_WEIGHTS[capability])
        return sum(capabilities) / sum(CAPABILITY_WEIGHTS.values())

    def delta(selection: dict[str, list[int]] | None = None) -> float:
        selected_left, selected_right = {}, {}
        for task in required:
            pairs = paired_by_task.get(task, [])
            indices = selection.get(task, []) if selection is not None else range(len(pairs))
            selected_left[task] = [pairs[index][0] for index in indices]
            selected_right[task] = [pairs[index][1] for index in indices]
        return model_score(selected_right) - model_score(selected_left)

    point = delta()
    rng = random.Random(seed)
    draws = []
    for _ in range(rounds):
        selection = {
            task: [rng.randrange(len(pairs)) for _ in pairs]
            for task, pairs in paired_by_task.items() if pairs
        }
        draws.append(delta(selection))
    draws.sort()
    low = draws[int(rounds * 0.025)] if draws else point
    high = draws[min(int(rounds * 0.975), rounds - 1)] if draws else point
    return {
        "delta": point,
        "ci95": [low, high],
        "samples": sum(len(pairs) for pairs in paired_by_task.values()),
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = defaultdict(list)
    capabilities: dict[str, list[float]] = defaultdict(list)
    latencies, amortized_latencies, memory, failures = [], [], [], 0
    unsupported = 0
    task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_task_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        all_task_rows[row["task"]].append(row)
        if row.get("status") == "unsupported":
            unsupported += 1
            continue
        score = float(row.get("scores", {}).get("primary", 0.0))
        groups[row["task"]].append(score)
        capabilities[capability_name(row)].append(score)
        task_rows[row["task"]].append(row)
        if row.get("latency_seconds") is not None:
            latencies.append(float(row["latency_seconds"]))
            metadata = row.get("metadata", {})
            batch_wall = metadata.get("batch_wall_seconds")
            native_batch = metadata.get("native_batch_size")
            if batch_wall is not None and native_batch:
                amortized_latencies.append(float(batch_wall) / max(int(native_batch), 1))
            else:
                amortized_latencies.append(float(row["latency_seconds"]))
        if row.get("peak_memory_gb") is not None:
            memory.append(float(row["peak_memory_gb"]))
        failures += row.get("status") != "ok"
    tasks = {}
    for task, values in sorted(groups.items()):
        low, high = bootstrap_ci(values)
        tasks[task] = {"score": statistics.fmean(values), "ci95": [low, high], "samples": len(values)}
        current = task_rows[task]
        if current and current[0].get("metric") == "wer":
            tasks[task]["wer"] = statistics.fmean(
                word_error_rate(row.get("generated_text", ""), row.get("reference")) for row in current
            )
            tasks[task]["cer"] = statistics.fmean(
                character_error_rate(row.get("generated_text", ""), row.get("reference")) for row in current
            )
            rtfs = [
                float(row["latency_seconds"]) / float(row["metadata"]["input_duration_seconds"])
                for row in current
                if row.get("latency_seconds") is not None
                and float(row.get("metadata", {}).get("input_duration_seconds", 0)) > 0
            ]
            if rtfs:
                tasks[task]["rtf"] = statistics.fmean(rtfs)
        audio_rows = [row for row in current if row.get("metadata", {}).get("require_audio")]
        if audio_rows:
            tasks[task]["audio_valid_rate"] = statistics.fmean(
                float(row.get("audio_diagnostics", {}).get("audio_valid", 0.0)) for row in audio_rows
            )
            consistency = [
                float(row["scores"]["speech_semantic"])
                for row in audio_rows if row.get("scores", {}).get("speech_semantic") is not None
            ]
            if consistency:
                tasks[task]["speech_consistency"] = statistics.fmean(consistency)
            mos = [float(row["utmos"]) for row in audio_rows if row.get("utmos") is not None]
            if mos:
                tasks[task]["utmos"] = statistics.fmean(mos)
            duration_ratios = [
                float(row["duration_ratio"]) for row in audio_rows
                if row.get("duration_ratio") is not None
            ]
            if duration_ratios:
                tasks[task]["duration_ratio"] = statistics.fmean(duration_ratios)
        if task == "spoken_qa_a2a":
            text_scores = [
                float(row["scores"]["text_primary"])
                for row in current if row.get("scores", {}).get("text_primary") is not None
            ]
            if text_scores:
                tasks[task]["text_answer_score"] = statistics.fmean(text_scores)
        if task == "tts":
            tts_wers = [float(row["tts_wer"]) for row in current if row.get("tts_wer") is not None]
            if tts_wers:
                tasks[task]["wer"] = statistics.fmean(tts_wers)
            tts_cers = [float(row["tts_cer"]) for row in current if row.get("tts_cer") is not None]
            if tts_cers:
                tasks[task]["cer"] = statistics.fmean(tts_cers)
            natural_wers = [
                float(row["natural_label_wer"]) for row in current
                if row.get("natural_label_wer") is not None
            ]
            if natural_wers:
                tasks[task]["natural_label_wer"] = statistics.fmean(natural_wers)
            natural_mos = [
                float(row["natural_label_utmos"]) for row in current
                if row.get("natural_label_utmos") is not None
            ]
            if natural_mos:
                tasks[task]["natural_label_utmos"] = statistics.fmean(natural_mos)
        compliance = [
            float(row.get("scores", {}).get("format_compliance")) for row in current
            if row.get("scores", {}).get("format_compliance") is not None
        ]
        if compliance:
            tasks[task]["format_compliance"] = statistics.fmean(compliance)
        semantic = [
            float(row.get("scores", {}).get("semantic_choice")) for row in current
            if row.get("scores", {}).get("semantic_choice") is not None
        ]
        if semantic:
            tasks[task]["semantic_choice_score"] = statistics.fmean(semantic)
        parseable = [
            float(row.get("scores", {}).get("answer_parse_rate")) for row in current
            if row.get("scores", {}).get("answer_parse_rate") is not None
        ]
        if parseable:
            tasks[task]["answer_parse_rate"] = statistics.fmean(parseable)
        strict_compliance = [
            float(row.get("scores", {}).get("strict_format_compliance")) for row in current
            if row.get("scores", {}).get("strict_format_compliance") is not None
        ]
        if strict_compliance:
            tasks[task]["strict_format_compliance"] = statistics.fmean(strict_compliance)
        if task == "vision_pope":
            tp = fp = fn = tn = 0
            for row in current:
                pred = extract_choice(row.get("generated_text", ""), row.get("choices", []))
                ref = str(row.get("reference", "")).upper()
                tp += pred == "A" and ref == "A"
                fp += pred == "A" and ref == "B"
                fn += pred != "A" and ref == "A"
                tn += pred == "B" and ref == "B"
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            tasks[task].update({
                "accuracy": tasks[task]["score"], "precision": precision, "recall": recall, "f1": f1,
            })
            tasks[task]["score"] = f1
    by_id = {
        row["sample_id"]: float(row.get("scores", {}).get("primary", 0.0))
        for row in rows if row.get("status") != "unsupported"
    }
    for task, current in task_rows.items():
        paired = [
            (by_id.get(row.get("metadata", {}).get("paired_id")), float(row.get("scores", {}).get("primary", 0.0)))
            for row in current if row.get("metadata", {}).get("paired_id")
        ]
        paired = [(base, score) for base, score in paired if base is not None]
        if paired and task in tasks:
            tasks[task].update({
                "paired_samples": len(paired),
                "paired_base_score": statistics.fmean(base for base, _ in paired),
                "paired_score": statistics.fmean(score for _, score in paired),
                "paired_degradation": statistics.fmean(base - score for base, score in paired),
            })
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
    # Capability scores are task-macro averages. This avoids counting the
    # paired A2T/A2A questions or a larger visual set more heavily merely
    # because it contains more samples.
    capability_metrics = {}
    fixed_capability_metrics = {}
    for capability, required_tasks in REQUIRED_TASKS.items():
        available_tasks = [task for task in required_tasks if task in tasks]
        if available_tasks:
            task_scores = [float(tasks[task]["score"]) for task in available_tasks]
            low, high = bootstrap_ci(task_scores)
            capability_metrics[capability] = {
                "score": statistics.fmean(task_scores),
                "ci95": [low, high],
                "tasks": available_tasks,
                "samples": sum(int(tasks[task]["samples"]) for task in available_tasks),
            }
        fixed_scores = []
        for task in required_tasks:
            task_all = all_task_rows.get(task, [])
            supported = sum(row.get("status") != "unsupported" for row in task_all)
            if task in tasks and task_all:
                fixed_scores.append(float(tasks[task]["score"]) * supported / len(task_all))
            else:
                fixed_scores.append(0.0)
        fixed_capability_metrics[capability] = {
            "score": statistics.fmean(fixed_scores),
            "ci95": list(bootstrap_ci(fixed_scores)),
            "tasks": list(required_tasks),
        }
    if "text_regression" in tasks:
        capability_metrics["text_retention"] = {
            **tasks["text_regression"], "tasks": ["text_regression"],
        }
    weighted = sum(
        fixed_capability_metrics[name]["score"] * weight
        for name, weight in CAPABILITY_WEIGHTS.items()
    )
    available_scores = [value["score"] for value in capability_metrics.values()]
    # Keep the mock/legacy text-only smoke test meaningful. Real v2 runs have
    # Omni capabilities and therefore exclude text_retention from the score.
    text_only_values = capabilities.get("text_retention", [])
    if not available_scores and text_only_values:
        weighted = statistics.fmean(text_only_values)
        fixed_score = weighted
    else:
        fixed_score = weighted / sum(CAPABILITY_WEIGHTS.values())
    return {
        "scorer_version": SCORER_VERSION,
        "objective_score": 100.0 * fixed_score,
        "fixed_omni_score": 100.0 * fixed_score,
        "available_omni_score": 100.0 * statistics.fmean(available_scores) if available_scores else 100.0 * fixed_score,
        "capability_metrics": capability_metrics,
        "fixed_capability_metrics": fixed_capability_metrics,
        "task_metrics": tasks,
        "failure_rate": failures / max(len(rows) - unsupported, 1),
        "unsupported_rate": unsupported / max(len(rows), 1),
        "unsupported_samples": unsupported,
        "supported_task_count": sum(
            any(row.get("status") != "unsupported" for row in all_task_rows.get(task, []))
            for tasks_in_capability in REQUIRED_TASKS.values() for task in tasks_in_capability
        ),
        "required_task_count": sum(len(value) for value in REQUIRED_TASKS.values()),
        "mean_latency_seconds": statistics.fmean(latencies) if latencies else None,
        "mean_amortized_latency_seconds": (
            statistics.fmean(amortized_latencies) if amortized_latencies else None
        ),
        "throughput_samples_per_second": (
            1.0 / statistics.fmean(amortized_latencies) if amortized_latencies else None
        ),
        "peak_memory_gb": max(memory) if memory else None,
        "samples": len(rows),
    }
