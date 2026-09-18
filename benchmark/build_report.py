#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import statistics
import sys
from pathlib import Path

import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import SCORER_VERSION, paired_fixed_omni_comparison


LABELS = {
    "asr": "ASR",
    "speech_understanding": "Speech understanding",
    "speech_generation": "Speech generation / TTS",
    "vision": "Vision understanding",
    "cross_modal": "Cross-modal / multi-image",
    "text_retention": "Text retention (not in Omni score)",
    "asr_clean": "ASR test-clean",
    "asr_other": "ASR test-other",
    "spoken_qa_a2t": "Spoken QA: audio to text",
    "spoken_qa_a2a": "Spoken QA: audio to audio",
    "tts": "Text to speech",
    "vision_mmbench": "MMBench-EN",
    "vision_pope": "POPE hallucination",
    "vision_textvqa": "TextVQA / OCR",
    "audio_vision": "Image + spoken question",
    "multi_image": "NLVR2 multi-image",
    "text_regression": "Text regression guard",
}


def fmt(value, scale=100.0):
    return "—" if value is None else f"{float(value) * scale:.2f}"


def browser_audio(path_value: str | None, run_dir: Path) -> Path | None:
    """Return browser-friendly audio, converting FLAC references to PCM WAV."""
    if not path_value:
        return None
    source = Path(path_value)
    if not source.is_file():
        return None
    if source.suffix.lower() == ".wav":
        return source

    safe_parent = re.sub(r"[^A-Za-z0-9_.-]+", "-", source.parent.name)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "-", source.stem)
    target = run_dir / "report_audio" / f"{safe_parent}-{safe_stem}.wav"
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file() or target.stat().st_mtime_ns < source.stat().st_mtime_ns:
        waveform, sample_rate = sf.read(source, dtype="float32", always_2d=False)
        sf.write(target, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return target


def main():
    parser = argparse.ArgumentParser(description="Build a side-by-side Markdown Omni benchmark report")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--samples-per-model", type=int, default=8)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    output = Path(args.output).resolve() if args.output else run_dir / "REPORT.md"
    models = []
    for directory in sorted(path for path in run_dir.iterdir() if path.is_dir()):
        metrics_path = directory / "metrics_scored.json"
        if not metrics_path.exists():
            metrics_path = directory / "metrics.json"
        rows_path = directory / "per_sample_scored.jsonl"
        if not rows_path.exists():
            rows_path = directory / "per_sample.jsonl"
        if metrics_path.exists() and rows_path.exists():
            models.append((directory.name, json.loads(metrics_path.read_text()), [json.loads(x) for x in rows_path.read_text().splitlines() if x]))
    if not models:
        raise SystemExit(f"No completed model results under {run_dir}")
    task_names = sorted({task for _, metrics, _ in models for task in metrics.get("task_metrics", {})})
    task_sets = [
        set(metrics.get("task_metrics", {})) - {"text_regression"}
        for _, metrics, _ in models
    ]
    common_tasks = sorted(set.intersection(*task_sets)) if task_sets else []
    lines = [
        f"# Omni benchmark: `{run_dir.name}`", "",
        f"> Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}. "
        f"Scorer `{models[0][1].get('scorer_version', SCORER_VERSION)}`; failures remain in the denominator.", "",
        "## Objective summary", "",
        "Fixed Omni uses the same required task set for every model and counts unsupported tasks as zero. Common Task uses only the exact tasks completed by every model in this report. Text regression is excluded from both.",
        f"Common tasks: {', '.join(LABELS.get(name, name) for name in common_tasks) or 'none'}.", "",
        "| Model | Role | Fixed Omni /100 | Common Task /100 | Lowest capability | Task coverage | Failure % | Amortized latency (s) | Throughput (sample/s) | Peak memory (GB) | Device |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    csv_rows = []
    for model, metrics, _ in models:
        environment = metrics.get("environment", {})
        task_metrics = metrics.get("task_metrics", {})
        common_score = 100.0 * statistics.fmean(
            float(task_metrics[name]["score"]) for name in common_tasks
        ) if common_tasks else 0.0
        fixed_capabilities = metrics.get("fixed_capability_metrics", {})
        lowest = min((float(value["score"]) for value in fixed_capabilities.values()), default=0.0)
        coverage = metrics.get("supported_task_count", 0) / max(metrics.get("required_task_count", 1), 1)
        fixed_score = float(metrics.get("fixed_omni_score", metrics.get("objective_score", 0.0)))
        lines.append(
            f"| {model} | {environment.get('role') or '—'} | {fixed_score:.2f} | {common_score:.2f} | "
            f"{lowest*100:.2f} | {coverage*100:.2f}% | {fmt(metrics.get('failure_rate'))} | "
            f"{fmt(metrics.get('mean_amortized_latency_seconds'), 1)} | "
            f"{fmt(metrics.get('throughput_samples_per_second'), 1)} | {fmt(metrics.get('peak_memory_gb'), 1)} | "
            f"{environment.get('accelerator') or '—'} |"
        )
        csv_rows.append({"model": model, "common_task_score": common_score, **metrics})
    capability_names = sorted({
        name for _, metrics, _ in models for name in metrics.get("fixed_capability_metrics", {})
    })
    lines += ["", "## Capability breakdown", "", "| Capability | " + " | ".join(m[0] for m in models) + " |", "|---|" + "---:|" * len(models)]
    for capability in capability_names:
        cells = []
        for _, metrics, _ in models:
            value = metrics.get("fixed_capability_metrics", {}).get(capability)
            cells.append("N/A" if not value else f"{value['score']*100:.2f}")
        lines.append(f"| {LABELS.get(capability, capability)} | " + " | ".join(cells) + " |")
    lines += ["", "## Task diagnostics", "", "| Task | " + " | ".join(m[0] for m in models) + " |", "|---|" + "---:|" * len(models)]

    def choice_diagnostics(value):
        detail = ""
        if value.get("answer_parse_rate") is not None:
            detail += f" / parsed {value['answer_parse_rate']*100:.1f}%"
        strict = value.get("strict_format_compliance", value.get("format_compliance"))
        if strict is not None:
            detail += f" / strict {strict*100:.1f}%"
        return detail

    for task in task_names:
        cells = []
        for _, metrics, _ in models:
            value = metrics.get("task_metrics", {}).get(task)
            if not value:
                cells.append("N/A")
            elif "wer" in value:
                detail = f"WER {value['wer']*100:.2f}%"
                if value.get("cer") is not None:
                    detail += f" / CER {value['cer']*100:.2f}%"
                if value.get("natural_label_wer") is not None:
                    detail += f" / natural WER {value['natural_label_wer']*100:.2f}%"
                if value.get("utmos") is not None:
                    detail += f" / UTMOS {value['utmos']:.2f}"
                elif value.get("rtf") is not None:
                    detail += f" / RTF {value['rtf']:.2f}"
                if value.get("duration_ratio") is not None:
                    detail += f" / duration× {value['duration_ratio']:.2f}"
                if value.get("natural_label_utmos") is not None:
                    detail += f" / natural UTMOS {value['natural_label_utmos']:.2f}"
                if value.get("audio_valid_rate") is not None:
                    detail += f" / valid {value['audio_valid_rate']*100:.1f}%"
                cells.append(detail)
            elif task == "vision_pope":
                cells.append(
                    f"F1 {value['f1']*100:.2f}% / Acc {value['accuracy']*100:.2f}%"
                    + choice_diagnostics(value)
                )
            elif task == "spoken_qa_a2a" and value.get("text_answer_score") is not None:
                cells.append(
                    f"Spoken {value['score']*100:.2f}% / text {value['text_answer_score']*100:.2f}%"
                    + choice_diagnostics(value)
                )
            elif "paired_degradation" in value:
                cells.append(
                    f"{value['score']*100:.2f}% / paired drop {value['paired_degradation']*100:+.2f}"
                    + choice_diagnostics(value)
                )
            else:
                detail = f"{value['score']*100:.2f} [{value['ci95'][0]*100:.2f}, {value['ci95'][1]*100:.2f}]"
                detail += choice_diagnostics(value)
                cells.append(detail)
        lines.append(f"| {LABELS.get(task, task)} | " + " | ".join(cells) + " |")
    by_model = {name: rows for name, _, rows in models}
    comparisons = [
        ("miniqwen-v0", "miniqwen-v0.1", "V0.1 minus V0"),
    ]
    comparison_lines = []
    for left_name, right_name, label in comparisons:
        if left_name not in by_model or right_name not in by_model:
            continue
        comparison = paired_fixed_omni_comparison(
            by_model[left_name], by_model[right_name],
        )
        comparison_lines.append(
            f"- {label}: **{comparison['delta']*100:+.2f} Fixed Omni points** "
            f"(task-stratified paired bootstrap 95% CI "
            f"[{comparison['ci95'][0]*100:+.2f}, {comparison['ci95'][1]*100:+.2f}], "
            f"n={comparison['samples']}; text regression excluded)."
        )
    if comparison_lines:
        lines += ["", "## Paired MiniQwen comparison", "", *comparison_lines]
    lines += ["", "## Fixed qualitative samples", "", "These examples are for inspection only and do not alter the objective score.", ""]
    for model, _, rows in models:
        lines += [f"### {model}", "", "| ID | Reference | Text output | Speech | Status |", "|---|---|---|---|---|"]
        selected, seen_tasks = [], set()
        for row in rows:
            if row.get("task") not in seen_tasks:
                selected.append(row)
                seen_tasks.add(row.get("task"))
            if len(selected) == args.samples_per_model:
                break
        for row in selected:
            audio = row.get("generated_audio")
            link = f"[play]({os.path.relpath(audio, output.parent)})" if audio and Path(audio).is_file() else "—"
            clean = lambda value: str(value or "").replace("|", "\\|").replace("\n", " ")[:240]
            lines.append(f"| {clean(row['sample_id'])} | {clean(row.get('reference'))} | {clean(row.get('generated_text'))} | {link} | {row.get('status')} |")
        lines.append("")

    lines += ["## TTS listening samples", ""]
    for model, _, rows in models:
        tts_rows = [row for row in rows if row.get("task") == "tts"][:3]
        if not tts_rows:
            continue
        lines += [f"### {model}", "", "| Target | Natural label | Generated | ASR transcript | UTMOS |", "|---|---|---|---|---:|"]
        for row in tts_rows:
            def audio_link(path):
                playable = browser_audio(path, run_dir)
                return f"[play]({os.path.relpath(playable, output.parent)})" if playable else "—"
            target = str(row.get("reference", "")).replace("|", "\\|")[:180]
            label = audio_link(row.get("metadata", {}).get("label_audio"))
            generated = audio_link(row.get("generated_audio"))
            transcript = str(row.get("generated_audio_text", "")).replace("|", "\\|")[:180]
            lines.append(f"| {target} | {label} | {generated} | {transcript or '—'} | {fmt(row.get('utmos'), 1)} |")
        lines.append("")

    clone_root = run_dir / "voice_clone"
    clone_summary_path = clone_root / "summary.json"
    if clone_summary_path.exists():
        clone_summary = json.loads(clone_summary_path.read_text(encoding="utf-8"))
        reference = clone_summary["reference"]
        lines += [
            "## Reference voice cloning", "",
            f"Fixed LibriSpeech test-clean subset: {clone_summary['speakers']} speakers, "
            f"{clone_summary['samples']} target utterances. ERes2Net compares generated speech with "
            "the conditioning reference and is independent of CAMPPlus; content is scored by SenseVoice 1-WER.", "",
            f"- Real reference↔label cosine: **{reference['real_similarity']['mean']:.4f}**.",
            f"- Wrong-speaker↔label cosine: **{reference['wrong_similarity']['mean']:.4f}**.",
            f"- Natural-label ASR WER: **{reference.get('natural_label_wer', {}).get('mean', 0)*100:.2f}%**.", "",
            f"- Natural-label UTMOS: **{fmt(reference.get('natural_label_utmos', {}).get('mean'), 1)} / 5**.", "",
            "| Model | Speaker cosine ↑ | 95% CI | Content 1-WER ↑ | UTMOS ↑ | Valid audio ↑ | Mean latency (s) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for model_id, values in clone_summary["models"].items():
            sim = values["speaker_similarity"]
            lines.append(
                f"| {model_id} | {sim['mean']:.4f} | [{sim['ci95'][0]:.4f}, {sim['ci95'][1]:.4f}] | "
                f"{values['content_score']['mean']*100:.2f} | {fmt(values.get('utmos', {}).get('mean'), 1)} | "
                f"{values['audio_valid_rate']*100:.2f} | "
                f"{values['mean_latency_seconds']:.2f} |"
            )
        comparison = clone_summary.get("v01_minus_v0")
        if comparison:
            sim_delta = comparison["speaker_similarity"]
            content_delta = comparison["content_score"]
            lines += [
                "",
                f"- V0.1−V0 paired speaker-cosine difference: **{sim_delta['mean']:+.4f}** "
                f"(95% CI [{sim_delta['ci95'][0]:+.4f}, {sim_delta['ci95'][1]:+.4f}]).",
                f"- V0.1−V0 paired content difference: **{content_delta['mean']*100:+.2f} points** "
                f"(95% CI [{content_delta['ci95'][0]*100:+.2f}, {content_delta['ci95'][1]*100:+.2f}]).",
            ]

        clone_rows = {}
        for model_id in clone_summary["models"]:
            scored_path = clone_root / model_id / "per_sample_scored.jsonl"
            if scored_path.exists():
                clone_rows[model_id] = [json.loads(line) for line in scored_path.read_text(encoding="utf-8").splitlines() if line]
        if clone_rows:
            first_model = next(iter(clone_rows))
            examples, seen_speakers = [], set()
            for row in clone_rows[first_model]:
                if row["speaker_id"] not in seen_speakers:
                    examples.append(row)
                    seen_speakers.add(row["speaker_id"])
            lines += ["", "### Voice-cloning listening samples", ""]
            headers = ["Speaker", "Reference", "Natural label"] + list(clone_rows)
            lines += ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
            indexed = {
                model_id: {row["sample_id"]: row for row in rows}
                for model_id, rows in clone_rows.items()
            }
            for row in examples:
                def audio_link(path):
                    playable = browser_audio(path, run_dir)
                    return f"[play]({os.path.relpath(playable, output.parent)})" if playable else "—"
                cells = [str(row["speaker_id"]), audio_link(row["ref_audio"]), audio_link(row["label_audio"])]
                for model_id in clone_rows:
                    candidate = indexed[model_id].get(row["sample_id"], {})
                    link = audio_link(candidate.get("generated_audio"))
                    cells.append(f"{link} (SIM {candidate.get('speaker_similarity', 0):.3f})")
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Interpretation rules", "",
        "- Qwen2.5-Omni-3B is a strong upper bound, not a parameter-matched baseline.",
        "- MCQ accuracy accepts an explicit letter, one uniquely stated option text, or one unambiguous standalone capital option letter; strict letter-format compliance is diagnostic only.",
        "- TextVQA keeps annotation-consensus scoring but accepts a normalized reference answer as a complete token span inside a brief explanation.",
        "- Text regression is a catastrophic-regression guard and is excluded from Omni scores.",
        "- Unsupported tasks count as zero in Fixed Omni and are N/A in task diagnostics.",
        "- Common Task compares the exact same supported tasks, not merely matching capability labels.",
        "- Efficiency is comparable only for runs made on the same device, precision, batch size and decoding settings.", "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "scorer_version", "common_task_score", "fixed_omni_score", "available_omni_score", "failure_rate", "unsupported_rate", "mean_amortized_latency_seconds", "throughput_samples_per_second", "peak_memory_gb", "samples"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(csv_rows)
    failures = [
        {"model": model, "sample_id": row.get("sample_id"), "status": row.get("status"), "error": row.get("error")}
        for model, _, rows in models for row in rows if row.get("status") not in {"ok", "unsupported"}
    ]
    with (run_dir / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "sample_id", "status", "error"])
        writer.writeheader(); writer.writerows(failures)
    print(output)


if __name__ == "__main__":
    main()
