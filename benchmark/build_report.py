#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from benchmark.metrics import bootstrap_ci


LABELS = {
    "text": "Text understanding",
    "audio": "Audio understanding / ASR",
    "vision": "Vision understanding",
    "audio_vision": "Image + spoken question",
    "speech_semantic": "Text–speech consistency",
    "speech_quality": "Speech quality / validity",
    "robustness": "Robustness",
}


def fmt(value, scale=100.0):
    return "—" if value is None else f"{float(value) * scale:.2f}"


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
    lines = [
        f"# Omni benchmark: `{run_dir.name}`", "",
        f"> Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}. All headline metrics are objective; failures remain in the denominator.", "",
        "## Objective summary", "",
        "| Model | Role | Score /100 | Failure % | Mean latency (s) | Peak memory (GB) | Device |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    csv_rows = []
    for model, metrics, _ in models:
        environment = metrics.get("environment", {})
        lines.append(f"| {model} | {environment.get('role') or '—'} | {metrics.get('objective_score', 0):.2f} | {fmt(metrics.get('failure_rate'))} | {fmt(metrics.get('mean_latency_seconds'), 1)} | {fmt(metrics.get('peak_memory_gb'), 1)} | {environment.get('accelerator') or '—'} |")
        csv_rows.append({"model": model, **metrics})
    lines += ["", "## Capability breakdown", "", "| Capability | " + " | ".join(m[0] for m in models) + " |", "|---|" + "---:|" * len(models)]
    for task in task_names:
        cells = []
        for _, metrics, _ in models:
            value = metrics.get("task_metrics", {}).get(task)
            cells.append("—" if not value else f"{value['score']*100:.2f} [{value['ci95'][0]*100:.2f}, {value['ci95'][1]*100:.2f}]")
        lines.append(f"| {LABELS.get(task, task)} | " + " | ".join(cells) + " |")
    by_model = {name: rows for name, _, rows in models}
    comparisons = [
        ("miniqwen-v0", "miniqwen-v0.1", "V0.1 minus V0"),
    ]
    comparison_lines = []
    for left_name, right_name, label in comparisons:
        if left_name not in by_model or right_name not in by_model:
            continue
        left = {row["sample_id"]: float(row.get("scores", {}).get("primary", 0)) for row in by_model[left_name]}
        right = {row["sample_id"]: float(row.get("scores", {}).get("primary", 0)) for row in by_model[right_name]}
        common = sorted(left.keys() & right.keys())
        deltas = [right[key] - left[key] for key in common]
        low, high = bootstrap_ci(deltas)
        mean_delta = sum(deltas) / max(len(deltas), 1)
        comparison_lines.append(
            f"- {label}: **{mean_delta*100:+.2f} points** "
            f"(paired bootstrap 95% CI [{low*100:+.2f}, {high*100:+.2f}], n={len(common)})."
        )
    if comparison_lines:
        lines += ["", "## Paired MiniQwen comparison", "", *comparison_lines]
    lines += ["", "## Fixed qualitative samples", "", "These examples are for inspection only and do not alter the objective score.", ""]
    for model, _, rows in models:
        lines += [f"### {model}", "", "| ID | Reference | Text output | Speech | Status |", "|---|---|---|---|---|"]
        for row in rows[:args.samples_per_model]:
            audio = row.get("generated_audio")
            link = f"[play]({os.path.relpath(audio, output.parent)})" if audio and Path(audio).is_file() else "—"
            clean = lambda value: str(value or "").replace("|", "\\|").replace("\n", " ")[:240]
            lines.append(f"| {clean(row['sample_id'])} | {clean(row.get('reference'))} | {clean(row.get('generated_text'))} | {link} | {row.get('status')} |")
        lines.append("")

    clone_root = run_dir / "voice_clone"
    clone_summary_path = clone_root / "summary.json"
    if clone_summary_path.exists():
        clone_summary = json.loads(clone_summary_path.read_text(encoding="utf-8"))
        reference = clone_summary["reference"]
        lines += [
            "## Voice cloning (MiniQwen only)", "",
            f"Fixed LibriSpeech test-clean subset: {clone_summary['speakers']} speakers, "
            f"{clone_summary['samples']} target utterances. ERes2Net speaker cosine is independent "
            "of the CAMPPlus encoder used for conditioning; content is scored by SenseVoice 1-WER.", "",
            f"- Real reference↔label cosine: **{reference['real_similarity']['mean']:.4f}**.",
            f"- Wrong-speaker↔label cosine: **{reference['wrong_similarity']['mean']:.4f}**.", "",
            "| Model | Speaker cosine ↑ | 95% CI | Content 1-WER ↑ | Valid audio ↑ | Mean latency (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for model_id, values in clone_summary["models"].items():
            sim = values["speaker_similarity"]
            lines.append(
                f"| {model_id} | {sim['mean']:.4f} | [{sim['ci95'][0]:.4f}, {sim['ci95'][1]:.4f}] | "
                f"{values['content_score']['mean']*100:.2f} | {values['audio_valid_rate']*100:.2f} | "
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
                    return f"[play]({os.path.relpath(path, output.parent)})" if path and Path(path).is_file() else "—"
                cells = [str(row["speaker_id"]), audio_link(row["ref_audio"]), audio_link(row["label_audio"])]
                for model_id in clone_rows:
                    candidate = indexed[model_id].get(row["sample_id"], {})
                    link = audio_link(candidate.get("generated_audio"))
                    cells.append(f"{link} (SIM {candidate.get('speaker_similarity', 0):.3f})")
                lines.append("| " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Interpretation rules", "", "- Qwen2.5-Omni-3B is a strong upper bound, not a parameter-matched baseline.", "- Efficiency is comparable only for runs made on the same device, precision and decoding settings.", "- Mini-Omni2 vision samples use the same spoken question input used by every model because its public vision API does not accept a text question.", ""]
    output.write_text("\n".join(lines), encoding="utf-8")
    with (run_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "objective_score", "failure_rate", "mean_latency_seconds", "peak_memory_gb", "samples"], extrasaction="ignore")
        writer.writeheader(); writer.writerows(csv_rows)
    failures = [
        {"model": model, "sample_id": row.get("sample_id"), "status": row.get("status"), "error": row.get("error")}
        for model, _, rows in models for row in rows if row.get("status") != "ok"
    ]
    with (run_dir / "failures.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "sample_id", "status", "error"])
        writer.writeheader(); writer.writerows(failures)
    print(output)


if __name__ == "__main__":
    main()
