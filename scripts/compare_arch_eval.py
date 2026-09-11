#!/usr/bin/env python3
"""Collect comparable MiniQwen-Omni architecture metrics into one CSV."""

import argparse
import csv
import json
from pathlib import Path


def nested(data, *keys, default=0.0):
    for key in keys:
        if not isinstance(data, dict) or key not in data:
            return default
        data = data[key]
    return data


def audio_nll(data, *keys):
    parent = nested(data, *keys, default={})
    if not isinstance(parent, dict):
        return 0.0
    return parent.get(
        "audio_nll_excluding_eos",
        parent.get("audio_loss", parent.get("audio_weighted_loss", 0.0)),
    )


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(project_root / ".runtime/arch_eval"))
    parser.add_argument("--stage", default="stage-4-i2t-joint.json")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.root)
    records = []
    for path in sorted(root.glob(f"*/{args.stage}")):
        report = json.loads(path.read_text(encoding="utf-8"))
        datasets = report["datasets"]
        records.append({
            "experiment": path.parent.name,
            "talker_layers": report["num_talker_hidden_layers"],
            "talker_hidden": report["talker_hidden_size"],
            "bridge_layer": report["accept_hidden_layer"],
            "t2a_audio_loss": audio_nll(datasets, "t2a", "correct"),
            "t2a_audio_acc": nested(datasets, "t2a", "correct", "audio_accuracy"),
            "a2a_text_loss": nested(datasets, "a2a", "correct", "text_loss"),
            "a2a_audio_loss": audio_nll(datasets, "a2a", "correct"),
            "audio_input_gain": audio_nll(datasets, "a2a", "audio_input_gain"),
            "speaker_gain": audio_nll(datasets, "a2a", "speaker_only_gain"),
            "i2t_text_loss": nested(datasets, "i2t", "correct", "text_loss"),
            "vision_gain": nested(datasets, "i2t", "vision_gain", "text_loss"),
            "peak_memory_gb": report.get("peak_memory_gb", 0.0),
        })
    if not records:
        raise SystemExit(f"No {args.stage} reports found under {root}")

    output = Path(args.output) if args.output else root / f"comparison-{Path(args.stage).stem}.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    columns = ["experiment", "t2a_audio_loss", "a2a_audio_loss", "i2t_text_loss", "audio_input_gain", "speaker_gain", "vision_gain"]
    widths = {column: max(len(column), *(len(f"{row[column]:.4f}") if isinstance(row[column], float) else len(str(row[column])) for row in records)) for column in columns}
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in records:
        values = []
        for column in columns:
            value = row[column]
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            values.append(text.ljust(widths[column]))
        print("  ".join(values))
    print(f"Comparison saved: {output}")


if __name__ == "__main__":
    main()
