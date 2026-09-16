#!/usr/bin/env python3
"""Build a self-contained Markdown index for one architecture experiment."""

import argparse
import datetime as dt
import html
import json
import os
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAGE_NUMBER = re.compile(r"stage-(\d+)-")

METRICS = [
    ("T2A 8-code NLL", ("t2a", "correct", "audio_nll_excluding_eos"), "lower"),
    ("T2A audio acc", ("t2a", "correct", "audio_accuracy"), "higher"),
    ("A2A text loss", ("a2a", "correct", "text_loss"), "lower"),
    ("A2A text acc", ("a2a", "correct", "text_accuracy"), "higher"),
    ("A2A 8-code NLL", ("a2a", "correct", "audio_nll_excluding_eos"), "lower"),
    ("A2A audio acc", ("a2a", "correct", "audio_accuracy"), "higher"),
    ("Audio input gain", ("a2a", "audio_input_gain", "text_loss"), "higher"),
    ("Speaker gain", ("a2a", "speaker_only_gain", "audio_nll_excluding_eos"), "higher"),
    ("I2T text loss", ("i2t", "correct", "text_loss"), "lower"),
    ("I2T text acc", ("i2t", "correct", "text_accuracy"), "higher"),
    ("Vision gain", ("i2t", "vision_gain", "text_loss"), "higher"),
]


def nested(report, path):
    value = report.get("datasets", {})
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def comparable_metric(report, path):
    value = nested(report, path)
    if value is None and path[-1] == "audio_nll_excluding_eos":
        # Preserved pre-v5 reports expose the equivalent unweighted metric as
        # audio_loss. EOS is one token, so this is the closest read-only
        # comparison without rerunning an old checkpoint.
        value = nested(report, (*path[:-1], "audio_loss"))
        if value is None:
            value = nested(report, (*path[:-1], "audio_weighted_loss"))
    return value


def fmt(value, percent=False):
    if value is None:
        return "—"
    return f"{value * 100:.2f}%" if percent else f"{value:.4f}"


def table_text(value):
    return str(value or "—").replace("|", "\\|").replace("\n", " ").strip()


def append_comparison(lines, title, baseline, candidate):
    lines += [
        "",
        title,
        "",
        "| Metric | Baseline | Candidate | Delta | Direction |",
        "|---|---:|---:|---:|---|",
    ]
    for label, path, direction in METRICS:
        before, after = comparable_metric(baseline, path), comparable_metric(candidate, path)
        if before is None or after is None:
            continue
        delta = after - before
        improved = delta < 0 if direction == "lower" else delta > 0
        verdict = "better" if improved else ("same" if abs(delta) < 1e-12 else "worse")
        percent = "accuracy" in path[-1]
        lines.append(
            f"| {label} | {fmt(before, percent)} | {fmt(after, percent)} | "
            f"{fmt(delta, percent) if delta < 0 else '+' + fmt(delta, percent)} | {direction}; **{verdict}** |"
        )


def stage_key(path):
    match = STAGE_NUMBER.search(path.name)
    return int(match.group(1)) if match else 999


def clean_log_line(line):
    line = ANSI.sub("", line)
    if "ALINPU INFO" in line:
        marker = re.search(r"\[\d{4}-\d{2}-\d{2}[^\]]*\]\[ALINPU INFO\]", line)
        return line[:marker.start()].rstrip() if marker else ""
    return line.rstrip()


def parse_generation(log_path):
    if not log_path.exists():
        return []
    samples, current, response_lines = [], None, []
    section = None

    def finish_response():
        nonlocal response_lines
        if current is not None and response_lines:
            current["response"] = "\n".join(response_lines).strip()
        response_lines = []

    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = clean_log_line(raw)
        if "text ->" in line:
            section = "text"
        elif "audio ->" in line:
            section = "audio"
        elif "clone voice" in line:
            section = "clone"
        elif "image ->" in line:
            section = "image"
        elif "text+audio+image" in line:
            section = "mixed"

        text_match = re.match(r"\s*📝 \[text-\d+\]:\s*(.*)", line)
        audio_match = re.match(r"🎤 \[audio-\d+\]:\s*(.*)", line)
        clone_match = re.match(r"🎵 \[clone:\s*([^\]]+)\]\s*(.*)", line)
        image_match = re.match(r"🖼️ \[image-\d+\]:\s*(.*)", line)
        thinker_match = re.match(r"📒 \[Thinker\]:\s*(.*)", line)
        talker_match = re.match(r"🎹 \[Talker\]:\s*(\d+) frames.*?Audio decoded to:\s*(.*)", line)

        if audio_match:
            finish_response()
            current = {"kind": "audio", "input": audio_match.group(1)}
        elif clone_match:
            finish_response()
            current = {"kind": "clone", "input": clone_match.group(1), "condition": clone_match.group(2)}
        elif image_match:
            finish_response()
            current = {"kind": "image", "input": image_match.group(1)}
        elif text_match:
            if section == "clone" and current is not None:
                current["prompt"] = text_match.group(1)
            else:
                finish_response()
                current = {"kind": section or "text", "input": text_match.group(1)}
        elif thinker_match:
            response_lines = [thinker_match.group(1)] if thinker_match.group(1) else []
        elif talker_match and current is not None:
            finish_response()
            current["frames"] = int(talker_match.group(1))
            current["output"] = talker_match.group(2).strip()
            timing = re.search(r"generation=([0-9.]+)s\s*\|\s*RTF=([0-9.]+)", line)
            if timing:
                current["generation_seconds"] = float(timing.group(1))
                current["rtf"] = float(timing.group(2))
            samples.append(current)
            current = None
        elif current is not None and response_lines and line.strip():
            response_lines.append(line.strip())
        elif current is not None and not response_lines and line.strip() and "Thinker" not in line:
            # The first response line can follow a device-info line that was removed.
            if not line.startswith(("=", "Model Params")):
                response_lines.append(line.strip())
    return samples


def relative_link(target, output):
    path = Path(target)
    if not path.is_absolute():
        path = (output.parent / path).resolve()
    return Path(os.path.relpath(path, output.parent)).as_posix()


def find_input(project_root, kind, name):
    if kind not in {"audio", "image", "mixed"}:
        return None
    candidate = project_root / "dataset" / "eval_omni" / name
    return candidate if candidate.exists() else None


def load_reports(result_dir):
    reports = []
    for path in sorted(result_dir.glob("stage-*.json"), key=stage_key):
        reports.append((path, json.loads(path.read_text(encoding="utf-8"))))
    return reports


def training_peaks(train_log_dir):
    peaks = {}
    if not train_log_dir.exists():
        return peaks
    for directory in sorted(train_log_dir.glob("stage-*")):
        match = re.match(r"stage-(\d+)-", directory.name)
        if not match:
            continue
        values = []
        for log in directory.glob("ranks/*/attempt_0/0/stdout.log"):
            values += [float(value) for value in re.findall(r"peak:([0-9.]+)GB", log.read_text(errors="replace"))]
        if not values and (directory / "console.log").exists():
            values = [float(value) for value in re.findall(r"peak:([0-9.]+)GB", (directory / "console.log").read_text(errors="replace"))]
        if values:
            peaks[int(match.group(1))] = max(values)
    return peaks


def build(args):
    project_root = Path(__file__).resolve().parents[1]
    result_dir = Path(args.result_dir).resolve()
    output = Path(args.output).resolve() if args.output else result_dir / "REPORT.md"
    reports = load_reports(result_dir)
    if not reports:
        raise SystemExit(f"No stage reports found in {result_dir}")
    manifest_path = result_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    peaks = training_peaks(Path(args.train_log_dir).resolve())
    experiment = result_dir.name
    final_path, final_report = reports[-1]
    final_stage_id = stage_key(final_path)
    lines = [
        f"# MiniQwen-Omni architecture report: `{experiment}`",
        "",
        f"> Generated automatically at {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        "## Experiment",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Completed stage | {final_report.get('stage', '—')} |",
        f"| Talker | {final_report.get('num_talker_hidden_layers', '—')} layers × {final_report.get('talker_hidden_size', '—')} hidden |",
        f"| Thinker bridge layer | {final_report.get('accept_hidden_layer', '—')} |",
        f"| Checkpoint | `{args.checkpoint}` |",
        "",
        "## Training-domain dataset (legacy stage diagnostics)",
        "",
        "| Task | Train | Dev | Train language | Dev language | Group overlap |",
        "|---|---:|---:|---|---|---:|",
    ]
    for task, stats in manifest.get("stats", {}).items():
        train_language = stats.get("train_language") or f"en={stats.get('train_en', 0)}, zh={stats.get('train_zh', 0)}"
        dev_language = f"en={stats.get('dev_en', 0)}, zh={stats.get('dev_zh', 0)}"
        lines.append(
            f"| {task.upper()} | {stats.get('train', '—')} | {stats.get('dev', '—')} | "
            f"{train_language} | {dev_language} | {stats.get('group_overlap', '—')} |"
        )

    holdout_path = result_dir / "holdout-final.json"
    holdout_manifest_path = result_dir / "holdout_manifest.json"
    holdout_report = json.loads(holdout_path.read_text(encoding="utf-8")) if holdout_path.exists() else None
    holdout_manifest = json.loads(holdout_manifest_path.read_text(encoding="utf-8")) if holdout_manifest_path.exists() else {}
    if holdout_report is not None:
        lines += [
            "",
            "## Full-corpus holdout evaluation (primary)",
            "",
            "These fixed English examples come from the full datasets and are excluded from mini training by auditable group keys. Use this section for architecture selection; the stage table below remains a training-domain diagnostic.",
            "",
            "| Task | Full source rows | Mini train rows | Holdout pool | Evaluated | Primary exclusion | Primary overlap | Content overlap |",
            "|---|---:|---:|---:|---:|---|---:|---|",
        ]
        for task, stats in holdout_manifest.get("stats", {}).items():
            evaluated = nested(holdout_report, (task, "samples"))
            if "conversation_overlap" in stats:
                content_overlap = (
                    f"{stats['conversation_overlap']}/{stats.get('dev_conversation_groups', '—')} conversations"
                )
            elif task == "t2a":
                content_overlap = "0 (primary key)"
            else:
                content_overlap = "—"
            lines.append(
                f"| {task.upper()} | {stats.get('source_rows', '—')} | {stats.get('train', '—')} | "
                f"{stats.get('dev', '—')} | {evaluated if evaluated is not None else '—'} | "
                f"{stats.get('group_key', '—')} | {stats.get('group_overlap', '—')} | {content_overlap} |"
            )
        a2a_stats = holdout_manifest.get("stats", {}).get("a2a", {})
        if a2a_stats.get("conversation_overlap", 0):
            lines += [
                "",
                "> **A2A scope:** this full corpus contains only a few English conversations outside mini training. "
                "The A2A holdout therefore measures unseen-speaker/acoustic generalisation; "
                f"{a2a_stats['conversation_overlap']}/{a2a_stats.get('dev_conversation_groups', '—')} unique conversations "
                "reuse text spoken by other training speakers. Do not interpret it as an unseen-question semantic benchmark.",
            ]
        lines += [
            "",
            "| T2A 8-code NLL | T2A acc | A2A text loss | A2A text acc | A2A 8-code NLL | A2A audio acc | Audio input gain | Speaker gain | I2T text loss | I2T text acc | Vision gain |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            "| " + " | ".join([
                fmt(comparable_metric(holdout_report, ("t2a", "correct", "audio_nll_excluding_eos"))),
                fmt(comparable_metric(holdout_report, ("t2a", "correct", "audio_accuracy")), True),
                fmt(comparable_metric(holdout_report, ("a2a", "correct", "text_loss"))),
                fmt(comparable_metric(holdout_report, ("a2a", "correct", "text_accuracy")), True),
                fmt(comparable_metric(holdout_report, ("a2a", "correct", "audio_nll_excluding_eos"))),
                fmt(comparable_metric(holdout_report, ("a2a", "correct", "audio_accuracy")), True),
                fmt(comparable_metric(holdout_report, ("a2a", "audio_input_gain", "text_loss"))),
                fmt(comparable_metric(holdout_report, ("a2a", "speaker_only_gain", "audio_nll_excluding_eos"))),
                fmt(comparable_metric(holdout_report, ("i2t", "correct", "text_loss"))),
                fmt(comparable_metric(holdout_report, ("i2t", "correct", "text_accuracy")), True),
                fmt(comparable_metric(holdout_report, ("i2t", "vision_gain", "text_loss"))),
            ]) + " |",
        ]
        examples = []
        for task, stats in holdout_manifest.get("stats", {}).items():
            for example in stats.get("examples", []):
                examples.append((task.upper(), example))
        if examples:
            lines += [
                "",
                "### Holdout examples",
                "",
                "| Task | Group | User input | Reference answer |",
                "|---|---|---|---|",
            ]
            for task, example in examples:
                lines.append(
                    f"| {task} | `{example.get('group', '—')}` | {table_text(example.get('user'))} | "
                    f"{table_text(example.get('assistant'))} |"
                )

        baseline_holdout_path = Path(args.baseline_result_dir).resolve() / "holdout-final.json"
        baseline_holdout_manifest = Path(args.baseline_result_dir).resolve() / "holdout_manifest.json"
        same_holdout = (
            baseline_holdout_path.exists() and baseline_holdout_manifest.exists() and
            holdout_manifest_path.read_bytes() == baseline_holdout_manifest.read_bytes()
        )
        if same_holdout:
            baseline_holdout = json.loads(baseline_holdout_path.read_text(encoding="utf-8"))
            append_comparison(
                lines,
                f"### Holdout comparison with `{Path(args.baseline_result_dir).resolve().name}`",
                baseline_holdout,
                holdout_report,
            )

    lines += [
        "",
        "## Metrics by stage (training-domain diagnostic)",
        "",
        "Dev metrics are teacher-forced. Lower loss and higher accuracy/gain are better.",
        "",
        "| Stage | T2A audio loss | T2A acc | A2A text loss | A2A audio loss | A2A acc | Audio input gain | Speaker gain | I2T text loss | I2T acc | Vision gain | Train peak/PPU |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, report in reports:
        stage = report.get("stage", "—")
        stage_id = int(str(stage).split("-", 1)[0]) if str(stage).split("-", 1)[0].isdigit() else -1
        values = {
            "t2a_loss": nested(report, ("t2a", "correct", "audio_weighted_loss")),
            "t2a_acc": nested(report, ("t2a", "correct", "audio_accuracy")),
            "a2a_text": nested(report, ("a2a", "correct", "text_loss")),
            "a2a_loss": nested(report, ("a2a", "correct", "audio_weighted_loss")),
            "a2a_acc": nested(report, ("a2a", "correct", "audio_accuracy")),
            "audio_gain": nested(report, ("a2a", "audio_input_gain", "text_loss")),
            "speaker_gain": nested(report, ("a2a", "speaker_only_gain", "audio_weighted_loss")),
            "i2t_loss": nested(report, ("i2t", "correct", "text_loss")),
            "i2t_acc": nested(report, ("i2t", "correct", "text_accuracy")),
            "vision_gain": nested(report, ("i2t", "vision_gain", "text_loss")),
        }
        lines.append(
            f"| {stage} | {fmt(values['t2a_loss'])} | {fmt(values['t2a_acc'], True)} | "
            f"{fmt(values['a2a_text'])} | {fmt(values['a2a_loss'])} | {fmt(values['a2a_acc'], True)} | "
            f"{fmt(values['audio_gain'])} | {fmt(values['speaker_gain'])} | {fmt(values['i2t_loss'])} | "
            f"{fmt(values['i2t_acc'], True)} | {fmt(values['vision_gain'])} | {fmt(peaks.get(stage_id))} GB |"
        )

    baseline_dir = Path(args.baseline_result_dir).resolve()
    if baseline_dir != result_dir and baseline_dir.exists():
        baseline_reports = load_reports(baseline_dir)
        baseline_manifest = baseline_dir / "data_manifest.json"
        same_data = manifest_path.exists() and baseline_manifest.exists() and manifest_path.read_bytes() == baseline_manifest.read_bytes()
        matching_baseline = next((report for path, report in baseline_reports if stage_key(path) == final_stage_id), None)
        if matching_baseline is not None and same_data:
            append_comparison(
                lines,
                f"## Stage {final_stage_id} comparison with `{baseline_dir.name}`",
                matching_baseline,
                final_report,
            )

    samples = parse_generation(result_dir / "generation.log")
    lines += ["", "## Qualitative generation", ""]
    if not samples:
        lines += [
            "Qualitative generation has not run yet or was explicitly disabled.",
            "Set `ARCH_EVAL_RUN_GENERATION=1` and rerun the same experiment command to append samples without retraining.",
        ]
    else:
        speech_metrics_path = result_dir / "speech_metrics.json"
        speech_metrics = json.loads(speech_metrics_path.read_text(encoding="utf-8")) if speech_metrics_path.exists() else {}
        speech_by_audio = {
            str(Path(item.get("audio", "")).resolve()): item
            for item in speech_metrics.get("utterances", [])
        }
        non_english = sum(bool(re.search(r"[\u4e00-\u9fff]", sample.get("response", ""))) for sample in samples)
        limit_hits = sum(sample.get("frames", 0) >= 248 for sample in samples)
        lines += [
            f"Generated samples: **{len(samples)}**. Use the text and media side by side for inspection.",
            f"Automatic flags: **{non_english}** non-English response(s), **{limit_hits}** generation-limit hit(s).",
            f"Generated-speech micro WER: **{fmt(speech_metrics.get('micro_wer'))}**.",
        ]
        labels = {"text": "Text → text/audio", "audio": "Audio → text/audio", "clone": "Voice clone", "image": "Image → text/audio", "mixed": "Mixed input"}
        for index, sample in enumerate(samples, 1):
            kind = sample.get("kind", "sample")
            lines += ["", f"### {index}. {labels.get(kind, kind)}", ""]
            if kind == "clone":
                lines.append(f"- Voice: `{sample.get('input', '—')}` ({sample.get('condition', '')})")
                lines.append(f"- Prompt: {sample.get('prompt', '—')}")
            else:
                lines.append(f"- Input: {sample.get('input', '—')}")
            source = find_input(project_root, kind, sample.get("input", ""))
            if source is not None and kind == "image":
                source_link = relative_link(source, output)
                lines += ["", f"![Input image]({source_link})"]
            elif source is not None and kind in {"audio", "mixed"}:
                source_link = relative_link(source, output)
                lines += ["", f"<audio controls preload=\"none\" src=\"{html.escape(source_link)}\"></audio>", "", f"[Open input audio]({source_link})"]
            lines += ["", "Thinker response:", "", sample.get("response", "_No text captured._"), ""]
            output_path = sample.get("output")
            if output_path:
                speech = speech_by_audio.get(str(Path(output_path).resolve()), {})
                audio_link = relative_link(output_path, output)
                frames = sample.get("frames", 0)
                duration = f", approximately {frames * 0.08:.1f}s" if frames else ""
                response = sample.get("response", "")
                language_warning = " — ⚠ non-English response" if re.search(r"[\u4e00-\u9fff]", response) else ""
                lines += [
                    f"Response language: **{'Chinese' if language_warning else 'English'}**{language_warning}",
                    "",
                    f"Talker output: **{frames or '—'} frames{duration}**" + (" — ⚠ reached generation limit" if frames >= 248 else ""),
                    f"Generation: **{sample.get('generation_seconds', '—')}s**, RTF: **{sample.get('rtf', '—')}**",
                    f"Generated-audio ASR: {speech.get('asr', '—')}",
                    f"Utterance WER: **{fmt(speech.get('wer'))}**",
                    "",
                    f"<audio controls preload=\"none\" src=\"{html.escape(audio_link)}\"></audio>",
                    "",
                    f"[Open output audio]({audio_link})",
                ]

    lines += [
        "",
        "## Decision guide",
        "",
        "- Compare candidates only when the data manifest and stage budget match the baseline.",
        "- Target loss should decrease and target accuracy/gain should increase.",
        "- Check Stage 4 for regressions in modalities unrelated to the architecture change.",
        "- Treat changes below roughly 1% as inconclusive unless repeated with another seed.",
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    temp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(temp, output)
    print(f"Architecture report: {output}")


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--train-log-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--baseline-result-dir", default=str(root / ".runtime/arch_eval/main_codec_cp_2l_v1"))
    parser.add_argument("--output", default="")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
