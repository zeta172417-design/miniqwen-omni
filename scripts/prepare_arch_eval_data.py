#!/usr/bin/env python3
"""Build fixed train/dev splits for MiniQwen-Omni architecture tests.

Training continues to reuse the complete mini audio datasets. Evaluation rows
come from the corresponding full corpora and are group-disjoint from mini
training: T2A by conversation content, A2A by speaker embedding, and I2T by
image bytes. This makes the report a generalisation check rather than a metric
on examples already seen during mini training.
"""

import argparse
import heapq
import hashlib
import io
import json
import math
import os
import random
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoTokenizer


CJK_RE = re.compile(r"[\u3400-\u9fff]")
def stable_digest(*parts):
    h = hashlib.sha256()
    for part in parts:
        if not isinstance(part, bytes):
            part = str(part).encode("utf-8")
        h.update(part)
        h.update(b"\0")
    return h.hexdigest()


def source_identity(path):
    stat = path.stat()
    parquet = pq.ParquetFile(path)
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "rows": parquet.metadata.num_rows,
    }


def read_sampled_rows(path, count, seed, batch_size=2048):
    """Read a deterministic sample without materialising a full parquet.

    Multi-row-group files are sampled across row groups. Audio mini files have
    one very large row group, so they are streamed from the beginning; scanning
    to random global indices would decompress the entire 1--5 GiB file merely
    to create a small benchmark.
    """
    parquet = pq.ParquetFile(path)
    total = parquet.metadata.num_rows
    count = min(count, total)
    rng = random.Random(seed)
    if parquet.metadata.num_row_groups > 1:
        average = max(1, total // parquet.metadata.num_row_groups)
        group_count = min(parquet.metadata.num_row_groups, max(1, math.ceil(count / average) + 2))
        groups = sorted(rng.sample(range(parquet.metadata.num_row_groups), group_count))
        rows = []
        for group in groups:
            rows.extend(parquet.read_row_group(group).to_pylist())
        rng.shuffle(rows)
        return rows[:count]

    rows = []
    for batch in parquet.iter_batches(batch_size=batch_size, use_threads=False):
        rows.extend(pa.Table.from_batches([batch]).to_pylist())
        if len(rows) >= count:
            break
    rows = rows[:count]
    rng.shuffle(rows)
    return rows


def parse_conversations(row):
    try:
        conversations = json.loads(row.get("conversations") or "[]")
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(conversations, list):
        return None
    users = [turn for turn in conversations if turn.get("role") == "user" and str(turn.get("content", "")).strip()]
    assistants = [turn for turn in conversations if turn.get("role") == "assistant" and str(turn.get("content", "")).strip()]
    return conversations if users and assistants else None


def row_language(conversations):
    user_text = " ".join(str(turn.get("content", "")) for turn in conversations if turn.get("role") == "user")
    return "zh" if CJK_RE.search(user_text) else "en"


def rendered_length(tokenizer, conversations, image_token_len=64):
    copied = []
    for turn in conversations:
        content = str(turn.get("content", ""))
        image_block = "<|vision_start|>" + "<|image_pad|>" * image_token_len + "<|vision_end|>"
        content = content.replace("<image>", image_block)
        copied.append({"role": turn.get("role"), "content": content})
    rendered = tokenizer.apply_chat_template(copied, tokenize=False, add_generation_prompt=False)
    return len(tokenizer(rendered, add_special_tokens=False).input_ids)


def validate_image(raw):
    if not raw or len(raw) < 1024:
        return False
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            width, height = image.size
    except Exception:
        return False
    short, long = sorted((width, height))
    return short >= 64 and long >= 128 and long / max(short, 1) <= 8


def row_group(kind, row):
    """Return the leakage-control group used by both dev and train writers."""
    if kind == "i2t":
        return stable_digest(row.get("image_bytes") or b"")
    if kind == "a2a":
        embedding = row.get("spk_emb") or []
        speaker_bytes = pa.array(embedding, type=pa.float32()).buffers()[1]
        return stable_digest(speaker_bytes.to_pybytes() if speaker_bytes else b"")
    # Exclude every full-corpus T2A row whose prompt/answer text was seen in
    # mini training, even if it was rendered with another voice.
    return stable_digest(row.get("conversations") or "")


def collect_training_groups(kind, path, batch_size=8192):
    """Collect compact leakage keys without decoding large codec columns."""
    columns = ["spk_emb"] if kind == "a2a" else ["conversations"]
    groups = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=columns, use_threads=False,
    ):
        values = batch.column(0).to_pylist()
        if kind == "a2a":
            groups.update(row_group(kind, {"spk_emb": value}) for value in values)
        else:
            groups.update(stable_digest(value or "") for value in values)
    return groups


def collect_conversation_groups(path, batch_size=8192):
    groups = set()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["conversations"], use_threads=False,
    ):
        groups.update(stable_digest(value or "") for value in batch.column(0).to_pylist())
    return groups


def read_holdout_pool(kind, path, count, seed, language, forbidden_groups,
                      batch_size=1024):
    """Stream a full parquet and retain a deterministic bounded candidate pool.

    A2A is stored as one multi-gigabyte row group, so random row-group reads
    would repeatedly sample its training prefix. A hash-ranked streaming pool
    gives deterministic corpus-wide sampling while keeping only ``count`` rows
    in memory.
    """
    heap = []
    serial = 0
    eligible = 0
    parquet = pq.ParquetFile(path)
    row_groups = None
    if parquet.metadata.num_row_groups > 1:
        average_rows = max(1, parquet.metadata.num_rows // parquet.metadata.num_row_groups)
        group_count = min(
            parquet.metadata.num_row_groups,
            max(32, math.ceil(count / average_rows) * 8),
        )
        row_groups = sorted(
            range(parquet.metadata.num_row_groups),
            key=lambda index: stable_digest(seed, "row-group", index),
        )[:group_count]
        row_groups.sort()
    scanned_rows = (
        sum(parquet.metadata.row_group(index).num_rows for index in row_groups)
        if row_groups is not None else parquet.metadata.num_rows
    )
    for batch_index, batch in enumerate(parquet.iter_batches(
        batch_size=batch_size, row_groups=row_groups, use_threads=False,
    ), 1):
        for row in pa.Table.from_batches([batch]).to_pylist():
            conversations = parse_conversations(row)
            if conversations is None or row_language(conversations) != language:
                continue
            group = row_group(kind, row)
            if group in forbidden_groups:
                continue
            if kind == "a2a" and not (
                row.get("answer_audios") and row.get("question_audios") and
                row.get("ref_audios") and row.get("spk_emb")
            ):
                continue
            if kind == "t2a" and not row.get("answer_audios"):
                continue
            eligible += 1
            rank = int(stable_digest(seed, group, row.get("conversations") or ""), 16)
            item = (-rank, serial, row)
            serial += 1
            if len(heap) < count:
                heapq.heappush(heap, item)
            elif rank < -heap[0][0]:
                heapq.heapreplace(heap, item)
        if batch_index % 100 == 0:
            print(
                f"  scanned {min(batch_index * batch_size, scanned_rows)}"
                f"/{scanned_rows} selected {kind} rows",
                flush=True,
            )
    rows = [item[2] for item in sorted(heap, key=lambda item: -item[0])]
    return rows, eligible, scanned_rows


def prepare_candidates(kind, rows, tokenizer, max_seq_len, language=None):
    prepared = []
    for row in rows:
        conversations = parse_conversations(row)
        if conversations is None:
            continue
        detected_language = row_language(conversations)
        if language is not None and detected_language != language:
            continue
        try:
            token_length = rendered_length(tokenizer, conversations)
        except Exception:
            continue
        if token_length > max_seq_len:
            continue

        if kind == "i2t":
            marker_count = sum(str(turn.get("content", "")).count("<image>") for turn in conversations)
            image_bytes = row.get("image_bytes")
            if marker_count != 1 or not validate_image(image_bytes):
                continue
            group = row_group(kind, row)
        else:
            answers = row.get("answer_audios") or []
            if not answers or not any(answer for answer in answers):
                continue
            if kind == "a2a":
                if not (row.get("question_audios") and row.get("ref_audios") and row.get("spk_emb")):
                    continue
            group = row_group(kind, row)

        signature = stable_digest(row.get("conversations"), group)
        prepared.append({
            "row": row,
            "language": detected_language,
            "group": group,
            "signature": signature,
        })
    return prepared


def select_language(candidates, count, seed, language, forbidden_groups=None):
    """Select a deterministic single-language split without duplicating rows."""
    forbidden_groups = set(forbidden_groups or ())
    selected, selected_signatures, selected_groups = [], set(), set()
    ordered = sorted(candidates, key=lambda item: stable_digest(seed, item["signature"]))
    for item in ordered:
        if item["language"] != language or item["group"] in forbidden_groups:
            continue
        if item["signature"] in selected_signatures or item["group"] in selected_groups:
            continue
        selected.append(item)
        selected_signatures.add(item["signature"])
        selected_groups.add(item["group"])
        if len(selected) == count:
            return selected
    raise RuntimeError(
        f"not enough {language} candidates: selected={len(selected)}, "
        f"requested={count}, available={len(candidates)}"
    )


def write_rows(path, items, schema):
    rows = [item["row"] for item in items]
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, tmp, compression="zstd", row_group_size=2048)
    os.replace(tmp, path)


def manifest_examples(items, limit=3):
    examples = []
    for item in items[:limit]:
        conversations = parse_conversations(item["row"]) or []
        user = next((str(turn.get("content", "")) for turn in conversations if turn.get("role") == "user"), "")
        assistant = next((str(turn.get("content", "")) for turn in conversations if turn.get("role") == "assistant"), "")
        examples.append({
            "group": item["group"][:16],
            "user": user[:240],
            "assistant": assistant[:240],
        })
    return examples


def upgrade_manifest_audits(output_dir, manifest, sources):
    """Add secondary content-overlap disclosure to a v3 holdout in place."""
    a2a_stats = manifest.get("stats", {}).get("a2a", {})
    if manifest.get("format_version", 0) >= 4 and "conversation_overlap" in a2a_stats:
        return False
    train_conversations = collect_conversation_groups(sources["a2a_train"])
    dev_conversations = collect_conversation_groups(output_dir / "a2a_dev.parquet")
    a2a_stats.update({
        "dev_conversation_groups": len(dev_conversations),
        "conversation_overlap": len(train_conversations & dev_conversations),
        "content_scope": "unseen speakers; conversation content may repeat",
    })
    manifest["format_version"] = 4
    tmp_manifest = (output_dir / "manifest.json").with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, output_dir / "manifest.json")
    return True


def build(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    sources = {
        "t2a_full": Path(args.t2a_source),
        "t2a_train": Path(args.t2a_train_source),
        "a2a_full": Path(args.a2a_source),
        "a2a_train": Path(args.a2a_train_source),
        "i2t": Path(args.i2t_source),
    }
    requested = {
        "data_policy": (
            "train_on_complete_audio_mini; evaluate_on_full_corpus_english_holdout; "
            "t2a_conversation_disjoint; a2a_speaker_disjoint; i2t_image_disjoint"
        ),
        "seed": args.seed,
        "eval_language": args.eval_language,
        "counts": {
            "t2a_train": "source_direct", "t2a_dev": args.t2a_dev,
            "a2a_train": "source_direct", "a2a_dev": args.a2a_dev,
            "i2t_train": args.i2t_train, "i2t_dev": args.i2t_dev,
        },
        "sources": {name: source_identity(path) for name, path in sources.items()},
    }
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("request") == requested:
            if upgrade_manifest_audits(output_dir, existing, sources):
                print(f"Added A2A content-overlap audit: {output_dir}")
                return
            print(f"Mini benchmark already prepared: {output_dir}")
            return
        if existing.get("format_version", 0) >= 4:
            raise RuntimeError(f"{manifest_path} was built with different settings; pass --force to replace it")
        print(f"Migrating legacy in-domain benchmark to full-corpus holdout: {output_dir}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, fix_mistral_regex=True)
    stats = {}
    outputs = {}
    audio_specs = {
        "t2a": (args.t2a_dev, 8, 512),
        "a2a": (args.a2a_dev, 48, 768),
    }
    for offset, (kind, (dev_count, multiplier, max_len)) in enumerate(audio_specs.items()):
        source = sources[f"{kind}_full"]
        train_source = sources[f"{kind}_train"]
        print(f"Collecting {kind} mini-training leakage groups", flush=True)
        forbidden_groups = collect_training_groups(kind, train_source)
        pool_count = dev_count * max(multiplier, 16)
        print(
            f"Selecting {kind} English holdout from full corpus "
            f"(excluding {len(forbidden_groups)} mini groups)",
            flush=True,
        )
        rows, eligible_rows, scanned_rows = read_holdout_pool(
            kind, source, pool_count, args.seed + offset * 1009,
            args.eval_language, forbidden_groups,
        )
        candidates = prepare_candidates(
            kind, rows, tokenizer, max_len, language=args.eval_language,
        )
        dev = select_language(
            candidates, dev_count, args.seed + offset * 1009 + 1,
            args.eval_language, forbidden_groups=forbidden_groups,
        )
        schema = pq.ParquetFile(source).schema_arrow
        outputs[f"{kind}_dev.parquet"] = (dev, schema)
        dev_groups = {item["group"] for item in dev}
        stats[kind] = {
            "source_rows": pq.ParquetFile(source).metadata.num_rows,
            "scanned_source_rows": scanned_rows,
            "eligible_scanned_rows": eligible_rows,
            "candidate_rows": len(candidates),
            "train": pq.ParquetFile(train_source).metadata.num_rows,
            "dev": len(dev),
            "dev_zh": sum(item["language"] == "zh" for item in dev),
            "dev_en": sum(item["language"] == "en" for item in dev),
            "train_language": "source distribution",
            "dev_source": "full corpus minus mini training groups",
            "group_key": "conversation" if kind == "t2a" else "speaker embedding",
            "group_overlap": len(dev_groups & forbidden_groups),
            "examples": manifest_examples(dev),
        }
        if kind == "a2a":
            train_conversations = collect_conversation_groups(train_source)
            dev_conversations = {
                stable_digest(item["row"].get("conversations") or "") for item in dev
            }
            stats[kind].update({
                "dev_conversation_groups": len(dev_conversations),
                "conversation_overlap": len(train_conversations & dev_conversations),
                "content_scope": "unseen speakers; conversation content may repeat",
            })

    kind = "i2t"
    source = sources["i2t"]
    raw_count = min(
        pq.ParquetFile(source).metadata.num_rows,
        (args.i2t_train + args.i2t_dev) * 6,
    )
    print(f"Selecting English I2T train/dev from {raw_count} candidates", flush=True)
    rows = read_sampled_rows(source, raw_count, args.seed + 2 * 1009)
    candidates = prepare_candidates(
        kind, rows, tokenizer, 768, language=args.eval_language,
    )
    dev = select_language(candidates, args.i2t_dev, args.seed + 2 * 1009 + 1, args.eval_language)
    dev_groups = {item["group"] for item in dev}
    train = select_language(
        candidates, args.i2t_train, args.seed + 2 * 1009 + 2,
        args.eval_language, forbidden_groups=dev_groups,
    )
    schema = pq.ParquetFile(source).schema_arrow
    outputs["i2t_train.parquet"] = (train, schema)
    outputs["i2t_dev.parquet"] = (dev, schema)
    stats[kind] = {
        "source_rows": pq.ParquetFile(source).metadata.num_rows,
        "candidate_rows": len(candidates),
        "train": len(train), "dev": len(dev),
        "train_zh": 0, "dev_zh": 0,
        "train_en": len(train), "dev_en": len(dev),
        "dev_source": "full corpus held out from architecture mini train",
        "group_key": "image SHA-256",
        "group_overlap": len({item["group"] for item in train} & dev_groups),
        "examples": manifest_examples(dev),
    }

    result = {
        "format_version": 4,
        "request": requested,
        "stats": stats,
    }
    # Do not replace any benchmark file until every split has been selected.
    for filename, (items, schema) in outputs.items():
        write_rows(output_dir / filename, items, schema)
    tmp_manifest = manifest_path.with_suffix(".json.tmp")
    tmp_manifest.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_manifest, manifest_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer-path", default=str(root / "model/Qwen3-0.6B"))
    parser.add_argument("--t2a-source", default=str(root / "dataset/sft_t2a.parquet"))
    parser.add_argument("--t2a-train-source", default=str(root / "dataset/sft_t2a_mini.parquet"))
    parser.add_argument("--a2a-source", default=str(root / "dataset/sft_a2a.parquet"))
    parser.add_argument("--a2a-train-source", default=str(root / "dataset/sft_a2a_mini.parquet"))
    parser.add_argument("--i2t-source", default=str(root / "dataset/sft_i2t.parquet"))
    parser.add_argument("--output-dir", default=str(root / "dataset/arch_eval_holdout"))
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--eval-language", choices=["en"], default="en")
    parser.add_argument("--t2a-dev", type=int, default=512)
    parser.add_argument("--a2a-dev", type=int, default=192)
    parser.add_argument("--i2t-train", type=int, default=32768)
    parser.add_argument("--i2t-dev", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    build(parser.parse_args())


if __name__ == "__main__":
    main()
