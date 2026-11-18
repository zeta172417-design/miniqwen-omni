#!/usr/bin/env python3
"""Audit prepared datasets before publishing them to ModelScope.

This is read-only. It inventories Parquet row counts/sizes and samples media
references so a dataset repository does not contain broken local paths.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def audit(root: Path) -> dict:
    files = []
    path_examples: dict[str, list[str]] = {}
    for path in sorted(root.rglob("*.parquet")):
        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        rows = parquet.metadata.num_rows
        item = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "rows": rows,
            "columns": names,
        }
        for name in names:
            if "path" not in name:
                continue
            try:
                values = parquet.read(columns=[name]).to_pylist()
            except Exception as exc:  # pragma: no cover - corrupt input is reported
                item.setdefault("errors", []).append(f"{name}: {exc}")
                continue
            examples = path_examples.setdefault(name, [])
            for row in values:
                value = row[name]
                candidates = value if isinstance(value, list) else [value]
                for candidate in candidates:
                    if isinstance(candidate, str) and candidate not in examples:
                        examples.append(candidate)
                    if len(examples) >= 10:
                        break
                if len(examples) >= 10:
                    break
        files.append(item)
    total_bytes = sum(item["bytes"] for item in files)
    total_rows = sum(item["rows"] for item in files)
    local_references = {
        name: values for name, values in path_examples.items()
        if any(value.startswith(("/", "dataset/", "./")) for value in values)
    }
    return {
        "root": str(root),
        "parquet_files": len(files),
        "total_rows": total_rows,
        "total_bytes": total_bytes,
        "total_gib": round(total_bytes / 1024**3, 3),
        "files": files,
        "path_examples": path_examples,
        "local_path_fields": local_references,
        "upload_ready": not bool(local_references),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="+", help="Dataset roots to audit")
    parser.add_argument("--output", type=Path, default=Path("dataset/modelscope-upload-audit.json"))
    args = parser.parse_args()
    result = {str(root): audit(root.resolve()) for root in args.root}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for name, value in result.items():
        state = "READY" if value["upload_ready"] else "MEDIA_PATHS_REQUIRE_PACKAGING"
        print(f"{name}: {state}, {value['parquet_files']} files, "
              f"{value['total_rows']} rows, {value['total_gib']:.2f} GiB")
    print(f"Audit written to {args.output}")


if __name__ == "__main__":
    main()
