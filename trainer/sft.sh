#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
exec "$ROOT_DIR/.venv/bin/python" "$ROOT_DIR/scripts/run_sft.py" "$@"
