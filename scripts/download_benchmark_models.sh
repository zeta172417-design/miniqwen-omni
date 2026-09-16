#!/usr/bin/env bash
set -euo pipefail

WORKSPACE=${OMNI_WORKSPACE:-/mnt/workspace/zhaozetao/multimodel/Omni}
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
MINI_OMNI_REV=49f474f4c38f80cf716859bb1b1442df2a8dea46
MINI_OMNI_CODE_REV=75f450ea4e7dc1e52a7fc0e4dcf45b4c3d45ab98

mkdir -p "$WORKSPACE/benchmark-data" "$WORKSPACE/benchmark-results" "$WORKSPACE/mini-omni2"
if [[ ! -e "$WORKSPACE/miniqwen-omni" ]]; then
  ln -s "$PROJECT_ROOT" "$WORKSPACE/miniqwen-omni"
fi

if [[ ! -s "$WORKSPACE/Qwen2.5-Omni-3B/model.safetensors.index.json" ]]; then
  modelscope download --model Qwen/Qwen2.5-Omni-3B --revision master \
    --local_dir "$WORKSPACE/Qwen2.5-Omni-3B" --max-workers 8
fi

if [[ ! -d "$WORKSPACE/mini-omni2/source/.git" ]]; then
  git clone https://github.com/gpt-omni/mini-omni2.git "$WORKSPACE/mini-omni2/source"
fi
git -C "$WORKSPACE/mini-omni2/source" checkout --detach "$MINI_OMNI_CODE_REV"

if [[ ! -s "$WORKSPACE/mini-omni2/checkpoint/lit_model.pth" ]]; then
  python - "$WORKSPACE/mini-omni2/checkpoint" "$MINI_OMNI_REV" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download("gpt-omni/mini-omni2", revision=sys.argv[2], local_dir=sys.argv[1], ignore_patterns=["data/*"], max_workers=8)
PY
fi

python "$PROJECT_ROOT/benchmark/model_manifest.py" --workspace "$WORKSPACE"
