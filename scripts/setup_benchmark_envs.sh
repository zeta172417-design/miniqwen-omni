#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=${OMNI_WORKSPACE:-/mnt/workspace/zhaozetao/multimodel/Omni}

# Safe for Omni-ppu: these packages do not replace torch/torchvision/torchaudio.
python -m pip install --no-deps qwen-omni-utils==0.0.9 av==18.1.0

# Mini-Omni2's upstream package pins conflict with the PPU Torch build. Keep
# only its Python-level compatibility packages in an isolated import directory.
MINIOMNI_DEPS=$PROJECT_ROOT/.runtime/mini_omni2_ppu_deps
mkdir -p "$MINIOMNI_DEPS"
python -m pip install --target "$MINIOMNI_DEPS" --no-deps \
  -r "$PROJECT_ROOT/benchmark/requirements-mini-omni2-ppu.txt"

echo "Mini-Omni2 PPU dependencies: $MINIOMNI_DEPS"
