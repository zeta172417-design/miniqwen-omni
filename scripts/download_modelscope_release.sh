#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAMESPACE="${MODELSCOPE_NAMESPACE:-peachPPP}"
PRETRAIN_REPO="${MODELSCOPE_PRETRAIN_REPO:-${NAMESPACE}/MiniQwen-Omni-V1-Pretrain}"
SFT_REPO="${MODELSCOPE_SFT_REPO:-${NAMESPACE}/MiniQwen-Omni-V1-SFT}"
V1_REPO="${MODELSCOPE_V1_REPO:-${NAMESPACE}/MiniQwen-Omni-V1}"
BASE_REPO="${MODELSCOPE_BASE_REPO:-${NAMESPACE}/MiniQwen-Omni-V1-Base}"

mkdir -p "${ROOT_DIR}/dataset" \
  "${ROOT_DIR}/out/miniqwen_sft/last/checkpoint" \
  "${ROOT_DIR}/out/miniqwen_pretrain/tts/last/checkpoint"

echo "Downloading prepared pretrain data to ${ROOT_DIR}/dataset"
modelscope download --dataset "${PRETRAIN_REPO}" --local_dir "${ROOT_DIR}/dataset"

echo "Downloading prepared SFT data and media to ${ROOT_DIR}/dataset"
modelscope download --dataset "${SFT_REPO}" --local_dir "${ROOT_DIR}/dataset"

echo "Downloading V1 model to out/miniqwen_sft/last/checkpoint"
modelscope download --model "${V1_REPO}" \
  --local_dir "${ROOT_DIR}/out/miniqwen_sft/last/checkpoint"

echo "Downloading V1-Base model to out/miniqwen_pretrain/tts/last/checkpoint"
modelscope download --model "${BASE_REPO}" \
  --local_dir "${ROOT_DIR}/out/miniqwen_pretrain/tts/last/checkpoint"

echo "Release download complete. Run: python scripts/validate_downloaded_release.py"
