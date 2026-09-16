#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=${OMNI_WORKSPACE:-/mnt/workspace/zhaozetao/multimodel/Omni}
OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$WORKSPACE/benchmark-results}
RUN_ID=${1:?Usage: $0 RUN_ID}
RUN_DIR=$OUTPUT_ROOT/$RUN_ID
DATA_DIR=${OMNI_CLONE_DATA:-$WORKSPACE/benchmark-data/voice-clone-v1}
CONFIG=${OMNI_BENCH_CONFIG:-$PROJECT_ROOT/benchmark/configs/models.json}
MODELS=${OMNI_CLONE_MODELS:-miniqwen-v0,miniqwen-v0.1}
DEVICES=${OMNI_CLONE_DEVICES:-0,1}
SCORE_DEVICE=${OMNI_CLONE_SCORE_DEVICE:-2}
PYTHON_BIN=${MINIQWEN_BENCH_PYTHON:-$(command -v python)}

export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$PROJECT_ROOT/.runtime/numba_cache}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$NUMBA_CACHE_DIR"

"$PYTHON_BIN" "$PROJECT_ROOT/benchmark/prepare_voice_clone.py" --output "$DATA_DIR"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra DEVICE_LIST <<< "$DEVICES"
pids=()
for index in "${!MODEL_LIST[@]}"; do
  model=${MODEL_LIST[$index]}
  device=${DEVICE_LIST[$((index % ${#DEVICE_LIST[@]}))]}
  echo "Voice clone: $model on PPU $device"
  env CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" -u "$PROJECT_ROOT/benchmark/run_voice_clone.py" \
    --config "$CONFIG" --model "$model" --manifest "$DATA_DIR/manifest.jsonl" \
    --run-dir "$RUN_DIR" --device cuda --resume &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "Voice-clone scoring on PPU $SCORE_DEVICE"
env CUDA_VISIBLE_DEVICES="$SCORE_DEVICE" "$PYTHON_BIN" -u "$PROJECT_ROOT/benchmark/evaluate_voice_clone.py" \
  --run-dir "$RUN_DIR" --manifest "$DATA_DIR/manifest.jsonl" --models "$MODELS" --device cuda:0

if [[ ${OMNI_CLONE_BUILD_REPORT:-1} == 1 ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/benchmark/build_report.py" --run-dir "$RUN_DIR"
  echo "Report: $RUN_DIR/REPORT.md"
fi
