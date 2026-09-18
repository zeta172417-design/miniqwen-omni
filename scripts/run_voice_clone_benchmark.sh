#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=${OMNI_WORKSPACE:-/mnt/workspace/zhaozetao/multimodel/Omni}
OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$WORKSPACE/benchmark-results}
RUN_ID=${1:?Usage: $0 RUN_ID}
RUN_DIR=$OUTPUT_ROOT/$RUN_ID
PROFILE=${OMNI_CLONE_PROFILE:-standard}
DATA_DIR=${OMNI_CLONE_DATA:-$WORKSPACE/benchmark-data/voice-clone-v2.3-$PROFILE}
MAIN_MANIFEST=${OMNI_BENCH_MANIFEST:-$WORKSPACE/benchmark-data/omni-core-v2.3-$PROFILE/manifest.jsonl}
CONFIG=${OMNI_BENCH_CONFIG:-$PROJECT_ROOT/benchmark/configs/models.json}
MODELS=${OMNI_CLONE_MODELS:-miniqwen-v0,miniqwen-v0.1,minimind-3o-moe}
DEVICES=${OMNI_CLONE_DEVICES:-0,1,2,3}
WORKERS_PER_DEVICE=${OMNI_CLONE_WORKERS_PER_DEVICE:-2}
SCORE_DEVICE=${OMNI_CLONE_SCORE_DEVICE:-0}
UTMOS_REPO=${OMNI_BENCH_UTMOS_REPO:-$WORKSPACE/SpeechMOS}
PYTHON_BIN=${MINIQWEN_BENCH_PYTHON:-$(command -v python)}

export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$PROJECT_ROOT/.runtime/numba_cache}
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.runtime/hf_cache}
export TORCH_HOME=${TORCH_HOME:-$WORKSPACE/benchmark-cache/torch}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$NUMBA_CACHE_DIR"

if [[ "$UTMOS_REPO" == "0" ]]; then
  UTMOS_REPO=""
elif [[ ! -f "$UTMOS_REPO/hubconf.py" ]]; then
  echo "UTMOS repository is missing: $UTMOS_REPO" >&2
  echo "Install SpeechMOS there, or set OMNI_BENCH_UTMOS_REPO=0 to explicitly disable MOS scoring." >&2
  exit 2
fi

if [[ "$PROFILE" == "quick" ]]; then
  CLONE_SPEAKERS=${OMNI_CLONE_SPEAKERS:-4}
  CLONE_TARGETS=${OMNI_CLONE_TARGETS_PER_SPEAKER:-2}
else
  CLONE_SPEAKERS=${OMNI_CLONE_SPEAKERS:-8}
  CLONE_TARGETS=${OMNI_CLONE_TARGETS_PER_SPEAKER:-2}
fi

PREPARE_CLONE_ARGS=()
if [[ -f "$MAIN_MANIFEST" ]]; then
  PREPARE_CLONE_ARGS+=(--exclude-manifest "$MAIN_MANIFEST")
fi
"$PYTHON_BIN" "$PROJECT_ROOT/benchmark/prepare_voice_clone.py" --output "$DATA_DIR" \
  --speakers "$CLONE_SPEAKERS" --targets-per-speaker "$CLONE_TARGETS" \
  "${PREPARE_CLONE_ARGS[@]}"

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra PHYSICAL_DEVICE_LIST <<< "$DEVICES"
if (( WORKERS_PER_DEVICE < 1 )); then
  echo "OMNI_CLONE_WORKERS_PER_DEVICE must be positive" >&2
  exit 2
fi
DEVICE_LIST=()
for device in "${PHYSICAL_DEVICE_LIST[@]}"; do
  for ((worker = 0; worker < WORKERS_PER_DEVICE; worker++)); do
    DEVICE_LIST+=("$device")
  done
done
NUM_SHARDS=${#DEVICE_LIST[@]}
if (( NUM_SHARDS < 1 )); then
  echo "OMNI_CLONE_DEVICES must contain at least one device" >&2
  exit 2
fi
for model in "${MODEL_LIST[@]}"; do
  echo "Concurrency: $WORKERS_PER_DEVICE process(es) per PPU, $NUM_SHARDS total data shards"
  echo "=== Voice clone: $model ($NUM_SHARDS data-parallel shards) ==="
  pids=()
  for shard_index in "${!DEVICE_LIST[@]}"; do
    device=${DEVICE_LIST[$shard_index]}
    echo "Voice clone: $model shard $((shard_index + 1))/$NUM_SHARDS on PPU $device"
    env CUDA_VISIBLE_DEVICES="$device" "$PYTHON_BIN" -u "$PROJECT_ROOT/benchmark/run_voice_clone.py" \
      --config "$CONFIG" --model "$model" --manifest "$DATA_DIR/manifest.jsonl" \
      --run-dir "$RUN_DIR" --device cuda --resume \
      --num-shards "$NUM_SHARDS" --shard-index "$shard_index" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
  "$PYTHON_BIN" "$PROJECT_ROOT/benchmark/merge_shards.py" \
    --run-dir "$RUN_DIR/voice_clone/$model" --num-shards "$NUM_SHARDS" --skip-metrics
  echo "=== Voice clone generation complete: $model ==="
done

echo "Voice-clone scoring on PPU $SCORE_DEVICE"
UTMOS_ARGS=()
if [[ -n "$UTMOS_REPO" ]]; then
  UTMOS_ARGS+=(--utmos-repo "$UTMOS_REPO")
fi
env CUDA_VISIBLE_DEVICES="$SCORE_DEVICE" "$PYTHON_BIN" -u "$PROJECT_ROOT/benchmark/evaluate_voice_clone.py" \
  --run-dir "$RUN_DIR" --manifest "$DATA_DIR/manifest.jsonl" --models "$MODELS" --device cuda:0 \
  "${UTMOS_ARGS[@]}"

if [[ ${OMNI_CLONE_BUILD_REPORT:-1} == 1 ]]; then
  "$PYTHON_BIN" "$PROJECT_ROOT/benchmark/build_report.py" --run-dir "$RUN_DIR"
  echo "Report: $RUN_DIR/REPORT.md"
fi
