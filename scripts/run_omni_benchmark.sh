#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=${OMNI_WORKSPACE:-/mnt/workspace/zhaozetao/multimodel/Omni}
CONFIG=${OMNI_BENCH_CONFIG:-$PROJECT_ROOT/benchmark/configs/models.json}
RUN_ID=${OMNI_BENCH_RUN_ID:-standard-$(date -u +%Y%m%dT%H%M%SZ)}
MODELS=${OMNI_BENCH_MODELS:-miniqwen-v0,miniqwen-v0.1,qwen2.5-omni-3b,mini-omni2}
OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$WORKSPACE/benchmark-results}
ASR_MODEL=${OMNI_BENCH_ASR_MODEL:-$PROJECT_ROOT/model/SenseVoiceSmall}
DEVICES=${OMNI_BENCH_DEVICES:-0,1,2,3}
PARALLEL=${OMNI_BENCH_PARALLEL:-1}
ASR_DEVICE=${OMNI_BENCH_ASR_DEVICE:-cuda:0}
VOICE_CLONE=${OMNI_BENCH_VOICE_CLONE:-1}
LIMIT=${OMNI_BENCH_LIMIT:-0}
RUN_MODE=${1:-full}
DEFAULT_PYTHON=$(command -v python)
MINIQWEN_PYTHON=${MINIQWEN_BENCH_PYTHON:-$DEFAULT_PYTHON}
QWEN_PYTHON=${QWEN_BENCH_PYTHON:-$DEFAULT_PYTHON}
MINIOMNI_VENV=$WORKSPACE/mini-omni2/.venv/bin/python
MINIOMNI_PYTHON=${MINIOMNI_BENCH_PYTHON:-$MINIOMNI_VENV}
MINIOMNI_SOURCE=$WORKSPACE/mini-omni2/source
MINIOMNI_DEPS=$PROJECT_ROOT/.runtime/mini_omni2_ppu_deps
SKIP_AUDIO_SCORING=0

case "$RUN_MODE" in
  full)
    ;;
  smoke)
    CONFIG=$PROJECT_ROOT/benchmark/configs/smoke.json
    MODELS=mock
    RUN_ID=${OMNI_BENCH_RUN_ID:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
    OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$PROJECT_ROOT/.runtime/benchmark-results}
    LIMIT=0
    SKIP_AUDIO_SCORING=1
    ;;
  mini-omni2-smoke)
    MODELS=mini-omni2
    RUN_ID=${OMNI_BENCH_RUN_ID:-mini-omni2-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
    OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$PROJECT_ROOT/.runtime/benchmark-results}
    LIMIT=${OMNI_BENCH_LIMIT:-1}
    SKIP_AUDIO_SCORING=1
    ;;
  *)
    echo "Usage: $0 [smoke|mini-omni2-smoke|full]" >&2
    exit 2
    ;;
esac

if [[ ! -x "$MINIOMNI_PYTHON" ]]; then
  # Mini-Omni2 also runs with the project's PPU Python when its small set of
  # compatibility packages is isolated under .runtime.
  MINIOMNI_PYTHON=$DEFAULT_PYTHON
fi

RUN_ARGS=()
if (( LIMIT > 0 )); then
  RUN_ARGS+=(--limit "$LIMIT")
fi

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra DEVICE_LIST <<< "$DEVICES"

run_model() {
  local model=$1
  local device=$2
  echo "Running $model on PPU $device -> $RUN_ID"
  case "$model" in
    mini-omni2)
      PYTHON_BIN=$MINIOMNI_PYTHON
      if [[ "$PYTHON_BIN" == "$DEFAULT_PYTHON" && ! -d "$MINIOMNI_DEPS/lightning" ]]; then
        echo "Missing Mini-Omni2 PPU compatibility dependencies: $MINIOMNI_DEPS" >&2
        exit 2
      fi
      MODEL_PYTHONPATH=$MINIOMNI_DEPS:$MINIOMNI_SOURCE${PYTHONPATH:+:$PYTHONPATH}
      ;;
    qwen2.5-omni-3b)
      PYTHON_BIN=$QWEN_PYTHON
      MODEL_PYTHONPATH=${PYTHONPATH:-}
      ;;
    *)
      PYTHON_BIN=$MINIQWEN_PYTHON
      MODEL_PYTHONPATH=${PYTHONPATH:-}
      ;;
  esac
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Missing interpreter for $model: $PYTHON_BIN" >&2
    exit 2
  fi
  env CUDA_VISIBLE_DEVICES="$device" PYTHONPATH="$MODEL_PYTHONPATH" \
    HF_HOME="$PROJECT_ROOT/.runtime/hf_cache" \
    "$PYTHON_BIN" -u "$PROJECT_ROOT/benchmark/run_benchmark.py" \
    --config "$CONFIG" --model "$model" --run-id "$RUN_ID" \
    --output-root "$OUTPUT_ROOT" --resume "${RUN_ARGS[@]}"
}

run_phase() {
  local pids=()
  local index=0
  for model in "${MODEL_LIST[@]}"; do
    device=${DEVICE_LIST[$((index % ${#DEVICE_LIST[@]}))]}
    if (( PARALLEL )); then
      run_model "$model" "$device" &
      pids+=("$!")
    else
      run_model "$model" "$device"
    fi
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
}

score_model() {
  local model=$1
  local device=$2
  echo "Scoring speech for $model on PPU $device"
  env CUDA_VISIBLE_DEVICES="$device" \
    NUMBA_CACHE_DIR="$PROJECT_ROOT/.runtime/numba_cache" \
    "$MINIQWEN_PYTHON" -u "$PROJECT_ROOT/benchmark/evaluate_audio.py" \
    --input "$OUTPUT_ROOT/$RUN_ID/$model/per_sample.jsonl" \
    --asr "$ASR_MODEL" --device "$ASR_DEVICE"
}

run_phase

if (( SKIP_AUDIO_SCORING == 0 )); then
  pids=()
  index=0
  for model in "${MODEL_LIST[@]}"; do
    device=${DEVICE_LIST[$((index % ${#DEVICE_LIST[@]}))]}
    if (( PARALLEL )); then
      score_model "$model" "$device" &
      pids+=("$!")
    else
      score_model "$model" "$device"
    fi
    index=$((index + 1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

if [[ "$RUN_MODE" == "full" && "$VOICE_CLONE" == 1 ]]; then
  OMNI_CLONE_BUILD_REPORT=0 OMNI_BENCH_OUTPUT_ROOT="$OUTPUT_ROOT" \
    "$PROJECT_ROOT/scripts/run_voice_clone_benchmark.sh" "$RUN_ID"
fi

"$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/build_report.py" \
  --run-dir "$OUTPUT_ROOT/$RUN_ID"
echo "Report: $OUTPUT_ROOT/$RUN_ID/REPORT.md"
