#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORKSPACE=${OMNI_WORKSPACE:-/mnt/workspace/zhaozetao/multimodel/Omni}
CONFIG=${OMNI_BENCH_CONFIG:-$PROJECT_ROOT/benchmark/configs/models.json}
RUN_ID=${OMNI_BENCH_RUN_ID:-omni-benchmark-v0-standard-$(date -u +%Y%m%dT%H%M%SZ)}
MODELS=${OMNI_BENCH_MODELS:-miniqwen-v0,miniqwen-v0.1,minimind-3o-moe,qwen2.5-omni-3b,mini-omni2}
OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$WORKSPACE/benchmark-results}
ASR_MODEL=${OMNI_BENCH_ASR_MODEL:-$PROJECT_ROOT/model/SenseVoiceSmall}
DEVICES=${OMNI_BENCH_DEVICES:-0,1,2,3}
WORKERS_PER_DEVICE=${OMNI_BENCH_WORKERS_PER_DEVICE:-2}
LARGE_MODEL_WORKERS_PER_DEVICE=${OMNI_BENCH_LARGE_MODEL_WORKERS_PER_DEVICE:-1}
BATCH_SIZE=${OMNI_BENCH_BATCH_SIZE:-4}
PARALLEL=${OMNI_BENCH_PARALLEL:-1}
ASR_DEVICE=${OMNI_BENCH_ASR_DEVICE:-cuda:0}
UTMOS_REPO=${OMNI_BENCH_UTMOS_REPO:-$WORKSPACE/SpeechMOS}
VOICE_CLONE=${OMNI_BENCH_VOICE_CLONE:-1}
LIMIT=${OMNI_BENCH_LIMIT:-0}
RUN_MODE=${1:-standard}
DEFAULT_PYTHON=$(command -v python)
MINIQWEN_PYTHON=${MINIQWEN_BENCH_PYTHON:-$DEFAULT_PYTHON}
QWEN_PYTHON=${QWEN_BENCH_PYTHON:-$DEFAULT_PYTHON}
MINIOMNI_VENV=$WORKSPACE/mini-omni2/.venv/bin/python
MINIOMNI_PYTHON=${MINIOMNI_BENCH_PYTHON:-$MINIOMNI_VENV}
MINIOMNI_SOURCE=$WORKSPACE/mini-omni2/source
MINIOMNI_DEPS=$PROJECT_ROOT/.runtime/mini_omni2_ppu_deps
SKIP_AUDIO_SCORING=0

export TORCH_HOME=${TORCH_HOME:-$WORKSPACE/benchmark-cache/torch}
export HF_HOME=${HF_HOME:-$PROJECT_ROOT/.runtime/hf_cache}
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$PROJECT_ROOT/.runtime/numba_cache}
export PYTHONWARNINGS=${PYTHONWARNINGS:-ignore}
export TRANSFORMERS_VERBOSITY=${TRANSFORMERS_VERBOSITY:-error}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export DATASETS_DISABLE_PROGRESS_BARS=1
mkdir -p "$NUMBA_CACHE_DIR"
if [[ "$UTMOS_REPO" == "0" ]]; then
  UTMOS_REPO=""
elif [[ ! -f "$UTMOS_REPO/hubconf.py" ]]; then
  echo "UTMOS repository is missing: $UTMOS_REPO" >&2
  echo "Install SpeechMOS there, or set OMNI_BENCH_UTMOS_REPO=0 to explicitly disable MOS scoring." >&2
  exit 2
fi

case "$RUN_MODE" in
  standard|full)
    PROFILE=standard
    DATA_CONFIG=$PROJECT_ROOT/benchmark/configs/standard.json
    DATA_DIR=$WORKSPACE/benchmark-data/omni-core-v2.3-standard
    ;;
  quick)
    PROFILE=quick
    DATA_CONFIG=$PROJECT_ROOT/benchmark/configs/quick.json
    DATA_DIR=$WORKSPACE/benchmark-data/omni-core-v2.3-quick
    RUN_ID=${OMNI_BENCH_RUN_ID:-omni-benchmark-v0-quick-$(date -u +%Y%m%dT%H%M%SZ)}
    ;;
  smoke)
    PROFILE=smoke
    CONFIG=$PROJECT_ROOT/benchmark/configs/smoke.json
    MODELS=mock
    RUN_ID=${OMNI_BENCH_RUN_ID:-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
    OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$PROJECT_ROOT/.runtime/benchmark-results}
    LIMIT=0
    SKIP_AUDIO_SCORING=1
    ;;
  mini-omni2-smoke)
    PROFILE=smoke
    MODELS=mini-omni2
    RUN_ID=${OMNI_BENCH_RUN_ID:-mini-omni2-smoke-$(date -u +%Y%m%dT%H%M%SZ)}
    OUTPUT_ROOT=${OMNI_BENCH_OUTPUT_ROOT:-$PROJECT_ROOT/.runtime/benchmark-results}
    LIMIT=${OMNI_BENCH_LIMIT:-1}
    SKIP_AUDIO_SCORING=1
    ;;
  *)
    echo "Usage: $0 [quick|standard|smoke|mini-omni2-smoke]" >&2
    exit 2
    ;;
esac

LOG_DIR=$OUTPUT_ROOT/$RUN_ID/logs
mkdir -p "$LOG_DIR"
echo "Omni Benchmark $PROFILE | run=$RUN_ID | models=$(tr ',' ' ' <<< "$MODELS")"
echo "Detailed logs: $LOG_DIR"

if [[ ! -x "$MINIOMNI_PYTHON" ]]; then
  # Mini-Omni2 also runs with the project's PPU Python when its small set of
  # compatibility packages is isolated under .runtime.
  MINIOMNI_PYTHON=$DEFAULT_PYTHON
fi

RUN_ARGS=()
if (( LIMIT > 0 )); then
  RUN_ARGS+=(--limit "$LIMIT")
fi
if [[ "$PROFILE" != "smoke" ]]; then
  MANIFEST=${OMNI_BENCH_MANIFEST:-$DATA_DIR/manifest.jsonl}
  PREPARE_ARGS=()
  if [[ "$PROFILE" == "quick" ]]; then
    PREPARE_ARGS+=(--skip-decontamination)
  fi
  if [[ ! -f "$MANIFEST" ]]; then
    echo "[Setup] Preparing v2.3 $PROFILE manifest"
    if ! "$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/prepare_data.py" \
      --config "$DATA_CONFIG" --output "$DATA_DIR" "${PREPARE_ARGS[@]}" \
      >"$LOG_DIR/prepare-data.log" 2>&1; then
      echo "ERROR: data preparation failed; tail of $LOG_DIR/prepare-data.log" >&2
      tail -30 "$LOG_DIR/prepare-data.log" >&2
      exit 1
    fi
  elif ! "$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/validate_manifest.py" \
      "$MANIFEST" --config "$DATA_CONFIG" --expected-builder 2.3.0 \
      >"$LOG_DIR/validate-manifest.log" 2>&1; then
    echo "[Setup] Rebuilding stale manifest as v2.3"
    if ! "$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/prepare_data.py" \
      --config "$DATA_CONFIG" --output "$DATA_DIR" "${PREPARE_ARGS[@]}" \
      >"$LOG_DIR/prepare-data.log" 2>&1; then
      echo "ERROR: data preparation failed; tail of $LOG_DIR/prepare-data.log" >&2
      tail -30 "$LOG_DIR/prepare-data.log" >&2
      exit 1
    fi
  fi
  "$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/validate_manifest.py" \
    "$MANIFEST" --config "$DATA_CONFIG" --expected-builder 2.3.0 \
    >"$LOG_DIR/validate-manifest.log" 2>&1
  RUN_ARGS+=(--manifest "$MANIFEST")
  TOTAL_SAMPLES=$(wc -l < "$MANIFEST")
  if (( LIMIT > 0 && LIMIT < TOTAL_SAMPLES )); then
    TOTAL_SAMPLES=$LIMIT
  fi
  echo "[Setup] Manifest ready: $TOTAL_SAMPLES samples"
else
  TOTAL_SAMPLES=1
fi

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra PHYSICAL_DEVICE_LIST <<< "$DEVICES"
if (( ${#PHYSICAL_DEVICE_LIST[@]} < 1 )); then
  echo "OMNI_BENCH_DEVICES must contain at least one device" >&2
  exit 2
fi
if (( WORKERS_PER_DEVICE < 1 || LARGE_MODEL_WORKERS_PER_DEVICE < 1 || BATCH_SIZE < 1 )); then
  echo "benchmark workers and batch size must be positive" >&2
  exit 2
fi

set_active_devices() {
  local model=$1
  local workers=$WORKERS_PER_DEVICE
  local device worker
  if [[ "$model" == "qwen2.5-omni-3b" ]]; then
    workers=$LARGE_MODEL_WORKERS_PER_DEVICE
  fi
  DEVICE_LIST=()
  for device in "${PHYSICAL_DEVICE_LIST[@]}"; do
    for ((worker = 0; worker < workers; worker++)); do
      DEVICE_LIST+=("$device")
    done
  done
  NUM_SHARDS=${#DEVICE_LIST[@]}
}

count_phase_rows() {
  local run_dir=$1
  local stem=$2
  if (( NUM_SHARDS > 1 )); then
    find "$run_dir" -maxdepth 1 -type f -name "$stem.shard-*-of-*.jsonl" \
      -exec cat {} + 2>/dev/null | wc -l
  elif [[ -f "$run_dir/$stem.jsonl" ]]; then
    wc -l < "$run_dir/$stem.jsonl"
  else
    echo 0
  fi
}

monitor_phase() {
  local label=$1
  local run_dir=$2
  local stem=$3
  local last_bucket=-1 rows percent bucket
  while true; do
    rows=$(count_phase_rows "$run_dir" "$stem")
    (( rows > TOTAL_SAMPLES )) && rows=$TOTAL_SAMPLES
    percent=$((rows * 100 / TOTAL_SAMPLES))
    bucket=$((percent / 5))
    if (( bucket > last_bucket )); then
      printf '[Model %d/%d][%s] %d/%d (%d%%)\n' \
        "$CURRENT_MODEL_INDEX" "$MODEL_COUNT" "$label" "$rows" "$TOTAL_SAMPLES" "$percent"
      last_bucket=$bucket
    fi
    sleep 5
  done
}

run_model_shard() {
  local model=$1
  local device=$2
  local shard_index=$3
  local PYTHON_BIN MODEL_PYTHONPATH inference_batch=1
  echo "Running $model shard $((shard_index + 1))/$NUM_SHARDS on PPU $device -> $RUN_ID"
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
    miniqwen-v0.1)
      PYTHON_BIN=$MINIQWEN_PYTHON
      MODEL_PYTHONPATH=${PYTHONPATH:-}
      inference_batch=$BATCH_SIZE
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
    --output-root "$OUTPUT_ROOT" --resume \
    --num-shards "$NUM_SHARDS" --shard-index "$shard_index" \
    --batch-size "$inference_batch" "${RUN_ARGS[@]}"
}

generate_model() {
  local model=$1
  local pids=()
  local logs=()
  local shard_index device pid
  local run_dir=$OUTPUT_ROOT/$RUN_ID/$model
  echo "[Model $CURRENT_MODEL_INDEX/$MODEL_COUNT][$model] generation on $NUM_SHARDS data shards"
  for shard_index in "${!DEVICE_LIST[@]}"; do
    device=${DEVICE_LIST[$shard_index]}
    log="$LOG_DIR/$model.generate.shard-$shard_index.log"
    if (( PARALLEL )); then
      run_model_shard "$model" "$device" "$shard_index" >"$log" 2>&1 &
      pids+=("$!")
      logs+=("$log")
    else
      if ! run_model_shard "$model" "$device" "$shard_index" >"$log" 2>&1; then
        echo "ERROR: $model generation failed; tail of $log" >&2
        tail -30 "$log" >&2
        return 1
      fi
    fi
  done
  if (( PARALLEL )); then
    monitor_phase "$model generation" "$run_dir" per_sample &
    local monitor_pid=$! failed=0 index
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        echo "ERROR: $model generation shard failed; tail of ${logs[$index]}" >&2
        tail -30 "${logs[$index]}" >&2
        failed=1
      fi
    done
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    (( failed == 0 )) || return 1
  fi
  echo "[Model $CURRENT_MODEL_INDEX/$MODEL_COUNT][$model generation] $TOTAL_SAMPLES/$TOTAL_SAMPLES (100%)"
  "$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/merge_shards.py" \
    --run-dir "$run_dir" --num-shards "$NUM_SHARDS" >"$LOG_DIR/$model.merge.log" 2>&1
}

score_model_shard() {
  local model=$1
  local device=$2
  local shard_index=$3
  local suffix
  local utmos_args=()
  if [[ -n "$UTMOS_REPO" ]]; then
    utmos_args+=(--utmos-repo "$UTMOS_REPO")
  fi
  printf -v suffix 'shard-%02d-of-%02d' "$shard_index" "$NUM_SHARDS"
  echo "Scoring $model $suffix on PPU $device"
  env CUDA_VISIBLE_DEVICES="$device" \
    NUMBA_CACHE_DIR="$PROJECT_ROOT/.runtime/numba_cache" \
    "$MINIQWEN_PYTHON" -u "$PROJECT_ROOT/benchmark/evaluate_audio.py" \
    --input "$OUTPUT_ROOT/$RUN_ID/$model/per_sample.$suffix.jsonl" \
    --output "$OUTPUT_ROOT/$RUN_ID/$model/per_sample_scored.$suffix.jsonl" \
    --asr "$ASR_MODEL" --device "$ASR_DEVICE" --skip-metrics --shard-label "$suffix" \
    --reference-cache "$OUTPUT_ROOT/$RUN_ID/audio_reference_cache" \
    "${utmos_args[@]}"
}

score_model_parallel() {
  local model=$1
  local pids=()
  local logs=()
  local shard_index device pid
  local run_dir=$OUTPUT_ROOT/$RUN_ID/$model
  echo "[Model $CURRENT_MODEL_INDEX/$MODEL_COUNT][$model] speech scoring"
  for shard_index in "${!DEVICE_LIST[@]}"; do
    device=${DEVICE_LIST[$shard_index]}
    log="$LOG_DIR/$model.score.shard-$shard_index.log"
    if (( PARALLEL )); then
      score_model_shard "$model" "$device" "$shard_index" >"$log" 2>&1 &
      pids+=("$!")
      logs+=("$log")
    else
      if ! score_model_shard "$model" "$device" "$shard_index" >"$log" 2>&1; then
        echo "ERROR: $model speech scoring failed; tail of $log" >&2
        tail -30 "$log" >&2
        return 1
      fi
    fi
  done
  if (( PARALLEL )); then
    monitor_phase "$model speech score" "$run_dir" per_sample_scored &
    local monitor_pid=$! failed=0 index
    for index in "${!pids[@]}"; do
      if ! wait "${pids[$index]}"; then
        echo "ERROR: $model scoring shard failed; tail of ${logs[$index]}" >&2
        tail -30 "${logs[$index]}" >&2
        failed=1
      fi
    done
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    (( failed == 0 )) || return 1
  fi
  echo "[Model $CURRENT_MODEL_INDEX/$MODEL_COUNT][$model speech score] $TOTAL_SAMPLES/$TOTAL_SAMPLES (100%)"
  "$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/merge_shards.py" \
    --run-dir "$OUTPUT_ROOT/$RUN_ID/$model" --num-shards "$NUM_SHARDS" \
    --stem per_sample_scored >"$LOG_DIR/$model.merge-scored.log" 2>&1
}

MODEL_COUNT=${#MODEL_LIST[@]}
CURRENT_MODEL_INDEX=0
for model in "${MODEL_LIST[@]}"; do
  CURRENT_MODEL_INDEX=$((CURRENT_MODEL_INDEX + 1))
  set_active_devices "$model"
  generate_model "$model"
  if (( SKIP_AUDIO_SCORING == 0 )); then
    score_model_parallel "$model"
  fi
  echo "[Model $CURRENT_MODEL_INDEX/$MODEL_COUNT][$model] complete"
done

if [[ "$PROFILE" != "smoke" && "$VOICE_CLONE" == 1 ]]; then
  echo "[Voice clone] generation and scoring started"
  OMNI_CLONE_PROFILE="$PROFILE" OMNI_CLONE_BUILD_REPORT=0 OMNI_BENCH_OUTPUT_ROOT="$OUTPUT_ROOT" \
    "$PROJECT_ROOT/scripts/run_voice_clone_benchmark.sh" "$RUN_ID" \
    >"$LOG_DIR/voice-clone.log" 2>&1 || {
      echo "ERROR: voice-clone evaluation failed; tail of $LOG_DIR/voice-clone.log" >&2
      tail -30 "$LOG_DIR/voice-clone.log" >&2
      exit 1
    }
  echo "[Voice clone] complete"
fi

"$MINIQWEN_PYTHON" "$PROJECT_ROOT/benchmark/build_report.py" \
  --run-dir "$OUTPUT_ROOT/$RUN_ID" >"$LOG_DIR/build-report.log" 2>&1
echo "Complete. Report: $OUTPUT_ROOT/$RUN_ID/REPORT.md"
