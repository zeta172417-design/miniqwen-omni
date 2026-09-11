#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")/.." && pwd)
PYTHON=${ARCH_EVAL_PYTHON:-${PROJECT_ROOT}/envs/Omni-ppu/bin/python}
TORCHRUN=${ARCH_EVAL_TORCHRUN:-${PROJECT_ROOT}/envs/Omni-ppu/bin/torchrun}
MODEL_PATH=${ARCH_EVAL_MODEL_PATH:-${PROJECT_ROOT}/model/Qwen3-0.6B}
DATA_DIR=${ARCH_EVAL_DATA_DIR:-${PROJECT_ROOT}/dataset/arch_eval}
EXPERIMENT_NAME=${ARCH_EVAL_NAME:-main_codec_cp_2l_v1}
OUTPUT_ROOT=${PROJECT_ROOT}/out
OUTPUT_NAME=arch_eval/${EXPERIMENT_NAME}
EXPERIMENT_ROOT=${OUTPUT_ROOT}/${OUTPUT_NAME}
CHECKPOINT=${EXPERIMENT_ROOT}/checkpoint
PIPELINE_STATE=${EXPERIMENT_ROOT}/pipeline_stage
RESULT_ROOT=${PROJECT_ROOT}/.runtime/arch_eval/${EXPERIMENT_NAME}
LOG_ROOT=${PROJECT_ROOT}/.runtime/train_logs/arch_eval/${EXPERIMENT_NAME}

NUM_TALKER_LAYERS=${ARCH_EVAL_NUM_TALKER_LAYERS:-6}
TALKER_HIDDEN_SIZE=${ARCH_EVAL_TALKER_HIDDEN_SIZE:-768}
ACCEPT_HIDDEN_LAYER=${ARCH_EVAL_ACCEPT_HIDDEN_LAYER:-14}
STOP_AFTER_STAGE=${ARCH_EVAL_STOP_AFTER_STAGE:-4}
USE_SWANLAB=${ARCH_EVAL_USE_SWANLAB:-1}
RUN_GENERATION=${ARCH_EVAL_RUN_GENERATION:-1}
EVAL_MAX_SAMPLES=${ARCH_EVAL_MAX_SAMPLES:-256}
EVAL_BATCH_SIZE=${ARCH_EVAL_EVAL_BATCH_SIZE:-8}
VISIBLE_DEVICES=${ARCH_EVAL_VISIBLE_DEVICES:-0,1,2,3}
EVAL_DEVICE=${ARCH_EVAL_EVAL_DEVICE:-0}

export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
export PCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTHONWARNINGS=ignore
export MINIQWEN_DATASET_CACHE=${PROJECT_ROOT}/dataset/.miniqwen_arch_eval_cache
export NUMBA_CACHE_DIR=${PROJECT_ROOT}/.runtime/numba_cache

if [[ ! -x "${PYTHON}" || ! -x "${TORCHRUN}" ]]; then
  echo "Omni PPU environment not found. Expected: ${PYTHON}" >&2
  exit 1
fi
if ! [[ "${NUM_TALKER_LAYERS}" =~ ^(4|6|8)$ ]]; then
  echo "ARCH_EVAL_NUM_TALKER_LAYERS must be 4, 6, or 8" >&2
  exit 1
fi
if ! [[ "${STOP_AFTER_STAGE}" =~ ^[0-4]$ ]]; then
  echo "ARCH_EVAL_STOP_AFTER_STAGE must be in [0,4]" >&2
  exit 1
fi
if ! CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" "${PYTHON}" -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() and torch.cuda.device_count() >= 4 else 1)'; then
  echo "Architecture evaluation requires four visible PPU devices, but the runtime cannot see them" >&2
  exit 1
fi

mkdir -p "${EXPERIMENT_ROOT}" "${RESULT_ROOT}" "${LOG_ROOT}" "${NUMBA_CACHE_DIR}"

CONFIG_SIGNATURE="talker_layers=${NUM_TALKER_LAYERS} talker_hidden=${TALKER_HIDDEN_SIZE} bridge_layer=${ACCEPT_HIDDEN_LAYER} mrope=1 modality_boundaries=1 talker_ref_boundaries=1 max_images=4 audio_head=main_codec_predictor cp_layers=2 cp_hidden=768 residual_weight=0.3"
if [[ -f "${RESULT_ROOT}/architecture.conf" ]]; then
  read -r SAVED_SIGNATURE < "${RESULT_ROOT}/architecture.conf"
  if [[ "${SAVED_SIGNATURE}" != "${CONFIG_SIGNATURE}" ]]; then
    echo "Architecture settings changed for existing experiment ${EXPERIMENT_NAME}; use a new ARCH_EVAL_NAME" >&2
    exit 1
  fi
else
  printf '%s\n' "${CONFIG_SIGNATURE}" > "${RESULT_ROOT}/architecture.conf"
fi

if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
  echo "Preparing fixed architecture-evaluation datasets..."
  "${PYTHON}" "${PROJECT_ROOT}/scripts/prepare_arch_eval_data.py" --output-dir "${DATA_DIR}"
fi
if [[ -f "${RESULT_ROOT}/data_manifest.json" ]] && ! cmp -s "${DATA_DIR}/manifest.json" "${RESULT_ROOT}/data_manifest.json"; then
  echo "Dataset manifest changed for existing experiment ${EXPERIMENT_NAME}; use a new ARCH_EVAL_NAME" >&2
  exit 1
fi
cp "${DATA_DIR}/manifest.json" "${RESULT_ROOT}/data_manifest.json"

COMPLETED_STAGE=-1
if [[ -f "${PIPELINE_STATE}" ]]; then
  read -r COMPLETED_STAGE < "${PIPELINE_STATE}"
fi
if ! [[ "${COMPLETED_STAGE}" =~ ^(-1|0|1|2|3|4)$ ]]; then
  echo "Invalid architecture-eval pipeline state: ${COMPLETED_STAGE}" >&2
  exit 1
fi

cleanup_dataset_cache() {
  rm -rf -- "${MINIQWEN_DATASET_CACHE}"
}
trap cleanup_dataset_cache EXIT

build_report() {
  "${PYTHON}" "${PROJECT_ROOT}/scripts/build_arch_eval_report.py" \
    --result-dir "${RESULT_ROOT}" \
    --train-log-dir "${LOG_ROOT}" \
    --checkpoint "${CHECKPOINT}"
}

SWANLAB_ARGS=()
if [[ "${USE_SWANLAB}" == "1" ]]; then
  SWANLAB_ARGS=(--use_swanlab --swanlab_project MiniQwen-Omni-ArchEval)
fi

COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --audio_encoder_dir "${PROJECT_ROOT}/model/SenseVoiceSmall"
  --vision_dir "${PROJECT_ROOT}/model/siglip2-base-p32-256-ve"
  --save_dir "${OUTPUT_ROOT}"
  --save_weight "${OUTPUT_NAME}"
  --num_talker_hidden_layers "${NUM_TALKER_LAYERS}"
  --talker_hidden_size "${TALKER_HIDDEN_SIZE}"
  --accept_hidden_layer "${ACCEPT_HIDDEN_LAYER}"
  --audio_head_type main_codec_predictor
  --code_predictor_num_layers 2
  --code_predictor_hidden_size 768
  --residual_codec_loss_weight 0.3
  --use_mrope 1
  --use_modality_boundaries 1
  --use_talker_ref_boundaries 1
  --max_images 4
  --dtype bfloat16
  --use_moe 0
  --use_compile 0
  --gradient_checkpointing 0
  --dynamic_padding 1
  --save_interval 0
  --save_at_epoch_end 1
  --from_resume 1
  --num_workers 2
  --log_interval 25
  "${SWANLAB_ARGS[@]}"
)

evaluate_stage() {
  local stage_id="$1"
  local stage_name="$2"
  local output=${RESULT_ROOT}/stage-${stage_id}-${stage_name}.json
  echo "Evaluating stage ${stage_id}: ${stage_name}"
  cleanup_dataset_cache
  CUDA_VISIBLE_DEVICES="${EVAL_DEVICE}" "${PYTHON}" "${PROJECT_ROOT}/trainer/eval_arch.py" \
    --checkpoint "${CHECKPOINT}" \
    --data-dir "${DATA_DIR}" \
    --output "${output}" \
    --stage "${stage_id}-${stage_name}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-samples "${EVAL_MAX_SAMPLES}" \
    2>&1 | tee "${RESULT_ROOT}/stage-${stage_id}-${stage_name}-eval.log"
}

run_stage() {
  local stage_id="$1"
  local port="$2"
  local stage_name="$3"
  shift 3
  if (( stage_id <= COMPLETED_STAGE )); then
    echo "Skip completed architecture-eval stage ${stage_id}: ${stage_name}"
    return
  fi
  if (( stage_id > STOP_AFTER_STAGE )); then
    return
  fi

  local attempt_log=${LOG_ROOT}/stage-${stage_id}-$(date -u +%Y%m%dT%H%M%SZ)
  mkdir -p "${attempt_log}"
  echo "Stage ${stage_id} (${stage_name}) logs: ${attempt_log}"
  CUDA_VISIBLE_DEVICES="${VISIBLE_DEVICES}" "${TORCHRUN}" \
    --master_port "${port}" \
    --nproc_per_node 4 \
    --log-dir "${attempt_log}/ranks" \
    --tee 3 \
    --local-ranks-filter=0 \
    "${PROJECT_ROOT}/trainer/train_sft_omni.py" \
    "${COMMON_ARGS[@]}" \
    --stage_id "${stage_id}" \
    --swanlab_run_name "${EXPERIMENT_NAME}-${stage_id}-${stage_name}" \
    "$@" 2>&1 \
    | sed -u -e '/ALINPU INFO/d' -e '/offline_cache\.hpp.*No cache file exist/d' \
    | tee "${attempt_log}/console.log"

  evaluate_stage "${stage_id}" "${stage_name}"
  printf '%s\n' "${stage_id}" > "${PIPELINE_STATE}.tmp"
  mv "${PIPELINE_STATE}.tmp" "${PIPELINE_STATE}"
  COMPLETED_STAGE="${stage_id}"
  build_report
  cleanup_dataset_cache
}

# Stage 0: train the complete mini T2A set as the audio-generation baseline.
# mini dataset. Qwen uses a low differential LR; newly initialised omni
# modules, including the 1024->768 bridge and Talker, use the original 5e-4.
run_stage 0 29710 t2a-full \
  --data_path "${PROJECT_ROOT}/dataset/sft_t2a_mini.parquet" \
  --epochs "${ARCH_EVAL_T2A_EPOCHS:-1}" --batch_size 24 --accumulation_steps 1 --max_seq_len 512 \
  --qwen_learning_rate 1e-5 --omni_learning_rate 5e-4 \
  --from_weight qwen --mode all --train_modules all --save_optimizer_state 1

# Stage 1: align frozen SenseVoice features to Qwen hidden size.
run_stage 1 29711 a2a-projector \
  --data_path "${PROJECT_ROOT}/dataset/sft_a2a_mini.parquet" \
  --epochs 1 --batch_size 16 --accumulation_steps 1 --max_seq_len 640 \
  --qwen_learning_rate 0 --omni_learning_rate 5e-4 \
  --from_weight "${CHECKPOINT}" --mode audio_proj --train_modules audio_proj --save_optimizer_state 1

# Stage 2: low-LR joint A2A tuning on the complete A2A mini dataset. This is
# the stage that teaches Thinker, speaker conditioning and Talker to cooperate.
run_stage 2 29712 a2a-joint \
  --data_path "${PROJECT_ROOT}/dataset/sft_a2a_mini.parquet" \
  --epochs "${ARCH_EVAL_A2A_JOINT_EPOCHS:-1}" --batch_size 12 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 2e-6 --omni_learning_rate 2e-5 \
  --from_weight "${CHECKPOINT}" --mode all --train_modules all --save_optimizer_state 1

# Stage 3: vision-only alignment. Talker is skipped in the forward pass.
run_stage 3 29713 i2t-projector \
  --data_path "${DATA_DIR}/i2t_train.parquet" \
  --epochs 1 --batch_size 16 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 0 --omni_learning_rate 5e-4 \
  --from_weight "${CHECKPOINT}" --mode vision_proj --train_modules vision_proj --save_optimizer_state 1

# Stage 4: low-LR English I2T joint tuning. Talker/audio modules stay frozen,
# and text-only batches skip Talker forward, so vision work cannot erase the
# audio behaviour learned in stages 0-2.
run_stage 4 29714 i2t-joint \
  --data_path "${DATA_DIR}/i2t_train.parquet" \
  --epochs "${ARCH_EVAL_I2T_JOINT_EPOCHS:-1}" \
  --batch_size 12 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 1e-6 --omni_learning_rate 1e-5 \
  --from_weight "${CHECKPOINT}" --mode all --train_modules thinker,text_head,vision_proj --save_optimizer_state 0

if (( COMPLETED_STAGE >= 4 )) && [[ "${RUN_GENERATION}" == "1" ]] && [[ ! -f "${RESULT_ROOT}/generation.done" ]]; then
  echo "Running fixed-seed qualitative generation suite..."
  CUDA_VISIBLE_DEVICES="${EVAL_DEVICE}" \
    MINI_CHECKPOINT="${CHECKPOINT}" \
    MINI_EVAL_OUTPUT="${RESULT_ROOT}/generated" \
    MINI_EVAL_MODE="0,2,3,4" \
    MINI_EVAL_MAX_SAMPLES=1 \
    MINI_EVAL_MAX_NEW_TOKENS=256 \
    MINI_EVAL_PROMPT_LANG=0 \
    bash "${PROJECT_ROOT}/eval_mini.sh" 2>&1 | tee "${RESULT_ROOT}/generation.log"
  "${PYTHON}" "${PROJECT_ROOT}/scripts/eval_generated_speech.py" \
    --generation-log "${RESULT_ROOT}/generation.log" \
    --model "${PROJECT_ROOT}/model/SenseVoiceSmall" \
    --output "${RESULT_ROOT}/speech_metrics.json"
  touch "${RESULT_ROOT}/generation.done"
fi

build_report

echo "Architecture evaluation complete through stage ${COMPLETED_STAGE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Metrics: ${RESULT_ROOT}"
