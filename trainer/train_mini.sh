#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export NCCL_DEBUG=ERROR
export PCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_DISTRIBUTED_DEBUG=OFF
export PYTHONWARNINGS=ignore
export MINIQWEN_DATASET_CACHE=../dataset/.miniqwen_mini_cache

MODEL_PATH=../model/Qwen3-0.6B
OUTPUT_ROOT=../out
# The same-frame head is format-v5 and must not reuse delayed-head weights.
OUTPUT_NAME=miniqwen_omni_mini_main_codec_cp_v5
CHECKPOINT=${OUTPUT_ROOT}/${OUTPUT_NAME}/checkpoint
PIPELINE_STATE=${OUTPUT_ROOT}/${OUTPUT_NAME}/pipeline_stage
SWANLAB_PROJECT=MiniQwen-Omni-Mini
LOG_ROOT=../.runtime/train_logs/mini
MINI_MAX_STEPS="${MINI_MAX_STEPS:-0}"
MINI_LOG_INTERVAL="${MINI_LOG_INTERVAL:-100}"
MINI_STOP_AFTER_STAGE="${MINI_STOP_AFTER_STAGE:-2}"

mkdir -p "${OUTPUT_ROOT}/${OUTPUT_NAME}"
mkdir -p "${LOG_ROOT}"

# If pipeline_stage does not exist yet, recover a completed stage from the
# format-v5 checkpoint's trainer_state.
COMPLETED_STAGE=-1
if [[ -f "${PIPELINE_STATE}" ]]; then
  read -r COMPLETED_STAGE < "${PIPELINE_STATE}"
elif [[ -f "${CHECKPOINT}/trainer_state.pt" ]]; then
  COMPLETED_STAGE=$(python -c "import torch; s=torch.load('${CHECKPOINT}/trainer_state.pt', map_location='cpu', weights_only=False); print(s.get('stage_id', -1) if s.get('epoch', 0) >= 1 and s.get('step', 0) == 0 else -1)")
fi
if ! [[ "${COMPLETED_STAGE}" =~ ^(-1|0|1|2)$ ]]; then
  echo "Invalid mini pipeline state: ${COMPLETED_STAGE}" >&2
  exit 1
fi
if ! [[ "${MINI_STOP_AFTER_STAGE}" =~ ^(0|1|2)$ ]]; then
  echo "MINI_STOP_AFTER_STAGE must be 0, 1, or 2" >&2
  exit 1
fi

cleanup_dataset_cache() {
  rm -rf "${MINIQWEN_DATASET_CACHE}"
}
trap cleanup_dataset_cache EXIT

COMMON_ARGS=(
  --model_path "${MODEL_PATH}"
  --audio_encoder_dir ../model/SenseVoiceSmall
  --vision_dir ../model/siglip2-base-p32-256-ve
  --save_dir "${OUTPUT_ROOT}"
  --save_weight "${OUTPUT_NAME}"
  --num_talker_hidden_layers 6
  --talker_hidden_size 768
  --accept_hidden_layer 14
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
  --save_interval 0
  --save_optimizer_state 1
  --from_resume 1
  --num_workers 2
  --max_steps "${MINI_MAX_STEPS}"
  --log_interval "${MINI_LOG_INTERVAL}"
  --use_swanlab
  --swanlab_project "${SWANLAB_PROJECT}"
)

run_stage() {
  local stage_id="$1"
  local port="$2"
  local attempt_log
  shift 2
  if (( stage_id <= COMPLETED_STAGE )); then
    echo "Skip completed mini stage ${stage_id}"
    return
  fi
  if (( stage_id > MINI_STOP_AFTER_STAGE )); then
    return
  fi
  attempt_log="${LOG_ROOT}/stage-${stage_id}-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${attempt_log}"
  echo "Mini stage ${stage_id} logs: ${attempt_log}"
  CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
    --master_port "${port}" \
    --nproc_per_node 4 \
    --log-dir "${attempt_log}/ranks" \
    --tee 3 \
    --local-ranks-filter=0 \
    train_sft_omni.py "${COMMON_ARGS[@]}" --stage_id "${stage_id}" "$@" 2>&1 \
    | sed -u -e '/ALINPU INFO/d' -e '/offline_cache\.hpp.*No cache file exist/d' \
    | tee "${attempt_log}/console.log"
  printf '%s\n' "${stage_id}" > "${PIPELINE_STATE}.tmp"
  mv "${PIPELINE_STATE}.tmp" "${PIPELINE_STATE}"
  COMPLETED_STAGE="${stage_id}"
  cleanup_dataset_cache
}

# Stage 0: T2A warm-up. 24/card leaves safe headroom under FP32 master
# training; the previous 32/card setting reached about 95.1/97.9 GiB.
run_stage 0 29559 \
  --swanlab_run_name 00-mini-t2a \
  --data_path ../dataset/sft_t2a_mini.parquet \
  --epochs 1 --batch_size 24 --accumulation_steps 1 --max_seq_len 512 \
  --qwen_learning_rate 1e-5 --omni_learning_rate 5e-4 \
  --from_weight qwen --mode all

# Stage 1: train only the SenseVoice 512->1024 projector on mini A2A.
run_stage 1 29558 \
  --swanlab_run_name 01-mini-a2a-audio-projector \
  --data_path ../dataset/sft_a2a_mini.parquet \
  --epochs 1 --batch_size 16 --accumulation_steps 1 --max_seq_len 640 \
  --qwen_learning_rate 1e-5 --omni_learning_rate 5e-4 \
  --from_weight "${CHECKPOINT}" --mode audio_proj

# Stage 2: joint A2A consolidation with the confirmed differential-LR policy.
run_stage 2 29557 \
  --swanlab_run_name 02-mini-a2a-joint \
  --data_path ../dataset/sft_a2a_mini.parquet \
  --epochs 1 --batch_size 12 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 2e-6 --omni_learning_rate 2e-5 \
  --from_weight "${CHECKPOINT}" --mode all

echo "Mini audio pipeline complete through stage ${COMPLETED_STAGE}: ${CHECKPOINT}"
