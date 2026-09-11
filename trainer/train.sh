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
export MINIQWEN_DATASET_CACHE=../dataset/.miniqwen_training_cache

# MiniQwen-Omni Full dataset pipeline. Defaults to one 16x PPU-ZW810E node.
# Run from trainer/: source ../envs/Omni-ppu/bin/activate && bash train.sh
#
# Storage policy:
#   - save after every epoch and atomically overwrite one stable checkpoint.
#   - optimizer/scaler/RNG are included so `bash train.sh` resumes automatically.
#   - completed stages are recorded in pipeline_stage and skipped after restart.
# Monitoring:
#   - training metrics are recorded with SwanLab.

MODEL_PATH=../model/Qwen3-0.6B
# Format-v5 uses the same-frame Main+Code-Predictor architecture and starts
# from Qwen rather than reusing delayed-head checkpoints.
OUTPUT_NAME=miniqwen_omni_full_main_codec_cp_v5
OUTPUT_ROOT=../out
CHECKPOINT=${OUTPUT_ROOT}/${OUTPUT_NAME}/checkpoint
SWANLAB_PROJECT=MiniQwen-Omni-Full
PIPELINE_STATE=${OUTPUT_ROOT}/${OUTPUT_NAME}/pipeline_stage
LOG_ROOT=../.runtime/train_logs/full

# Keep the proven per-device batches when scaling from 4 to 16 PPUs so every
# device remains well utilized. Override these two variables for another host.
NPROC_PER_NODE=${NPROC_PER_NODE:-16}
PPU_DEVICES=${PPU_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}
NUM_WORKERS=${NUM_WORKERS:-4}
if ! [[ "${NPROC_PER_NODE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "NPROC_PER_NODE must be a positive integer: ${NPROC_PER_NODE}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/${OUTPUT_NAME}"
mkdir -p "${LOG_ROOT}"
COMPLETED_STAGE=0
if [[ -f "${PIPELINE_STATE}" ]]; then
  read -r COMPLETED_STAGE < "${PIPELINE_STATE}"
fi
if ! [[ "${COMPLETED_STAGE}" =~ ^[0-7]$ ]]; then
  echo "Invalid pipeline state: ${PIPELINE_STATE}" >&2
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
  --use_moe 0
  --use_compile 0
  --gradient_checkpointing 0
  --dynamic_padding 1
  --save_interval 0
  --save_at_epoch_end 1
  --save_optimizer_state 1
  --from_resume 1
  --num_workers "${NUM_WORKERS}"
  --use_swanlab
  --swanlab_project "${SWANLAB_PROJECT}"
)

run_stage() {
  local stage_id="$1"
  local port="$2"
  local attempt_log
  shift 2
  if (( stage_id <= COMPLETED_STAGE )); then
    echo "Skip completed stage ${stage_id}"
    return
  fi
  attempt_log="${LOG_ROOT}/stage-${stage_id}-$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "${attempt_log}"
  echo "Stage ${stage_id} logs: ${attempt_log}"
  CUDA_VISIBLE_DEVICES="${PPU_DEVICES}" torchrun \
    --master_port "${port}" \
    --nproc_per_node "${NPROC_PER_NODE}" \
    --log-dir "${attempt_log}/ranks" \
    --tee 3 \
    --local-ranks-filter=0 \
    train_sft_omni.py "${COMMON_ARGS[@]}" --stage_id "${stage_id}" "$@" 2>&1 \
    | sed -u -e '/ALINPU INFO/d' -e '/offline_cache\.hpp.*No cache file exist/d' \
    | tee "${attempt_log}/console.log"
  printf '%s\n' "${stage_id}" > "${PIPELINE_STATE}.tmp"
  mv "${PIPELINE_STATE}.tmp" "${PIPELINE_STATE}"
  COMPLETED_STAGE="${stage_id}"
  # HF Arrow cache can be tens of GB; it is no longer needed after this stage.
  cleanup_dataset_cache
}

# Stage 1: T2A warm-up. Initialize Qwen3 and train the new Talker/bridge jointly.
run_stage 1 29560 \
  --swanlab_run_name 01-t2a-warmup \
  --data_path ../dataset/sft_t2a.parquet \
  --epochs 6 --batch_size 24 --accumulation_steps 1 --max_seq_len 512 \
  --qwen_learning_rate 1e-5 --omni_learning_rate 5e-4 \
  --from_weight qwen --mode all

# Stage 2: align the SenseVoice audio projector while preserving the trained model.
run_stage 2 29561 \
  --swanlab_run_name 02-a2a-audio-projector \
  --data_path ../dataset/sft_a2a.parquet \
  --epochs 1 --batch_size 12 --accumulation_steps 2 --max_seq_len 1024 \
  --qwen_learning_rate 1e-5 --omni_learning_rate 5e-4 \
  --from_weight "${CHECKPOINT}" --mode audio_proj

# Stage 3: A2A joint fine-tuning with a lower Qwen LR.
run_stage 3 29562 \
  --swanlab_run_name 03-a2a-joint \
  --data_path ../dataset/sft_a2a.parquet \
  --epochs 3 --batch_size 12 --accumulation_steps 2 --max_seq_len 1024 \
  --qwen_learning_rate 5e-6 --omni_learning_rate 5e-5 \
  --from_weight "${CHECKPOINT}" --mode all

# Stage 4: align the SigLIP2 vision projector.
run_stage 4 29563 \
  --swanlab_run_name 04-i2t-vision-projector \
  --data_path ../dataset/sft_i2t.parquet \
  --epochs 1 --batch_size 16 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 5e-6 --omni_learning_rate 5e-5 \
  --from_weight "${CHECKPOINT}" --mode vision_proj

# Stage 5: low-LR I2T joint fine-tuning.
run_stage 5 29564 \
  --swanlab_run_name 05-i2t-joint \
  --data_path ../dataset/sft_i2t.parquet \
  --epochs 1 --batch_size 16 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 1e-6 --omni_learning_rate 5e-6 \
  --from_weight "${CHECKPOINT}" --mode all --train_modules thinker,text_head,vision_proj

# Stage 6: final A2A low-LR consolidation.
run_stage 6 29565 \
  --swanlab_run_name 06-a2a-final \
  --data_path ../dataset/sft_a2a.parquet \
  --epochs 1 --batch_size 12 --accumulation_steps 2 --max_seq_len 1024 \
  --qwen_learning_rate 1e-6 --omni_learning_rate 5e-6 \
  --from_weight "${CHECKPOINT}" --mode all

# Stage 7: final vision-projector calibration. This leaves the final checkpoint in CHECKPOINT.
run_stage 7 29566 \
  --swanlab_run_name 07-i2t-vision-final \
  --data_path ../dataset/sft_i2t.parquet \
  --epochs 1 --batch_size 16 --accumulation_steps 1 --max_seq_len 768 \
  --qwen_learning_rate 1e-6 --omni_learning_rate 5e-6 \
  --from_weight "${CHECKPOINT}" --mode vision_proj
echo "Training pipeline complete: ${CHECKPOINT}"
