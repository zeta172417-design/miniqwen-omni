#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

source envs/Omni-ppu/bin/activate

: "${MINIQWEN_WEB_PASSWORD:?Set MINIQWEN_WEB_PASSWORD (at least 8 characters)}"
if (( ${#MINIQWEN_WEB_PASSWORD} < 8 )); then
  echo "MINIQWEN_WEB_PASSWORD must contain at least 8 characters" >&2
  exit 2
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export HF_HUB_DISABLE_TELEMETRY=1
export GRADIO_ANALYTICS_ENABLED=False
export NCCL_DEBUG=ERROR
export PCCL_DEBUG=ERROR
export TORCH_CPP_LOG_LEVEL=ERROR
export NO_PROXY="127.0.0.1,localhost,0.0.0.0${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.runtime/hf_cache}"
export GRADIO_TEMP_DIR="${GRADIO_TEMP_DIR:-${PROJECT_ROOT}/.runtime/gradio}"
mkdir -p "${HF_HOME}" "${GRADIO_TEMP_DIR}"

# Optional outbound proxy for creating a gradio.live tunnel. Local model and
# browser traffic do not use it. Example: http://127.0.0.1:7890
if [[ -n "${MINIQWEN_WEB_PROXY:-}" ]]; then
  export HTTP_PROXY="${MINIQWEN_WEB_PROXY}"
  export HTTPS_PROXY="${MINIQWEN_WEB_PROXY}"
  export http_proxy="${MINIQWEN_WEB_PROXY}"
  export https_proxy="${MINIQWEN_WEB_PROXY}"
fi

web_args=(
  --model-path "${MINIQWEN_MODEL_PATH:-${PROJECT_ROOT}/out/miniqwen_omni_full/checkpoint}"
  --audio-encoder "${MINIQWEN_AUDIO_ENCODER:-${PROJECT_ROOT}/model/SenseVoiceSmall}"
  --vision-model "${MINIQWEN_VISION_MODEL:-${PROJECT_ROOT}/model/siglip2-base-p32-256-ve}"
  --mimi-path "${MINIQWEN_MIMI_PATH:-${PROJECT_ROOT}/model/mimi}"
  --device "${MINIQWEN_WEB_DEVICE:-cuda}"
  --asr-device "${MINIQWEN_WEB_ASR_DEVICE:-cpu}"
  --dtype "${MINIQWEN_WEB_DTYPE:-bfloat16}"
  --host "${MINIQWEN_WEB_HOST:-0.0.0.0}"
  --port "${MINIQWEN_WEB_PORT:-7860}"
  --max-audio-seconds "${MINIQWEN_WEB_MAX_AUDIO_SECONDS:-30}"
  --queue-size "${MINIQWEN_WEB_QUEUE_SIZE:-8}"
)

if [[ "${MINIQWEN_WEB_SHARE:-0}" == "1" ]]; then
  web_args+=(--share)
fi
if [[ "${MINIQWEN_WEB_DISABLE_ASR:-0}" == "1" ]]; then
  web_args+=(--disable-asr)
fi
if [[ "${MINIQWEN_WEB_OPEN_THINKING:-0}" == "1" ]]; then
  web_args+=(--open-thinking)
fi
if [[ -n "${MINIQWEN_WEB_ROOT_PATH:-}" ]]; then
  web_args+=(--root-path "${MINIQWEN_WEB_ROOT_PATH}")
fi

exec python -u scripts/web_demo_omni.py "${web_args[@]}" "$@" \
  > >(sed -u \
      -e '/ALINPU INFO/d' \
      -e '/ACOMPUTE: \[device caps\]/d' \
      -e '/offline_cache\.hpp.*No cache file exist/d' \
      -e '/offline_cache\.hpp.*Saved_file/d') 2>&1
