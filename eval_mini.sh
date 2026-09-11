#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "$0")" && pwd)
CHECKPOINT=${MINI_CHECKPOINT:-${PROJECT_ROOT}/out/miniqwen_omni_mini_main_codec_cp_v5/checkpoint}
OUTPUT_DIR=${MINI_EVAL_OUTPUT:-${PROJECT_ROOT}/output_audio/mini}
EVAL_MODE=${MINI_EVAL_MODE:-0}
MAX_SAMPLES=${MINI_EVAL_MAX_SAMPLES:-1}
MAX_NEW_TOKENS=${MINI_EVAL_MAX_NEW_TOKENS:-256}
PROMPT_LANG=${MINI_EVAL_PROMPT_LANG:-0}

cd "${PROJECT_ROOT}"
python eval_omni.py \
  --load_from "${CHECKPOINT}" \
  --output_dir "${OUTPUT_DIR}" \
  --mode "${EVAL_MODE}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  --prompt_lang "${PROMPT_LANG}" \
  --open_thinking 0 \
  --decode_audio 1
