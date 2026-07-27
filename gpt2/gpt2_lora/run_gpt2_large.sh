#!/usr/bin/env bash
set -euo pipefail

# GPT-2 Large (774M) needs substantially more GPU memory than GPT-2 small.
# Every value can still be overridden from the environment.
MODEL_NAME="${MODEL_NAME:-openai-community/gpt2-large}" \
MODEL_TAG="${MODEL_TAG:-gpt2_large}" \
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-2}" \
bash "$(dirname "$0")/run_gpt2.sh"
