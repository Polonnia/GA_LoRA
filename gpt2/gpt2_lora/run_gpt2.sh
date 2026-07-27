#!/usr/bin/env bash
set -euo pipefail

# Run from the GA_LoRA repository root.
# The default experiment uses 16 SST-2 examples per class for training,
# SST-2 validation as ID, and IMDb/Yelp/Amazon as OOD datasets.

GPU="${GPU:-0}"
SEEDS="${SEEDS:-1 2 3}"
MODEL_NAME="${MODEL_NAME:-openai-community/gpt2}"
MODEL_TAG="${MODEL_TAG:-gpt2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/${MODEL_TAG}_sst2}"

for SEED in ${SEEDS}; do
  python -m gpt2_lora.train_eval \
    --optimizer original \
    --mode eval \
    --model_name "${MODEL_NAME}" \
    --device "cuda:${GPU}" \
    --seed "${SEED}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --output_dir "${OUTPUT_ROOT}/original/seed${SEED}"

  python -m gpt2_lora.train_eval \
    --optimizer adam \
    --mode train_eval \
    --model_name "${MODEL_NAME}" \
    --device "cuda:${GPU}" \
    --seed "${SEED}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --shots_per_class 16 \
    --lora_rank 2 \
    --lora_layers 2 \
    --learning_rate 5e-4 \
    --adam_epochs 20 \
    --output_dir "${OUTPUT_ROOT}/adam/seed${SEED}"

  python -m gpt2_lora.train_eval \
    --optimizer ga \
    --mode train_eval \
    --model_name "${MODEL_NAME}" \
    --device "cuda:${GPU}" \
    --seed "${SEED}" \
    --train_batch_size "${TRAIN_BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --shots_per_class 16 \
    --lora_rank 2 \
    --lora_layers 2 \
    --population_size 24 \
    --generations 100 \
    --elites 2 \
    --parents 8 \
    --crossover_probability 0.3 \
    --initial_mutation_std 0.01 \
    --final_mutation_std 0.0005 \
    --initial_mutation_ratio 1.0 \
    --final_mutation_ratio 0.05 \
    --output_dir "${OUTPUT_ROOT}/ga/seed${SEED}"
done
