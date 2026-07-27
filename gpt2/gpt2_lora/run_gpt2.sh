#!/usr/bin/env bash
set -euo pipefail

# Run from the GA_LoRA repository root.
# The default experiment uses 16 SST-2 examples per class for training,
# SST-2 validation as ID, and IMDb/Yelp/Amazon as OOD datasets.

GPU="${GPU:-0}"
SEEDS="${SEEDS:-1 2 3}"

for SEED in ${SEEDS}; do
  python -m gpt2_lora.train_eval \
    --optimizer zero_shot \
    --mode eval \
    --device "cuda:${GPU}" \
    --seed "${SEED}" \
    --output_dir "outputs/gpt2_sst2/zero_shot/seed${SEED}"

  python -m gpt2_lora.train_eval \
    --optimizer adam \
    --mode train_eval \
    --device "cuda:${GPU}" \
    --seed "${SEED}" \
    --shots_per_class 16 \
    --lora_rank 2 \
    --lora_layers 2 \
    --learning_rate 5e-4 \
    --adam_epochs 20 \
    --output_dir "outputs/gpt2_sst2/adam/seed${SEED}"

  python -m gpt2_lora.train_eval \
    --optimizer ga \
    --mode train_eval \
    --device "cuda:${GPU}" \
    --seed "${SEED}" \
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
    --output_dir "outputs/gpt2_sst2/ga/seed${SEED}"
done
