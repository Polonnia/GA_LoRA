# GPT-2 Small/Large ID/OOD experiment for GA-LoRA

This add-on adapts **GPT-2 Small** or **GPT-2 Large (774M)** on few-shot SST-2 sentiment
classification and evaluates:

- **ID:** SST-2 validation
- **OOD:** IMDb, Yelp Polarity, and Amazon Polarity

The prediction is made with a fixed causal-LM prompt:

```text
Review: <text>
Sentiment:
```

The two next-token verbalizers are ` negative` and ` positive`. No randomly initialized classifier
head is introduced. LoRA is applied by default to `c_attn` in the final two GPT-2 blocks. In GPT-2,
`c_attn` is the combined Q/K/V projection.

## 1. Copy into the repository

Copy the `gpt2_lora/` directory and `requirements_gpt2.txt` into the root of the `GA_LoRA`
repository (the code was prepared against the current `seedchain` branch).

## 2. Install dependencies

```bash
pip install -r requirements_gpt2.txt
```

The datasets and GPT-2 checkpoint are downloaded from Hugging Face on first use.

The repository also contains a local package named `datasets`. The GPT-2 data loader explicitly
imports the Hugging Face package, so running from the repository root or having the root on
`PYTHONPATH` will not shadow the dependency.

## 3. Original pretrained model evaluation

This path loads the pretrained causal LM directly and does not create or load a LoRA adapter:

```bash
python -m gpt2_lora.train_eval \
  --optimizer original \
  --mode eval \
  --model_name openai-community/gpt2-large \
  --device cuda:0 \
  --eval_batch_size 4 \
  --output_dir outputs/gpt2_large_sst2/original/seed1
```

`zero_shot` remains available as a backwards-compatible alias for `original`.

## 4. Quick smoke test

```bash
python -m gpt2_lora.train_eval \
  --optimizer ga \
  --mode train_eval \
  --device cuda:0 \
  --shots_per_class 16 \
  --population_size 4 \
  --generations 2 \
  --max_eval_samples 100 \
  --output_dir outputs/gpt2_smoke
```

## 5. Main GA-LoRA experiment

```bash
python -m gpt2_lora.train_eval \
  --optimizer ga \
  --mode train_eval \
  --device cuda:0 \
  --seed 1 \
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
  --output_dir outputs/gpt2_sst2/ga/seed1
```

## 6. Adam baseline

```bash
python -m gpt2_lora.train_eval \
  --optimizer adam \
  --mode train_eval \
  --device cuda:0 \
  --seed 1 \
  --shots_per_class 16 \
  --lora_rank 2 \
  --lora_layers 2 \
  --learning_rate 5e-4 \
  --adam_epochs 20 \
  --output_dir outputs/gpt2_sst2/adam/seed1
```

Run `bash gpt2_lora/run_gpt2.sh` for original-model evaluation, Adam, and GA over three
seeds using GPT-2 Small.

For GPT-2 Large, run:

```bash
bash gpt2_lora/run_gpt2_large.sh
```

The Large launcher selects `openai-community/gpt2-large` and uses conservative defaults of
`eval_batch_size=4` and `train_batch_size=2`. Override them according to available GPU memory:

```bash
GPU=0 SEEDS="1 2 3" EVAL_BATCH_SIZE=8 TRAIN_BATCH_SIZE=4 \
  bash gpt2_lora/run_gpt2_large.sh
```

## 7. Outputs

Each run writes:

- `args.json`: complete configuration
- `adapter/`: PEFT LoRA adapter and tokenizer
- `evaluation.json`: ID, each OOD result, OOD average, and ID–OOD gap
- `ga_history.json` or `adam_history.json`: training curve
- `ga_summary.json`: GA fitness-evaluation and forward-batch counts
- `timing.json`: training/evaluation wall-clock time
- `best_chromosome.pt`: GA genes and parameter-name checks

## Important reporting notes

1. Report the exact checkpoint (**GPT-2 Small** or **GPT-2 Large, 774M**); GPT-2 Large is
   larger than GPT-2 Small but is not a modern instruction-tuned LLM.
2. Keep the same sampled SST-2 examples for all optimizers by using the same seed.
3. Report three seeds and include wall-clock time or forward-pass count.
4. `max_eval_samples=2000` is the default for fast rebuttal experiments. Use `-1` for complete
   test sets if time permits.
5. The default GA fitness and selection criterion is training accuracy only. The optional
   `--ga_tie_breaker loss` setting can reduce plateaus on very small binary datasets, but it changes
   selection among equal-accuracy individuals and must be disclosed if used.

Aggregate three seeds with:

```bash
python -m gpt2_lora.aggregate_results \
  outputs/gpt2_sst2/ga/seed1 \
  outputs/gpt2_sst2/ga/seed2 \
  outputs/gpt2_sst2/ga/seed3 \
  --output outputs/gpt2_sst2/ga/summary.json
```
