from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.optim import AdamW
from tqdm import tqdm

from .data import build_eval_loader, build_train_loader
from .ga_optimizer import GAConfig, GALoRAOptimizer, load_chromosome
from .modeling import (
    autocast_context,
    build_model_and_tokenizer,
    classification_loss,
    materialize_batches,
    move_batch,
    trainable_parameter_dict,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="GPT-2 LoRA training with GA/Adam and ID/OOD sentiment evaluation."
    )
    parser.add_argument("--mode", choices=("train", "eval", "train_eval"), default="train_eval")
    parser.add_argument(
        "--optimizer",
        choices=("ga", "adam", "original", "zero_shot"),
        default="ga",
        help="'original' evaluates the untouched pretrained model; 'zero_shot' is a legacy alias.",
    )
    parser.add_argument("--model_name", default="openai-community/gpt2")
    parser.add_argument("--output_dir", default="outputs/gpt2_sst2/ga/seed1")
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--chromosome_path", default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")

    parser.add_argument("--shots_per_class", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    parser.add_argument(
        "--ood_datasets",
        default="imdb,yelp_polarity,amazon_polarity",
        help="Comma-separated OOD datasets.",
    )

    parser.add_argument("--lora_rank", type=int, default=2)
    parser.add_argument("--lora_alpha", type=int, default=4)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--lora_layers", type=int, default=2)
    parser.add_argument(
        "--target_modules",
        default="c_attn",
        help="Comma-separated GPT-2 module suffixes. c_attn jointly adapts Q/K/V.",
    )

    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_epochs", type=int, default=20)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    parser.add_argument("--population_size", type=int, default=24)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--elites", type=int, default=2)
    parser.add_argument("--parents", type=int, default=8)
    parser.add_argument("--crossover_probability", type=float, default=0.3)
    parser.add_argument("--initial_mutation_std", type=float, default=0.01)
    parser.add_argument("--final_mutation_std", type=float, default=5e-4)
    parser.add_argument("--initial_mutation_ratio", type=float, default=1.0)
    parser.add_argument("--final_mutation_ratio", type=float, default=0.05)
    parser.add_argument("--mutation_schedule", choices=("linear", "exponential"), default="linear")
    parser.add_argument(
        "--ga_tie_breaker",
        choices=("none", "loss"),
        default="none",
        help="Use 'none' for the paper-faithful accuracy-only GA objective.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.inference_mode()
def evaluate(model, loader: Iterable[dict], device: torch.device, dtype: torch.dtype) -> Dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        batch = move_batch(batch, device)
        with autocast_context(device, dtype):
            logits = model(batch["input_ids"], batch["attention_mask"])
        loss = classification_loss(logits, batch["labels"])
        correct += int((logits.argmax(dim=-1) == batch["labels"]).sum().item())
        total += int(batch["labels"].numel())
        loss_sum += float(loss.item()) * batch["labels"].numel()
    return {
        "accuracy": correct / max(total, 1),
        "loss": loss_sum / max(total, 1),
        "num_examples": total,
    }


def train_adam(
    model,
    train_loader,
    device: torch.device,
    dtype: torch.dtype,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    max_grad_norm: float,
    output_dir: Path,
) -> List[dict]:
    params = list(trainable_parameter_dict(model).values())
    optimizer = AdamW(params, lr=learning_rate, weight_decay=weight_decay)
    history: List[dict] = []
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        correct = 0
        total = 0
        loss_sum = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, dtype):
                logits = model(batch["input_ids"], batch["attention_mask"])
                loss = classification_loss(logits, batch["labels"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_grad_norm)
            optimizer.step()

            correct += int((logits.argmax(dim=-1) == batch["labels"]).sum().item())
            total += int(batch["labels"].numel())
            loss_sum += float(loss.detach().item()) * batch["labels"].numel()

        record = {
            "epoch": epoch,
            "train_accuracy": correct / max(total, 1),
            "train_loss": loss_sum / max(total, 1),
            "elapsed_seconds": time.time() - start_time,
        }
        history.append(record)
        print(
            f"[Adam] epoch={epoch:03d} "
            f"train_acc={100.0 * record['train_accuracy']:.2f}% "
            f"train_loss={record['train_loss']:.4f}"
        )
        with (output_dir / "adam_history.json").open("w", encoding="utf-8") as file:
            json.dump(history, file, indent=2)
    return history


def save_adapter(model, tokenizer, output_dir: Path) -> None:
    adapter_dir = output_dir / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.causal_lm.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)


def load_adapter_if_requested(model, adapter_path: str | None) -> None:
    if not adapter_path:
        return
    # Loading into an already constructed PEFT model keeps architecture checks explicit.
    from peft import set_peft_model_state_dict
    from peft.utils.save_and_load import load_peft_weights

    weights = load_peft_weights(adapter_path, device=str(next(model.parameters()).device))
    result = set_peft_model_state_dict(model.causal_lm, weights, adapter_name="default")
    if getattr(result, "unexpected_keys", None):
        raise RuntimeError(f"Unexpected adapter keys: {result.unexpected_keys}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    run_start = time.time()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "args.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)

    device = torch.device(args.device)
    is_original = args.optimizer in ("original", "zero_shot")
    if is_original and args.mode != "eval":
        raise ValueError("--optimizer original/zero_shot only supports --mode eval.")
    if is_original and (args.adapter_path or args.chromosome_path):
        raise ValueError("Original-model evaluation cannot load an adapter or chromosome.")
    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    model, tokenizer, dtype = build_model_and_tokenizer(
        model_name=args.model_name,
        device=device,
        dtype_name=args.dtype,
        use_lora=not is_original,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_layers=args.lora_layers,
        target_modules=target_modules,
    )
    trainable = {}
    if is_original:
        print("Evaluating the original pretrained model without LoRA adapters.")
    else:
        trainable = trainable_parameter_dict(model)
        print(f"Trainable LoRA tensors: {len(trainable)}")
        print(f"Trainable LoRA parameters: {sum(p.numel() for p in trainable.values()):,}")

    if args.adapter_path:
        load_adapter_if_requested(model, args.adapter_path)
    if args.chromosome_path:
        load_chromosome(Path(args.chromosome_path), trainable)

    train_loader = None
    training_seconds = 0.0
    evaluation_seconds = 0.0
    if args.mode in ("train", "train_eval") and not is_original:
        training_start = time.time()
        train_loader = build_train_loader(
            tokenizer=tokenizer,
            shots_per_class=args.shots_per_class,
            max_length=args.max_length,
            batch_size=args.train_batch_size,
            seed=args.seed,
            num_workers=args.num_workers,
        )

        if args.optimizer == "ga":
            # The few-shot fitness set is intentionally cached on the GPU because every
            # individual is evaluated on the same examples.
            train_batches = materialize_batches(train_loader, device)
            ga_config = GAConfig(
                population_size=args.population_size,
                generations=args.generations,
                elites=args.elites,
                parents=args.parents,
                crossover_probability=args.crossover_probability,
                initial_mutation_std=args.initial_mutation_std,
                final_mutation_std=args.final_mutation_std,
                initial_mutation_ratio=args.initial_mutation_ratio,
                final_mutation_ratio=args.final_mutation_ratio,
                schedule=args.mutation_schedule,
                tie_breaker=args.ga_tie_breaker,
                seed=args.seed,
            )
            optimizer = GALoRAOptimizer(
                model=model,
                trainable_params=trainable,
                train_batches=train_batches,
                device=device,
                dtype=dtype,
                config=ga_config,
                output_dir=output_dir,
            )
            optimizer.run()
        elif args.optimizer == "adam":
            train_adam(
                model=model,
                train_loader=train_loader,
                device=device,
                dtype=dtype,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                epochs=args.adam_epochs,
                max_grad_norm=args.max_grad_norm,
                output_dir=output_dir,
            )
        save_adapter(model, tokenizer, output_dir)
        training_seconds = time.time() - training_start

    if args.mode in ("eval", "train_eval"):
        evaluation_start = time.time()
        dataset_names = ["sst2"] + [
            item.strip() for item in args.ood_datasets.split(",") if item.strip()
        ]
        results: Dict[str, dict] = {}
        for dataset_name in dataset_names:
            loader = build_eval_loader(
                dataset_name=dataset_name,
                tokenizer=tokenizer,
                max_length=args.max_length,
                batch_size=args.eval_batch_size,
                max_samples=args.max_eval_samples,
                seed=args.seed,
                num_workers=args.num_workers,
            )
            metrics = evaluate(model, loader, device, dtype)
            results[dataset_name] = metrics
            print(
                f"[{dataset_name}] accuracy={100.0 * metrics['accuracy']:.2f}% "
                f"loss={metrics['loss']:.4f} n={metrics['num_examples']}"
            )

        ood_names = dataset_names[1:]
        ood_average = sum(results[name]["accuracy"] for name in ood_names) / max(
            len(ood_names), 1
        )
        results["summary"] = {
            "id_dataset": "sst2",
            "id_accuracy": results["sst2"]["accuracy"],
            "ood_datasets": ood_names,
            "ood_average_accuracy": ood_average,
            "id_ood_gap": results["sst2"]["accuracy"] - ood_average,
        }
        with (output_dir / "evaluation.json").open("w", encoding="utf-8") as file:
            json.dump(results, file, indent=2)
        print(json.dumps(results["summary"], indent=2))
        evaluation_seconds = time.time() - evaluation_start

    timing = {
        "optimizer": args.optimizer,
        "training_seconds": training_seconds,
        "evaluation_seconds": evaluation_seconds,
        "total_seconds": time.time() - run_start,
    }
    with (output_dir / "timing.json").open("w", encoding="utf-8") as file:
        json.dump(timing, file, indent=2)


if __name__ == "__main__":
    main()
