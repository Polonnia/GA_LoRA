from __future__ import annotations

from contextlib import nullcontext
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase


def resolve_dtype(name: str, device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if name == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if name not in mapping:
        raise ValueError(f"Unknown dtype '{name}'.")
    return mapping[name]


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype in (torch.float16, torch.bfloat16):
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def verbalizer_token_ids(
    tokenizer: PreTrainedTokenizerBase,
    negative_word: str = " negative",
    positive_word: str = " positive",
) -> Tuple[int, int]:
    token_ids: List[int] = []
    for word in (negative_word, positive_word):
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(
                f"Verbalizer {word!r} maps to {len(ids)} tokens ({ids}), but this "
                "experiment requires one token per class."
            )
        token_ids.append(ids[0])
    return token_ids[0], token_ids[1]


class GPT2VerbalizerClassifier(nn.Module):
    """Classify sentiment using next-token probabilities from GPT-2."""

    def __init__(self, causal_lm: nn.Module, label_token_ids: Sequence[int]) -> None:
        super().__init__()
        if len(label_token_ids) != 2:
            raise ValueError("Exactly two verbalizer token IDs are required.")
        self.causal_lm = causal_lm
        self.register_buffer(
            "label_token_ids",
            torch.tensor(list(label_token_ids), dtype=torch.long),
            persistent=True,
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = self.causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # Prompts are right padded, so gather logits at the last non-padding token.
        last_positions = attention_mask.sum(dim=1) - 1
        rows = torch.arange(input_ids.size(0), device=input_ids.device)
        next_token_logits = outputs.logits[rows, last_positions]
        return next_token_logits.index_select(-1, self.label_token_ids)


def build_model_and_tokenizer(
    model_name: str,
    device: torch.device,
    dtype_name: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_layers: int,
    target_modules: Sequence[str],
) -> Tuple[GPT2VerbalizerClassifier, PreTrainedTokenizerBase, torch.dtype]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = resolve_dtype(dtype_name, device)
    base_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.config.use_cache = False

    num_hidden_layers = int(base_model.config.n_layer)
    if lora_layers <= 0 or lora_layers > num_hidden_layers:
        raise ValueError(
            f"lora_layers must be in [1, {num_hidden_layers}], got {lora_layers}."
        )
    layers_to_transform = list(
        range(num_hidden_layers - lora_layers, num_hidden_layers)
    )

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=list(target_modules),
        layers_to_transform=layers_to_transform,
        layers_pattern="h",
        fan_in_fan_out=True,
        bias="none",
    )
    peft_model = get_peft_model(base_model, peft_config)
    label_ids = verbalizer_token_ids(tokenizer)
    model = GPT2VerbalizerClassifier(peft_model, label_ids).to(device)
    return model, tokenizer, dtype


def trainable_parameter_dict(model: nn.Module) -> Dict[str, nn.Parameter]:
    params = {name: param for name, param in model.named_parameters() if param.requires_grad}
    if not params:
        raise RuntimeError("No trainable LoRA parameters were found.")
    unexpected = [name for name in params if "lora_" not in name]
    if unexpected:
        raise RuntimeError(
            "Non-LoRA trainable parameters found: " + ", ".join(unexpected[:10])
        )
    return params


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float(), labels)


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def materialize_batches(loader: Iterable[dict], device: torch.device) -> List[dict]:
    return [move_batch(batch, device) for batch in loader]
