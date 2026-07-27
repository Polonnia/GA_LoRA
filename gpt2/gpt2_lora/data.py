from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import PreTrainedTokenizerBase


@dataclass(frozen=True)
class DatasetSpec:
    hub_name: str
    config_name: Optional[str]
    split: str
    text_fields: Tuple[str, ...]
    label_field: str = "label"


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "sst2": DatasetSpec("glue", "sst2", "validation", ("sentence",)),
    "imdb": DatasetSpec("imdb", None, "test", ("text",)),
    "yelp_polarity": DatasetSpec("yelp_polarity", None, "test", ("text",)),
    "amazon_polarity": DatasetSpec(
        "amazon_polarity", None, "test", ("title", "content")
    ),
}


def _load_hf_dataset(spec: DatasetSpec, split: Optional[str] = None) -> Dataset:
    kwargs = {"path": spec.hub_name, "split": split or spec.split}
    if spec.config_name is not None:
        kwargs["name"] = spec.config_name
    return load_dataset(**kwargs)


def _join_text(example: dict, fields: Sequence[str]) -> str:
    parts = [str(example.get(field, "")).strip() for field in fields]
    return " ".join(part for part in parts if part)


def balanced_subset(
    dataset: Dataset,
    label_field: str,
    per_class: Optional[int],
    max_samples: Optional[int],
    seed: int,
) -> Dataset:
    """Return a deterministic class-balanced subset for binary classification."""
    if per_class is None and (
        max_samples is None or max_samples < 0 or max_samples >= len(dataset)
    ):
        return dataset

    labels = list(dataset[label_field])
    by_class: Dict[int, List[int]] = {}
    for index, label in enumerate(labels):
        by_class.setdefault(int(label), []).append(index)

    if set(by_class) != {0, 1}:
        raise ValueError(
            f"Expected binary labels {{0, 1}}, but found {sorted(by_class)}."
        )

    generator = torch.Generator().manual_seed(seed)
    selected: List[int] = []
    if per_class is not None:
        for label in (0, 1):
            indices = by_class[label]
            if len(indices) < per_class:
                raise ValueError(
                    f"Class {label} has only {len(indices)} examples, fewer than "
                    f"requested per_class={per_class}."
                )
            order = torch.randperm(len(indices), generator=generator).tolist()
            selected.extend(indices[i] for i in order[:per_class])
    else:
        assert max_samples is not None
        n_per_class = min(len(by_class[0]), len(by_class[1]), max_samples // 2)
        for label in (0, 1):
            indices = by_class[label]
            order = torch.randperm(len(indices), generator=generator).tolist()
            selected.extend(indices[i] for i in order[:n_per_class])

    # Shuffle the combined subset without changing which examples were selected.
    order = torch.randperm(len(selected), generator=generator).tolist()
    return dataset.select([selected[i] for i in order])


class PromptDataset(TorchDataset):
    """Pre-tokenized prompts that preserve the final ``Sentiment:`` suffix."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer: PreTrainedTokenizerBase,
        text_fields: Sequence[str],
        max_length: int,
        prompt_prefix: str = "Review: ",
        prompt_suffix: str = "\nSentiment:",
        label_field: str = "label",
    ) -> None:
        if max_length < 16:
            raise ValueError("max_length must be at least 16.")

        prefix_ids = tokenizer.encode(prompt_prefix, add_special_tokens=False)
        suffix_ids = tokenizer.encode(prompt_suffix, add_special_tokens=False)
        available_text_tokens = max_length - len(prefix_ids) - len(suffix_ids)
        if available_text_tokens <= 0:
            raise ValueError("max_length is too small for the prompt template.")

        self.examples: List[dict] = []
        for example in dataset:
            text = _join_text(example, text_fields)
            text_ids = tokenizer.encode(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=available_text_tokens,
            )
            input_ids = prefix_ids + text_ids + suffix_ids
            self.examples.append(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "label": torch.tensor(int(example[label_field]), dtype=torch.long),
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict:
        return self.examples[index]


class PromptCollator:
    def __init__(self, pad_token_id: int) -> None:
        self.pad_token_id = int(pad_token_id)

    def __call__(self, examples: Iterable[dict]) -> dict:
        examples = list(examples)
        max_len = max(item["input_ids"].numel() for item in examples)
        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, max_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        labels = torch.empty(batch_size, dtype=torch.long)

        for row, item in enumerate(examples):
            length = item["input_ids"].numel()
            input_ids[row, :length] = item["input_ids"]
            attention_mask[row, :length] = 1
            labels[row] = item["label"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def build_train_loader(
    tokenizer: PreTrainedTokenizerBase,
    shots_per_class: int,
    max_length: int,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    spec = DatasetSpec("glue", "sst2", "train", ("sentence",))
    raw = _load_hf_dataset(spec)
    raw = balanced_subset(
        raw,
        label_field=spec.label_field,
        per_class=shots_per_class,
        max_samples=None,
        seed=seed,
    )
    encoded = PromptDataset(
        raw,
        tokenizer=tokenizer,
        text_fields=spec.text_fields,
        max_length=max_length,
        label_field=spec.label_field,
    )
    return DataLoader(
        encoded,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=PromptCollator(tokenizer.pad_token_id),
    )


def build_eval_loader(
    dataset_name: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    batch_size: int,
    max_samples: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Choices: {sorted(DATASET_SPECS)}"
        )
    spec = DATASET_SPECS[dataset_name]
    raw = _load_hf_dataset(spec)
    raw = balanced_subset(
        raw,
        label_field=spec.label_field,
        per_class=None,
        max_samples=max_samples,
        seed=seed,
    )
    encoded = PromptDataset(
        raw,
        tokenizer=tokenizer,
        text_fields=spec.text_fields,
        max_length=max_length,
        label_field=spec.label_field,
    )
    return DataLoader(
        encoded,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=PromptCollator(tokenizer.pad_token_id),
    )
