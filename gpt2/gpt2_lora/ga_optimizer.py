from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import nn

from .modeling import autocast_context, classification_loss


@dataclass
class GAConfig:
    population_size: int = 24
    generations: int = 100
    elites: int = 2
    parents: int = 8
    crossover_probability: float = 0.3
    initial_mutation_std: float = 0.01
    final_mutation_std: float = 5e-4
    initial_mutation_ratio: float = 1.0
    final_mutation_ratio: float = 0.05
    schedule: str = "linear"
    tie_breaker: str = "none"
    seed: int = 1

    def validate(self) -> None:
        if self.population_size < 2:
            raise ValueError("population_size must be at least 2.")
        if not (1 <= self.elites < self.population_size):
            raise ValueError("elites must be in [1, population_size - 1].")
        if not (2 <= self.parents <= self.population_size):
            raise ValueError("parents must be in [2, population_size].")
        if self.generations <= 0:
            raise ValueError("generations must be positive.")
        if not (0.0 <= self.crossover_probability <= 1.0):
            raise ValueError("crossover_probability must be in [0, 1].")
        for value in (self.initial_mutation_ratio, self.final_mutation_ratio):
            if not (0.0 <= value <= 1.0):
                raise ValueError("mutation ratios must be in [0, 1].")
        if self.initial_mutation_std < 0 or self.final_mutation_std < 0:
            raise ValueError("mutation standard deviations must be non-negative.")
        if self.tie_breaker not in {"none", "loss"}:
            raise ValueError("tie_breaker must be 'none' or 'loss'.")


@dataclass
class Individual:
    genes: List[torch.Tensor]
    fitness: Optional[float] = None
    loss: Optional[float] = None

    def clone(self) -> "Individual":
        return Individual(
            genes=[gene.clone() for gene in self.genes],
            fitness=self.fitness,
            loss=self.loss,
        )


class GALoRAOptimizer:
    """Forward-only genetic optimization over the model's LoRA parameters."""

    def __init__(
        self,
        model: nn.Module,
        trainable_params: Dict[str, nn.Parameter],
        train_batches: Sequence[dict],
        device: torch.device,
        dtype: torch.dtype,
        config: GAConfig,
        output_dir: Path,
    ) -> None:
        config.validate()
        self.model = model
        self.param_names = list(trainable_params)
        self.params = list(trainable_params.values())
        self.train_batches = train_batches
        self.device = device
        self.dtype = dtype
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = random.Random(config.seed)
        self.cpu_generator = torch.Generator(device="cpu").manual_seed(config.seed)
        self.fitness_evaluations = 0
        self.fitness_examples = 0
        self.forward_batches = 0

    def _base_genes(self) -> List[torch.Tensor]:
        return [param.detach().float().cpu().clone() for param in self.params]

    def _initial_population(self) -> List[Individual]:
        base = self._base_genes()
        population = [Individual([gene.clone() for gene in base])]
        while len(population) < self.config.population_size:
            genes = []
            for gene in base:
                noise = torch.randn(
                    gene.shape, generator=self.cpu_generator, dtype=gene.dtype
                ) * self.config.initial_mutation_std
                genes.append(gene + noise)
            population.append(Individual(genes))
        return population

    @torch.no_grad()
    def _apply(self, individual: Individual) -> None:
        if len(individual.genes) != len(self.params):
            raise ValueError("Chromosome length does not match trainable parameters.")
        for param, gene in zip(self.params, individual.genes):
            param.copy_(gene.to(device=param.device, dtype=param.dtype))

    @torch.inference_mode()
    def _evaluate(self, individual: Individual) -> Tuple[float, float]:
        self._apply(individual)
        self.model.eval()
        correct = 0
        total = 0
        loss_sum = 0.0
        for batch in self.train_batches:
            with autocast_context(self.device, self.dtype):
                logits = self.model(batch["input_ids"], batch["attention_mask"])
            labels = batch["labels"]
            loss = classification_loss(logits, labels)
            correct += int((logits.argmax(dim=-1) == labels).sum().item())
            total += int(labels.numel())
            loss_sum += float(loss.item()) * labels.numel()
        accuracy = correct / max(total, 1)
        average_loss = loss_sum / max(total, 1)
        self.fitness_evaluations += 1
        self.fitness_examples += total
        self.forward_batches += len(self.train_batches)
        individual.fitness = accuracy
        individual.loss = average_loss
        return accuracy, average_loss

    def _schedule(self, start: float, end: float, generation: int) -> float:
        if self.config.generations <= 1:
            return end
        progress = generation / (self.config.generations - 1)
        if self.config.schedule == "linear":
            return start + progress * (end - start)
        if self.config.schedule == "exponential":
            if start == 0:
                return 0.0
            ratio = max(end, 1e-12) / start
            return start * math.pow(ratio, progress)
        raise ValueError(f"Unsupported schedule '{self.config.schedule}'.")

    def _arithmetic_crossover(
        self, parent1: Individual, parent2: Individual
    ) -> Individual:
        alpha = self.rng.random()
        genes = [
            alpha * gene1 + (1.0 - alpha) * gene2
            for gene1, gene2 in zip(parent1.genes, parent2.genes)
        ]
        return Individual(genes)

    def _mutate(
        self, parent: Individual, mutation_std: float, mutation_ratio: float
    ) -> Individual:
        genes: List[torch.Tensor] = []
        for gene in parent.genes:
            mask = torch.rand(
                gene.shape, generator=self.cpu_generator, dtype=torch.float32
            ) < mutation_ratio
            noise = torch.randn(
                gene.shape, generator=self.cpu_generator, dtype=gene.dtype
            ) * mutation_std
            genes.append(gene + mask.to(gene.dtype) * noise)
        return Individual(genes)

    def _sort_key(self, individual: Individual) -> Tuple[float, float]:
        fitness = individual.fitness if individual.fitness is not None else -math.inf
        if self.config.tie_breaker == "loss":
            loss = individual.loss if individual.loss is not None else math.inf
            return fitness, -loss
        # Faithful paper setting: accuracy is the only selection criterion. Python's
        # stable sort preserves the seeded population order for exact ties.
        return fitness, 0.0

    def _reproduce(self, ranked: Sequence[Individual], generation: int) -> List[Individual]:
        mutation_std = self._schedule(
            self.config.initial_mutation_std,
            self.config.final_mutation_std,
            generation,
        )
        mutation_ratio = self._schedule(
            self.config.initial_mutation_ratio,
            self.config.final_mutation_ratio,
            generation,
        )
        next_population = [ranked[i].clone() for i in range(self.config.elites)]
        mating_pool = list(ranked[: self.config.parents])

        while len(next_population) < self.config.population_size:
            if self.rng.random() < self.config.crossover_probability:
                parent1, parent2 = self.rng.sample(mating_pool, 2)
                child = self._arithmetic_crossover(parent1, parent2)
            else:
                parent = self.rng.choice(mating_pool)
                child = self._mutate(parent, mutation_std, mutation_ratio)
            next_population.append(child)
        return next_population

    def _save_best(self, best: Individual, generation: int) -> None:
        payload = {
            "generation": generation,
            "fitness": best.fitness,
            "loss": best.loss,
            "parameter_names": self.param_names,
            "genes": best.genes,
            "config": asdict(self.config),
        }
        torch.save(payload, self.output_dir / "best_chromosome.pt")

    def run(self) -> Tuple[Individual, List[dict]]:
        population = self._initial_population()
        history: List[dict] = []
        global_best: Optional[Individual] = None
        start_time = time.time()

        for generation in range(self.config.generations):
            for individual in population:
                if individual.fitness is None:
                    self._evaluate(individual)

            ranked = sorted(population, key=self._sort_key, reverse=True)
            best = ranked[0]
            if global_best is None or self._sort_key(best) > self._sort_key(global_best):
                global_best = best.clone()
                self._save_best(global_best, generation)

            mutation_std = self._schedule(
                self.config.initial_mutation_std,
                self.config.final_mutation_std,
                generation,
            )
            mutation_ratio = self._schedule(
                self.config.initial_mutation_ratio,
                self.config.final_mutation_ratio,
                generation,
            )
            record = {
                "generation": generation,
                "best_accuracy": best.fitness,
                "best_loss": best.loss,
                "mean_accuracy": sum(ind.fitness or 0.0 for ind in population)
                / len(population),
                "mutation_std": mutation_std,
                "mutation_ratio": mutation_ratio,
                "elapsed_seconds": time.time() - start_time,
            }
            history.append(record)
            print(
                f"[GA] generation={generation:03d} "
                f"best_acc={100.0 * (best.fitness or 0.0):.2f}% "
                f"best_loss={best.loss:.4f} "
                f"mean_acc={100.0 * record['mean_accuracy']:.2f}% "
                f"std={mutation_std:.6g} ratio={mutation_ratio:.4f}"
            )
            with (self.output_dir / "ga_history.json").open("w", encoding="utf-8") as file:
                json.dump(history, file, indent=2)

            if generation + 1 < self.config.generations:
                population = self._reproduce(ranked, generation)

        assert global_best is not None
        self._apply(global_best)
        summary = {
            "best_accuracy": global_best.fitness,
            "best_loss": global_best.loss,
            "fitness_evaluations": self.fitness_evaluations,
            "fitness_examples": self.fitness_examples,
            "forward_batches": self.forward_batches,
            "elapsed_seconds": time.time() - start_time,
            "config": asdict(self.config),
        }
        with (self.output_dir / "ga_summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, indent=2)
        return global_best, history


def load_chromosome(
    path: Path,
    trainable_params: Dict[str, nn.Parameter],
    map_location: str = "cpu",
) -> dict:
    payload = torch.load(path, map_location=map_location)
    expected_names = list(trainable_params)
    saved_names = payload.get("parameter_names")
    if saved_names != expected_names:
        raise ValueError(
            "The saved chromosome does not match this model's trainable parameters."
        )
    genes = payload["genes"]
    with torch.no_grad():
        for param, gene in zip(trainable_params.values(), genes):
            param.copy_(gene.to(device=param.device, dtype=param.dtype))
    return payload
