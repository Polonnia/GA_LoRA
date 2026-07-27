from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate GPT-2 ID/OOD results across seeds.")
    parser.add_argument(
        "paths",
        nargs="+",
        help="Run directories or evaluation.json files for one optimizer.",
    )
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def metric_stats(values: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def main() -> None:
    args = parse_args()
    evaluations = []
    timings = []
    for raw_path in args.paths:
        path = Path(raw_path)
        eval_path = path if path.name == "evaluation.json" else path / "evaluation.json"
        with eval_path.open("r", encoding="utf-8") as file:
            evaluations.append(json.load(file))
        timing_path = eval_path.parent / "timing.json"
        if timing_path.exists():
            with timing_path.open("r", encoding="utf-8") as file:
                timings.append(json.load(file))

    dataset_names = [name for name in evaluations[0] if name != "summary"]
    output = {
        "datasets": {
            name: metric_stats([run[name]["accuracy"] for run in evaluations])
            for name in dataset_names
        },
        "id_accuracy": metric_stats(
            [run["summary"]["id_accuracy"] for run in evaluations]
        ),
        "ood_average_accuracy": metric_stats(
            [run["summary"]["ood_average_accuracy"] for run in evaluations]
        ),
        "id_ood_gap": metric_stats(
            [run["summary"]["id_ood_gap"] for run in evaluations]
        ),
    }
    if timings:
        output["training_seconds"] = metric_stats(
            [run["training_seconds"] for run in timings]
        )

    rendered = json.dumps(output, indent=2)
    print(rendered)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
