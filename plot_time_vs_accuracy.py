#!/usr/bin/env python3
"""
Plot training-time vs accuracy trade-offs for multiple optimizers.

- x-axis: training time (minutes)
- left subplot y-axis: ID accuracy
- right subplot y-axis: OOD average accuracy
- point colors indicate different methods/optimizers
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def get_data() -> list[dict[str, float | str]]:
    return [
        {"method": "SGD", "id": 69.6, "ood": 59.6, "time": 32},
        {"method": "Adam", "id": 70.5, "ood": 59.7, "time": 34},
        {"method": "E-SGD", "id": 70.4, "ood": 60.1, "time": 539},
        {"method": "SAM", "id": 71.1, "ood": 59.6, "time": 54},
        {"method": "ASAM", "id": 71.2, "ood": 59.9, "time": 56},
        {"method": "FisherSAM", "id": 71.0, "ood": 59.9, "time": 50},
        {"method": "FocalSAM", "id": 71.0, "ood": 59.8, "time": 63},
        {"method": "GA(500gen)", "id": 70.9, "ood": 60.5, "time": 116},
        {"method": "GA(400gen)", "id": 70.8, "ood": 60.4, "time": 90},
        {"method": "GA(300gen)", "id": 70.8, "ood": 60.2, "time": 71},
        {"method": "GA(200gen)", "id": 70.6, "ood": 59.9, "time": 49},
    ]


def build_color_map(methods: list[str]) -> dict[str, tuple[float, float, float, float]]:
    ga_methods = sorted([m for m in methods if m.startswith("GA(")])
    non_ga_methods = [m for m in methods if m not in ga_methods]

    color_map: dict[str, tuple[float, float, float, float]] = {}

    # Distinct non-blue colors for non-GA baselines.
    non_blue_palette = [
        "#E69F00",  # orange
        "#009E73",  # green
        "#D55E00",  # vermillion
        "#CC79A7",  # magenta
        "#8C564B",  # brown
        "#7F7F7F",  # gray
        "#BCBD22",  # olive
        "#2CA25F",  # deep green
    ]
    for i, m in enumerate(non_ga_methods):
        color_map[m] = non_blue_palette[i % len(non_blue_palette)]

    # Same color family for all GA budgets (dark-to-light blues).
    if ga_methods:
        ga_cmap = plt.cm.get_cmap("Blues")
        shades = np.linspace(0.55, 0.90, len(ga_methods))
        for shade, m in zip(shades, ga_methods):
            color_map[m] = ga_cmap(shade)

    return color_map


def compress_time(t: float, break_start: float = 130.0, break_end: float = 500.0, gap_width: float = 35.0) -> float:
    """Piecewise x-axis compression to visually skip a long middle interval."""
    if t <= break_start:
        return t
    if t >= break_end:
        return break_start + gap_width + (t - break_end)
    # Any value inside the omitted interval maps to the center of the visual gap.
    return break_start + gap_width / 2.0


def plot_time_vs_acc(output_path: str, show: bool = False) -> None:
    data = get_data()
    methods = [d["method"] for d in data]
    colors = build_color_map(methods)

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 12,
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.4), dpi=180, sharex=True)

    # Broken-axis settings for training time (omit long interval 130-500 min).
    break_start = 130.0
    break_end = 500.0
    gap_width = 35.0

    # Left panel: OOD accuracy
    ax_ood = axes[0]
    ga_rows = sorted([r for r in data if str(r["method"]).startswith("GA(")], key=lambda r: float(r["time"]))

    for row in data:
        x_plot = compress_time(float(row["time"]), break_start, break_end, gap_width)
        marker_style = "D" if str(row["method"]).startswith("GA(") else "o"
        ax_ood.scatter(
            x_plot,
            row["ood"],
            s=92,
            marker=marker_style,
            color=colors[row["method"]],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.96,
            zorder=3,
            label=row["method"],
        )

    # Add a subtle GA trajectory line to highlight budget-performance trade-off.
    ga_x = [compress_time(float(r["time"]), break_start, break_end, gap_width) for r in ga_rows]
    ga_ood = [float(r["ood"]) for r in ga_rows]
    ax_ood.plot(ga_x, ga_ood, color="#4c78a8", lw=1.6, alpha=0.65, linestyle="-", zorder=2)

    ax_ood.set_title("OOD Accuracy vs Training Time", pad=8)
    ax_ood.set_xlabel("Training Time (min)")
    ax_ood.set_ylabel("OOD Accuracy (%)")
    ax_ood.set_ylim(59.45, 60.65)
    ax_ood.set_yticks(np.arange(59.5, 60.61, 0.2))

    # Right panel: ID accuracy
    ax_id = axes[1]
    for row in data:
        x_plot = compress_time(float(row["time"]), break_start, break_end, gap_width)
        marker_style = "D" if str(row["method"]).startswith("GA(") else "o"
        ax_id.scatter(
            x_plot,
            row["id"],
            s=92,
            marker=marker_style,
            color=colors[row["method"]],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.96,
            zorder=3,
            label=row["method"],
        )

    ga_id = [float(r["id"]) for r in ga_rows]
    ax_id.plot(ga_x, ga_id, color="#4c78a8", lw=1.6, alpha=0.65, linestyle="-", zorder=2)

    ax_id.set_title("ID Accuracy vs Training Time", pad=8)
    ax_id.set_xlabel("Training Time (min)")
    ax_id.set_ylabel("ID Accuracy (%)")
    ax_id.set_ylim(69.4, 71.6)
    ax_id.set_yticks(np.arange(69.4, 71.61, 0.4))

    # Shared x-range, custom ticks, and cleaner spines.
    max_time = max(float(d["time"]) for d in data)
    max_time_plot = compress_time(max_time, break_start, break_end, gap_width)
    xticks = [0, 30, 60, 90, 120, break_start + gap_width + 0, break_start + gap_width + 39]
    xticklabels = ["0", "30", "60", "90", "120", "500", "539"]

    for ax in axes:
        ax.set_xlim(-8, max_time_plot + 18)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_facecolor("#fbfbfb")
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.40)
        ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#999999")
        ax.spines["bottom"].set_color("#999999")
        ax.tick_params(axis="both", colors="#333333")

        # Draw visual break marker near the omitted interval (bottom only).
        y0, y1 = ax.get_ylim()
        yr = y1 - y0
        xb = break_start + gap_width / 2.0
        ax.text(xb - 2.5, y0 + 0.04 * yr, "//", fontsize=12, color="#666666", fontweight="bold")

    # One global legend for both subplots.
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=colors[m],
            markeredgecolor="white",
            markeredgewidth=0.7,
            markersize=8,
            label=m,
        )
        for m in methods
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        frameon=True,
        framealpha=0.97,
        fancybox=True,
        edgecolor="#cdcdcd",
        facecolor="#fdfdfd",
        title="Optimizers",
        bbox_to_anchor=(0.5, 0.06),
    )

    fig.tight_layout(rect=[0, 0.055, 1, 1])
    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot training time vs accuracy.")
    parser.add_argument(
        "--output",
        type=str,
        default="train_time_vs_accuracy.png",
        help="Path to save the figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure window in addition to saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_time_vs_acc(output_path=args.output, show=args.show)
    print(f"Saved figure to: {args.output}")


if __name__ == "__main__":
    main()
