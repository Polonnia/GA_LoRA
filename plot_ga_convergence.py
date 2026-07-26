#!/usr/bin/env python3
"""
Plot synthetic GA convergence curves with confidence intervals.

Requirements:
- x-axis: generations (0 to 500)
- y-axis: ID and OOD accuracy
- 5 runs, show mean and 95% CI
- ID: monotonic increase and fast convergence
- OOD: oscillatory, little early gain, overall late increase
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


@dataclass
class CurveData:
    generations: np.ndarray
    id_runs: np.ndarray  # shape: [n_runs, n_points]
    ood_runs: np.ndarray  # shape: [n_runs, n_points]
    s_avg_runs: np.ndarray  # shape: [n_runs, n_points], in units of 10^-3
    s_avg_runs: np.ndarray  # shape: [n_runs, n_points], in units of 10^-3


def _sigmoid(x: np.ndarray, center: float, scale: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-(x - center) / scale))


def generate_synthetic_runs(
    n_runs: int = 5,
    max_generation: int = 500,
    step: int = 10,
    seed: int = 2026,
) -> CurveData:
    rng = np.random.default_rng(seed)
    generations = np.arange(0, max_generation + 1, step)

    id_runs = []
    ood_runs = []
    s_avg_runs = []

    for _ in range(n_runs):
        # ID: fast rise with rare late-stage improvements (still monotonic overall).
        start_id = 66.7 + rng.normal(0, 0.30)
        target_id = 70.3 + rng.normal(0, 0.28)
        speed = 52 + rng.normal(0, 13.0)
        speed = np.clip(speed, 28, 95)
        lag = rng.normal(10.0, 12.0)
        lag = np.clip(lag, -8.0, 34.0)

        g_shifted = np.clip(generations - lag, 0, None)
        fast_component = 0.90 * (1.0 - np.exp(-g_shifted / speed))
        slow_component = 0.10 * (1.0 - np.exp(-g_shifted / 250.0))
        id_curve = start_id + (target_id - start_id) * (fast_component + slow_component)

        # Keep a gentle late-stage rise instead of a fully flat tail.
        late_id_gain = rng.uniform(0.05, 0.16)
        late_id_trend = late_id_gain * _sigmoid(
            generations,
            center=400 + rng.normal(0, 8),
            scale=36 + rng.normal(0, 5),
        )
        id_curve += late_id_trend

        # Larger variance in early/mid generations, reduced noise after convergence.
        id_early_boost = 0.16 * (1.0 - _sigmoid(generations, center=175, scale=22))
        id_noise_sigma = 0.16 * (1.0 - _sigmoid(generations, center=310, scale=60)) + 0.022 + id_early_boost
        id_curve += rng.normal(0, id_noise_sigma, size=generations.shape)

        # Run-specific early-to-mid perturbation to widen CI while keeping overall trend realistic.
        bump_amp = rng.normal(0.0, 0.36)
        bump_center = rng.normal(155.0, 50.0)
        bump_width = np.clip(rng.normal(68.0, 24.0), 32.0, 120.0)
        id_curve += bump_amp * np.exp(-0.5 * ((generations - bump_center) / bump_width) ** 2)

        # Add sparse, tiny improvements to mimic occasional gains after near-convergence.
        prob = np.where(generations < 120, 0.03, 0.13)
        jump_mask = rng.random(generations.shape[0]) < prob
        jump_sizes = np.zeros_like(generations, dtype=float)
        jump_sizes[jump_mask] = rng.uniform(0.005, 0.030, size=jump_mask.sum())
        id_curve += np.cumsum(jump_sizes)

        # Force monotonic non-decreasing behavior for each run.
        id_curve = np.maximum.accumulate(id_curve)

        # OOD: near-flat early stage, oscillatory middle phase, then plateau near the end.
        start_ood = 57.1 + rng.normal(0, 0.28)
        late_gain = 3.7 + rng.normal(0, 0.30)
        late_trend = late_gain * _sigmoid(
            generations,
            center=255 + rng.normal(0, 8),
            scale=34 + rng.normal(0, 3),
        )

        osc_amp = 0.34 + rng.normal(0, 0.04)
        osc_phase = rng.uniform(0, np.pi)
        oscillation = osc_amp * np.sin(2 * np.pi * generations / 95 + osc_phase)

        early_suppression = 0.70 * _sigmoid(generations, center=115, scale=20)
        ood_curve_raw = start_ood + late_trend + early_suppression * oscillation

        # Blend into terminal regime while preserving visible late-stage uncertainty.
        plateau_level = 60.72 + rng.normal(0, 0.12)
        end_blend = _sigmoid(generations, center=390 + rng.normal(0, 3), scale=7 + rng.normal(0, 0.7))
        tail_osc = 0.036 * np.sin(2 * np.pi * generations / 160 + osc_phase / 2.0)
        ood_curve = (1.0 - end_blend) * ood_curve_raw + end_blend * (plateau_level + tail_osc)

        # Inject occasional early sharp drops (e.g., unstable selection in early generations).
        n_dips = int(rng.integers(3, 6))
        dip_centers = rng.choice(generations[3:19], size=n_dips, replace=False)
        for center in dip_centers:
            amp = rng.uniform(0.22, 0.50)  # drop depth in accuracy points
            width = rng.uniform(5.0, 10.0)
            ood_curve -= amp * np.exp(-0.5 * ((generations - center) / width) ** 2)

        # Explicitly boost early variance and keep late stage relatively tighter.
        early_boost = 0.12 * (1.0 - _sigmoid(generations, center=180, scale=28))
        tail_boost = 0.018 * _sigmoid(generations, center=390, scale=14)
        ood_noise_sigma = 0.24 * (1.0 - _sigmoid(generations, center=300, scale=65)) + 0.050 + early_boost + tail_boost
        ood_noise_sigma = ood_noise_sigma * (1.0 - 0.30 * end_blend)
        ood_curve += rng.normal(0, ood_noise_sigma, size=generations.shape)

        # S_avg: average-case sharpness, non-monotonic with early oscillation and late decay.
        # First 200 generations: oscillate around ~2.3-2.6 * 10^-3 (exploration phase).
        # Then: gradually decay toward ~0.35 * 10^-3.
        initial_s = 2.4 + rng.normal(0, 0.06)
        final_s = 0.35 + rng.normal(0, 0.016)
        
        # Early oscillatory phase (generations 0-200): moderate oscillation with minimal drift.
        early_osc_amp = 0.26 + rng.normal(0, 0.03)
        early_osc_freq = 2.0 * np.pi / 85.0  # ~85-generation period
        early_phase = rng.uniform(0, 2 * np.pi)
        early_osc = early_osc_amp * np.sin(early_osc_freq * generations + early_phase)
        
        # Slow early drift (slight downward trend in first 200 gen, but minimal).
        early_drift = 0.08 * _sigmoid(generations, center=100, scale=50)
        
        # Late decay phase (after 200 generations).
        late_decay_start = np.where(generations >= 200)[0][0] if np.any(generations >= 200) else len(generations) - 1
        decay_rate = 0.0035 + rng.normal(0, 0.0006)
        generations_from_200 = np.maximum(0, generations - 200)
        late_component = (final_s - (initial_s - early_drift)) * (1.0 - np.exp(-decay_rate * generations_from_200))
        
        # Combine: early oscillation + minimal drift + late decay.
        s_avg = initial_s + early_osc - early_drift + late_component
        
        # Add mid-stage non-monotonic bump (subtle rise around generation 250-350).
        mid_bump_amp = rng.uniform(0.06, 0.14)
        mid_bump_center = rng.uniform(250, 320)
        mid_bump_width = rng.uniform(70, 110)
        s_avg += mid_bump_amp * np.exp(-0.5 * ((generations - mid_bump_center) / mid_bump_width) ** 2)

        # Residual noise, mild across full range.
        residual_noise_sigma = 0.15 * (1.0 - _sigmoid(generations, center=180, scale=45))
        s_avg += rng.normal(0, residual_noise_sigma, size=generations.shape)

        # S_avg weakly correlated with OOD robustness gains.
        ood_progress = np.maximum(0, (ood_curve - ood_curve[0]) / np.maximum(1e-6, ood_curve[-1] - ood_curve[0]))
        s_avg = s_avg * (1.0 - 0.18 * ood_progress)

        # Floor at final value to ensure end-stage convergence.
        s_avg = np.maximum(s_avg, final_s * 0.90)

        id_runs.append(id_curve)
        ood_runs.append(ood_curve)
        s_avg_runs.append(s_avg)

    id_runs = np.vstack(id_runs)
    ood_runs = np.vstack(ood_runs)
    s_avg_runs = np.vstack(s_avg_runs)

    # Calibrate endpoints to requested targets while preserving overall trend/variance shape.
    t = generations / generations[-1]

    # ID: keep final around 70.3 and set start to 66.7.
    id_runs = id_runs + (70.3 - id_runs[:, -1].mean())
    id_start_delta = 66.7 - id_runs[:, 0].mean()
    id_runs = id_runs + id_start_delta * (1.0 - t)
    # At the very beginning, all runs should start almost identically.
    id_target_start = 66.7
    id_start_lock = np.exp(-generations / 35.0)
    id_runs = id_runs + (id_target_start - id_runs) * id_start_lock
    id_runs[:, 0] = id_target_start

    # OOD: set start to 60.8 and final to 63.9.
    ood_runs = ood_runs + (63.9 - ood_runs[:, -1].mean())
    ood_start_delta = 60.8 - ood_runs[:, 0].mean()
    ood_runs = ood_runs + ood_start_delta * (1.0 - t)
    ood_target_start = 60.8
    ood_start_lock = np.exp(-generations / 35.0)
    ood_runs = ood_runs + (ood_target_start - ood_runs) * ood_start_lock
    ood_runs[:, 0] = ood_target_start

    # Keep ID monotonic after final shift (shift itself preserves monotonicity, this is defensive).
    id_runs = np.maximum.accumulate(id_runs, axis=1)

    return CurveData(generations=generations, id_runs=id_runs, ood_runs=ood_runs, s_avg_runs=s_avg_runs)


def mean_and_ci95(runs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, lower CI, upper CI with t critical for n=5 (df=4)."""
    mean = runs.mean(axis=0)
    std = runs.std(axis=0, ddof=1)
    sem = std / np.sqrt(runs.shape[0])

    # 95% two-sided t critical for df=4.
    t_crit = 2.776
    margin = t_crit * sem
    return mean, mean - margin, mean + margin


def plot_convergence(data: CurveData, output_path: str, show: bool = False) -> None:
    try:
        plt.style.use("seaborn-v0_8-whitegrid")
    except OSError:
        plt.style.use("default")

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 11,
        }
    )

    x = data.generations

    id_mean, id_low, id_high = mean_and_ci95(data.id_runs)
    ood_mean, ood_low, ood_high = mean_and_ci95(data.ood_runs)

    # Compute S_avg metrics every 50 generations for sparse data points, but skip final few to avoid legend overlap.
    s_avg_x_indices = np.arange(0, len(x) - 2, 5)  # Skip last 2 points (100 generations) to clear legend area
    s_avg_x = x[s_avg_x_indices]
    s_avg_at_intervals = data.s_avg_runs[:, s_avg_x_indices]
    s_avg_mean = s_avg_at_intervals.mean(axis=0)
    s_avg_std = s_avg_at_intervals.std(axis=0, ddof=1)
    s_avg_sem = s_avg_std / np.sqrt(data.s_avg_runs.shape[0])
    s_avg_margin = 2.776 * s_avg_sem  # 95% CI with t-crit for df=4

    fig, ax = plt.subplots(figsize=(8.6, 5.3), dpi=180)

    id_color = "#1f77b4"
    ood_color = "#d62728"

    ax.plot(x, id_mean, color=id_color, lw=2.7, label="ID accuracy (mean)")
    ax.fill_between(x, id_low, id_high, color=id_color, alpha=0.20, label="ID 95% CI")

    ax.plot(x, ood_mean, color=ood_color, lw=2.7, label="OOD accuracy (mean)")
    ax.fill_between(x, ood_low, ood_high, color=ood_color, alpha=0.28, label="OOD 95% CI")
    ax.plot(x, ood_low, color=ood_color, lw=0.9, alpha=0.6, linestyle="--")
    ax.plot(x, ood_high, color=ood_color, lw=0.9, alpha=0.6, linestyle="--")

    # Mark final points (without text annotations).
    ax.scatter([x[-1]], [id_mean[-1]], color=id_color, s=28, zorder=4)
    ax.scatter([x[-1]], [ood_mean[-1]], color=ood_color, s=28, zorder=4)

    ax.set_xlim(0, x[-1])
    ax.set_ylim(56.2, 71.2)
    ax.set_xlabel("Generations", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.4)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.18)

    # Add secondary y-axis for S_avg (average-case sharpness) with better small-value visibility.
    ax2 = ax.twinx()
    s_avg_color = "#4a4a4a"
    ax2.errorbar(
        s_avg_x, s_avg_mean, yerr=s_avg_margin,
        fmt="o", color=s_avg_color, markersize=8, capsize=4, capthick=1.5,
        elinewidth=1.5, alpha=0.8, label="$S_{\\text{avg}}$ (mean ± 95% CI)"
    )
    ax2.set_ylabel("$S_{\\text{avg}}$ ($10^{-3}$)", fontsize=12, color="black")
    ax2.tick_params(axis="y", labelcolor="#333333")
    # Set tighter range with finer granularity to better show small values.
    ax2.set_ylim(0.25, 2.9)
    ax2.set_yticks([0.4, 0.8, 1.2, 1.6, 2.0, 2.4, 2.8])

    # Build one final merged legend so transparency settings are applied only once.
    lines_ax, labels_ax = ax.get_legend_handles_labels()
    lines_ax2, labels_ax2 = ax2.get_legend_handles_labels()
    legend = ax.legend(
        lines_ax + lines_ax2,
        labels_ax + labels_ax2,
        frameon=True,
        fancybox=True,
        edgecolor="#cfcfcf",
        facecolor="#e9edf2",
        loc="lower right",
        borderpad=0.8,
        labelspacing=0.45,
        handlelength=2.2,
        handletextpad=0.6,
        borderaxespad=0.8,
    )
    legend.get_frame().set_linewidth(0.9)
    legend.get_frame().set_alpha(0.45)

    for spine in ["top"]:
        ax.spines[spine].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax2.spines["right"].set_visible(True)
    ax2.spines["right"].set_color("#999999")
    ax.spines["left"].set_color("#999999")
    ax.spines["bottom"].set_color("#999999")
    ax.tick_params(axis="both", colors="#333333")
    ax2.spines["right"].set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")

    if show:
        plt.show()

    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot synthetic GA convergence curves.")
    parser.add_argument(
        "--output",
        type=str,
        default="ga_convergence_analysis.png",
        help="Path to save the figure.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot window in addition to saving.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = generate_synthetic_runs(n_runs=5, max_generation=500, step=10, seed=2026)
    plot_convergence(data, output_path=args.output, show=args.show)

    id_final = data.id_runs[:, -1].mean()
    ood_final = data.ood_runs[:, -1].mean()
    print(f"Saved figure to: {args.output}")
    print(f"Final mean ID acc: {id_final:.3f}")
    print(f"Final mean OOD acc: {ood_final:.3f}")


if __name__ == "__main__":
    main()
