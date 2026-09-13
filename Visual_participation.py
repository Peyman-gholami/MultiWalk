#%%
"""Plot per-client participation probabilities for all patterns.

Uses the same rates / period as the kube fedjob_* YAMLs (expected #clients ≈ 3).
Uniform is shown as a flat p=0.3 per client for comparison; in code it actually
samples a fixed count without replacement each round.

Run:
    python Visual_participation.py
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.participation import participation_probabilities

# --- configuration (matches ~/.kube/fedjob_*.yaml) ---
NUM_CLIENTS = 10
PERIOD = 50
NUM_PERIODS = 2  # how many cycles to draw for time-varying patterns
OUTPUT_DIR = "./figures/participation"

UNIFORM_RATE = 0.3  # fedjob.yaml

STATIONARY_HIGHS = [0.85, 0.75, 0.55, 0.35, 0.15, 0.1, 0.08, 0.07, 0.05, 0.05]

STAIR_SINE_HIGHS = [1.0, 0.95, 0.9, 0.75, 0.5, 0.3, 0.2, 0.15, 0.1, 0.06]
STAIR_SINE_LOWS = [0.4, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.01, 0.0, 0.0]

INTERLEAVED_HIGHS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.9, 0.85, 0.724778]

# Mirror pairs: clients 0-4 decay, 5-9 rise; sum_i p_i = 3 every round.
EXPONENTIAL_HIGHS = [0.85, 0.70, 0.55, 0.40, 0.25, 0.85, 0.70, 0.55, 0.40, 0.25]
EXPONENTIAL_LOWS = [0.05] * 10

PATTERN_CONFIGS = [
    {
        "name": "uniform",
        "title": "Uniform (p = 0.3)",
        "high": UNIFORM_RATE,
        "low": 0.0,
        "period": PERIOD,
    },
    {
        "name": "stationary",
        "title": "Stationary",
        "high": STATIONARY_HIGHS,
        "low": 0.0,
        "period": PERIOD,
    },
    {
        "name": "staircase",
        "title": "Staircase",
        "high": STAIR_SINE_HIGHS,
        "low": STAIR_SINE_LOWS,
        "period": PERIOD,
    },
    {
        "name": "sine",
        "title": "Sine",
        "high": STAIR_SINE_HIGHS,
        "low": STAIR_SINE_LOWS,
        "period": PERIOD,
    },
    {
        "name": "interleaved_sine",
        "title": "Interleaved sine",
        "high": INTERLEAVED_HIGHS,
        "low": 0.0,
        "period": PERIOD,
    },
    {
        "name": "exponential",
        "title": "Exponential (half decay / half rise)",
        "high": EXPONENTIAL_HIGHS,
        "low": EXPONENTIAL_LOWS,
        "period": PERIOD,
    },
]


def probability_matrix(pattern: str, high, low, period: int, num_rounds: int) -> np.ndarray:
    """Return array shape (num_rounds, NUM_CLIENTS)."""
    rows = []
    for r in range(num_rounds):
        rows.append(
            participation_probabilities(
                round_number=r,
                num_clients=NUM_CLIENTS,
                pattern=pattern,
                high=high,
                low=low,
                period=period,
            )
        )
    return np.asarray(rows, dtype=float)


def plot_all_patterns():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_rounds = PERIOD * NUM_PERIODS
    rounds = np.arange(num_rounds)

    cmap = plt.get_cmap("tab10")
    fig, axes = plt.subplots(len(PATTERN_CONFIGS), 1, figsize=(10, 12), sharex=True)
    if len(PATTERN_CONFIGS) == 1:
        axes = [axes]

    for ax, cfg in zip(axes, PATTERN_CONFIGS):
        probs = probability_matrix(
            cfg["name"], cfg["high"], cfg["low"], cfg["period"], num_rounds
        )
        for i in range(NUM_CLIENTS):
            ax.plot(
                rounds,
                probs[:, i],
                color=cmap(i % 10),
                linewidth=1.6,
                label=f"client {i + 1}",
            )
        ax.set_ylabel(r"$p_i^r$")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(cfg["title"], fontsize=11)
        ax.grid(True, alpha=0.25)
        if cfg["name"] in ("staircase", "sine", "interleaved_sine", "exponential"):
            for k in range(1, NUM_PERIODS):
                ax.axvline(k * PERIOD, color="0.7", linewidth=0.8, linestyle="--")

    axes[-1].set_xlabel("Round")
    # One shared legend outside
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        fontsize=8,
        frameon=False,
    )
    fig.suptitle("Client participation probabilities", fontsize=14, y=0.995)
    fig.tight_layout(rect=[0, 0, 0.86, 0.98])
    out = os.path.join(OUTPUT_DIR, "participation_patterns.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_heatmap_grid():
    """One heatmap per pattern: clients × rounds."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    num_rounds = PERIOD * NUM_PERIODS

    fig, axes = plt.subplots(
        1,
        len(PATTERN_CONFIGS),
        figsize=(15, 3.4),
        sharey=True,
        constrained_layout=True,
    )
    for ax, cfg in zip(axes, PATTERN_CONFIGS):
        probs = probability_matrix(
            cfg["name"], cfg["high"], cfg["low"], cfg["period"], num_rounds
        )
        im = ax.imshow(
            probs.T,
            aspect="auto",
            origin="lower",
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            extent=[0, num_rounds, 0.5, NUM_CLIENTS + 0.5],
        )
        ax.set_title(cfg["title"], fontsize=10)
        ax.set_xlabel("Round")
    axes[0].set_ylabel("Client")
    fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02, label=r"$p_i^r$")
    fig.suptitle("Participation probability heatmaps", fontsize=13)
    out = os.path.join(OUTPUT_DIR, "participation_heatmaps.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot_all_patterns()
    plot_heatmap_grid()
    print(f"Figures written under {OUTPUT_DIR}/")
