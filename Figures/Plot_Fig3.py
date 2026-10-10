# -*- coding: utf-8 -*-
"""Plot manuscript Fig3."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

metrics = ["nDCG@100", "Recall@100", "AUPRC", "AUROC"]

variants = [
    "Full model",
    "w/o expression\n-based evidence",
    "w/o GO-based\n evidence",
    "w/o expression-\nand GO-based evidence",
]

data = {
    "nDCG@100": {
        "mean":    [0.550041, 0.556208, 0.321203, 0.315639],
        "ci_low":  [0.443629, 0.461276, 0.258483, 0.243233],
        "ci_high": [0.658571, 0.647433, 0.381138, 0.388832],
    },
    "Recall@100": {
        "mean":    [0.463934, 0.464119, 0.293946, 0.284200],
        "ci_low":  [0.359016, 0.372138, 0.246980, 0.239890],
        "ci_high": [0.573770, 0.556599, 0.338656, 0.327576],
    },
    "AUPRC": {
        "mean":    [0.391425, 0.394894, 0.137869, 0.141000],
        "ci_low":  [0.272119, 0.291279, 0.096003, 0.089326],
        "ci_high": [0.514880, 0.498152, 0.180753, 0.199182],
    },
    "AUROC": {
        "mean":    [0.837147, 0.840716, 0.783405, 0.782833],
        "ci_low":  [0.811758, 0.816520, 0.761642, 0.759023],
        "ci_high": [0.862305, 0.864143, 0.804802, 0.806410],
    },
}

colors = {
    "nDCG@100": "#F9CCE0",
    "Recall@100": "#FFE2CE",
    "AUPRC": "#C4D8E9",
    "AUROC": "#BEBCDF",
}


def add_value_labels(ax, xs, means, upper_err):
    """Add numeric mean labels above bars, positioned above the upper CI."""
    for xi, v, ue in zip(xs, means, upper_err):
        ax.text(
            xi,
            v + ue + 0.015,
            f"{v:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            rotation=0,
        )


def style_axis(ax):
    ax.grid(False)
    ax.set_axisbelow(False)

    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(1.0)
        ax.spines[s].set_color("#333333")

    ax.tick_params(axis="both", labelsize=10)


def plot_grouped_bars(ax, variants, data, bar_width=0.16, bar_step=0.18):
    x = np.arange(len(variants))
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * bar_step

    max_y = 0.0

    for metric, offset in zip(metrics, offsets):
        means = np.array(data[metric]["mean"], dtype=float)
        lows = np.array(data[metric]["ci_low"], dtype=float)
        highs = np.array(data[metric]["ci_high"], dtype=float)

        lower_err = means - lows
        upper_err = highs - means
        yerr = np.vstack([lower_err, upper_err])

        ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=yerr,
            capsize=3,
            label=metric,
            color=colors[metric],
            edgecolor="none",
            linewidth=0,
            error_kw=dict(
                elinewidth=1.0,
                capthick=1.0,
                ecolor="#444444",
            ),
        )

        add_value_labels(ax, x + offset, means, upper_err)
        max_y = max(max_y, np.max(highs))

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(0, min(1.08, max_y + 0.12))

    ax.legend(
        loc="upper center",
        ncol=4,
        fontsize=10,
        frameon=True,
        framealpha=1.0,
        borderaxespad=0.5,
    )

    style_axis(ax)


fig, ax = plt.subplots(1, 1, figsize=(10.2, 5.4))

plot_grouped_bars(
    ax,
    variants,
    data,
    bar_width=0.16,
    bar_step=0.18,
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

fig.savefig("Fig3.png", dpi=300, bbox_inches="tight")
fig.savefig("Fig3.pdf", bbox_inches="tight")
fig.savefig("Fig3.svg", bbox_inches="tight")

print("saved Fig3.png / pdf / svg")
