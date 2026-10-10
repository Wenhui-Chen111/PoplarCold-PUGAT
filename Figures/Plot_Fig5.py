# -*- coding: utf-8 -*-
"""Plot manuscript Fig5."""

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
    "w/o edge weight",
    "w/o relation gate",
    "w/o feature-graph gate",
]

data = {
    "nDCG@100": {
        "mean":    [0.550041, 0.395748, 0.472927, 0.538353],
        "ci_low":  [0.443629, 0.353724, 0.367832, 0.428754],
        "ci_high": [0.658571, 0.442644, 0.579927, 0.644519],
    },
    "Recall@100": {
        "mean":    [0.463934, 0.314754, 0.386885, 0.449180],
        "ci_low":  [0.359016, 0.275410, 0.296721, 0.339344],
        "ci_high": [0.573770, 0.359016, 0.483607, 0.557377],
    },
    "AUPRC": {
        "mean":    [0.391425, 0.206191, 0.316685, 0.385204],
        "ci_low":  [0.272119, 0.166231, 0.204739, 0.262657],
        "ci_high": [0.514880, 0.250061, 0.437253, 0.509313],
    },
    "AUROC": {
        "mean":    [0.837147, 0.824814, 0.818619, 0.837256],
        "ci_low":  [0.811758, 0.805592, 0.797248, 0.819530],
        "ci_high": [0.862305, 0.843102, 0.841510, 0.855596],
    },
}

colors = {
    "nDCG@100": "#F9CCE0",
    "Recall@100": "#FFE2CE",
    "AUPRC": "#C4D8E9",
    "AUROC": "#BEBCDF",
}


def add_value_labels(ax, xs, means, ci_highs):
    """Add mean-value labels above the upper end of each 95% CI error bar."""
    for xi, v, upper in zip(xs, means, ci_highs):
        ax.text(
            xi,
            upper + 0.015,
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


def plot_ablation(ax, variants, data, bar_width=0.16, bar_step=0.18):
    x = np.arange(len(variants))
    offsets = (np.arange(len(metrics)) - (len(metrics) - 1) / 2) * bar_step

    for metric, offset in zip(metrics, offsets):
        means = np.asarray(data[metric]["mean"], dtype=float)
        ci_low = np.asarray(data[metric]["ci_low"], dtype=float)
        ci_high = np.asarray(data[metric]["ci_high"], dtype=float)

        yerr = np.vstack([
            means - ci_low,
            ci_high - means,
        ])

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

        add_value_labels(ax, x + offset, means, ci_high)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=10)
    ax.set_ylim(0, 1.08)

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

plot_ablation(
    ax,
    variants,
    data,
    bar_width=0.16,
    bar_step=0.18,
)

fig.tight_layout(rect=[0, 0, 1, 0.92])

fig.savefig("Fig5.png", dpi=300, bbox_inches="tight")
fig.savefig("Fig5.pdf", bbox_inches="tight")
fig.savefig("Fig5.svg", bbox_inches="tight")

print("saved Fig5.png / pdf / svg")
