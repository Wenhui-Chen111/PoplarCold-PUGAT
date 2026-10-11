"""Plot Fig. 4 using embedded, verified numerical summaries.

Run: python Plot_Fig4.py
Requires: numpy, matplotlib. No external data files are required.
Writes Fig4.png (600 dpi) and Fig4.pdf next to this script.
Error bars: precomputed bootstrap 95% confidence intervals.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

SHOW_VALUE_LABELS = True
VALUE_LABEL_ROTATION = 0

COLORS = ["#F9CCE0", "#9BDFDF", "#FFE2CE", "#C4D8E9", "#FFF3CA", "#BEBCDF"]

CONDITION_ORDER = [
    "full",
    "full_uniform",
    "nnpu_only",
    "ranking_only_highscore",
    "bce_highscore",
]
CONDITION_LABELS = {
    "full": "Full\n(high-score)",
    "full_uniform": "Full\n(uniform)",
    "nnpu_only": "nnPU-only",
    "ranking_only_highscore": "Ranking-only\n(high-score)",
    "bce_highscore": "BCE\n(high-score)",
}
METRICS = ["nDCG@100", "Recall@100", "AUPRC", "AUROC"]


# All numerical values are embedded below. No CSV, Excel, or other input file
# is read. Entries follow CONDITION_ORDER. The full-model values and 95% CIs
# are identical to Plot_Fig2.py and Supplementary Data 3, Primary_summary.
# Other conditions retain the numerical values in the supplied Fig. 4 data.
PLOT_DATA = {'random': {'nDCG@100': {'mean': [0.5500413253605877, 0.421, 0.243, 0.461, 0.378],
                         'low': [0.4436291928184023, 0.352, 0.189, 0.357, 0.274],
                         'high': [0.6585708543951956, 0.501, 0.296, 0.573, 0.497]},
            'Recall@100': {'mean': [0.4639344262295082, 0.333, 0.195, 0.385, 0.287],
                           'low': [0.3590163934426229, 0.267, 0.16, 0.3, 0.189],
                           'high': [0.5737704918032787, 0.413, 0.226, 0.487, 0.4]},
            'AUPRC': {'mean': [0.3914248275953733, 0.245, 0.103, 0.3, 0.232],
                      'low': [0.2721186642453343, 0.174, 0.07, 0.192, 0.122],
                      'high': [0.5148796942171421, 0.341, 0.135, 0.423, 0.354]},
            'AUROC': {'mean': [0.8371468547093575, 0.826, 0.782, 0.837, 0.782],
                      'low': [0.811757670045725, 0.805, 0.749, 0.813, 0.744],
                      'high': [0.8623046994491186, 0.85, 0.811, 0.861, 0.825]}},
 'grouped': {'nDCG@100': {'mean': [0.37674863093720645, 0.216, 0.119, 0.256, 0.246],
                          'low': [0.271076847968, 0.167, 0.085, 0.202, 0.16],
                          'high': [0.49019846141733264, 0.261, 0.149, 0.323, 0.346]},
             'Recall@100': {'mean': [0.29975727313455536, 0.171, 0.12, 0.195, 0.2],
                            'low': [0.19766011747730458, 0.134, 0.091, 0.149, 0.127],
                            'high': [0.4070882151795308, 0.206, 0.148, 0.252, 0.284]},
             'AUPRC': {'mean': [0.222693190781637, 0.083, 0.037, 0.112, 0.113],
                       'low': [0.1274661454759037, 0.061, 0.025, 0.075, 0.051],
                       'high': [0.33111163483923484, 0.105, 0.048, 0.166, 0.192]},
             'AUROC': {'mean': [0.8061873972890408, 0.76, 0.755, 0.777, 0.78],
                       'low': [0.767783047275086, 0.73, 0.727, 0.747, 0.741],
                       'high': [0.8460743329902483, 0.786, 0.777, 0.805, 0.822]}}}

def nice_upper_limit(max_value: float) -> float:
    """Round an upper bound to a clean axis limit."""
    if max_value <= 0.10:
        step = 0.02
    elif max_value <= 0.30:
        step = 0.05
    elif max_value <= 0.80:
        step = 0.10
    else:
        step = 0.20
    return float(np.ceil(max_value / step) * step)


def metric_panel(ax, data, split_name, metric, panel_label):
    sub = data[split_name][metric]

    x = np.arange(len(CONDITION_ORDER))
    means = np.asarray(sub["mean"], dtype=float)
    lows = np.asarray(sub["low"], dtype=float)
    highs = np.asarray(sub["high"], dtype=float)
    yerr = np.vstack([means - lows, highs - means])
    upper_err = highs - means

    ax.bar(
        x,
        means,
        yerr=yerr,
        capsize=3,
        width=0.70,
        color=COLORS[:len(CONDITION_ORDER)],
        edgecolor="none",
        linewidth=0,
        error_kw=dict(ecolor="#444444", elinewidth=1.0, capthick=1.0),
        zorder=3,
    )

    max_extent = float(np.max(means + upper_err))
    y_top = nice_upper_limit(max_extent * 1.12)
    if metric == "AUROC":
        y_top = min(1.0, max(y_top, 0.9))
    ax.set_ylim(0, y_top)

    if SHOW_VALUE_LABELS:
        label_offset = max(0.006, y_top * 0.018)
        for xi, v, e in zip(x, means, upper_err):
            ax.text(
                xi,
                v + float(e) + label_offset,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=8.0,
                rotation=VALUE_LABEL_ROTATION,
            )

    ax.set_title(metric, fontsize=11, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER], rotation=45, ha="right", fontsize=8.2)
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=9)

    ax.text(
        0.00, 1.04, panel_label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12, fontweight="bold",
        clip_on=False,
    )

    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("black")


def main():
    fig, axes = plt.subplots(2, 4, figsize=(16.2, 9.0), sharey=False)

    for i, metric in enumerate(METRICS):
        metric_panel(axes[0, i], PLOT_DATA, "random", metric, f"({chr(97+i)})")
        metric_panel(axes[1, i], PLOT_DATA, "grouped", metric, f"({chr(97+i)})")

    axes[0, 0].set_ylabel("Score", fontsize=12)
    axes[1, 0].set_ylabel("Score", fontsize=12)

    fig.text(0.02, 0.975, "A  Random split", ha="left", va="top", fontsize=15, fontweight="bold")
    fig.text(0.02, 0.49,  "B  Grouped split", ha="left", va="top", fontsize=15, fontweight="bold")

    legend_handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in COLORS[:5]]
    legend_labels = [
        "Full (high-score)", "Full (uniform)", "nnPU-only",
        "Ranking-only (high-score)", "BCE (high-score)"
    ]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.54, 0.995),
    )

    fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.955], h_pad=2.8, w_pad=1.2)

    out_png = Path(__file__).with_name("Fig4.png")
    out_pdf = Path(__file__).with_name("Fig4.pdf")
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"saved {out_png.name}")
    print(f"saved {out_pdf.name}")
    print("error bars: 95% CI")


if __name__ == "__main__":
    main()
