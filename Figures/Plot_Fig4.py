import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

DATA_COLUMNS = ["split_scheme", "condition", "metric", "mean", "ci95_low", "ci95_high"]
DATA_ROWS = [
    ("random", "full",                   "nDCG@100",   0.550, 0.443, 0.660),
    ("random", "full_uniform",           "nDCG@100",   0.421, 0.352, 0.501),
    ("random", "nnpu_only",              "nDCG@100",   0.243, 0.189, 0.296),
    ("random", "ranking_only_highscore", "nDCG@100",   0.461, 0.357, 0.573),
    ("random", "bce_highscore",          "nDCG@100",   0.378, 0.274, 0.497),

    ("random", "full",                   "Recall@100", 0.464, 0.359, 0.575),
    ("random", "full_uniform",           "Recall@100", 0.333, 0.267, 0.413),
    ("random", "nnpu_only",              "Recall@100", 0.195, 0.160, 0.226),
    ("random", "ranking_only_highscore", "Recall@100", 0.385, 0.300, 0.487),
    ("random", "bce_highscore",          "Recall@100", 0.287, 0.189, 0.400),

    ("random", "full",                   "AUPRC",      0.391, 0.272, 0.516),
    ("random", "full_uniform",           "AUPRC",      0.245, 0.174, 0.341),
    ("random", "nnpu_only",              "AUPRC",      0.103, 0.070, 0.135),
    ("random", "ranking_only_highscore", "AUPRC",      0.300, 0.192, 0.423),
    ("random", "bce_highscore",          "AUPRC",      0.232, 0.122, 0.354),

    ("random", "full",                   "AUROC",      0.837, 0.811, 0.864),
    ("random", "full_uniform",           "AUROC",      0.826, 0.805, 0.850),
    ("random", "nnpu_only",              "AUROC",      0.782, 0.749, 0.811),
    ("random", "ranking_only_highscore", "AUROC",      0.837, 0.813, 0.861),
    ("random", "bce_highscore",          "AUROC",      0.782, 0.744, 0.825),

    ("grouped", "full",                   "nDCG@100",   0.377, 0.269, 0.490),
    ("grouped", "full_uniform",           "nDCG@100",   0.216, 0.167, 0.261),
    ("grouped", "nnpu_only",              "nDCG@100",   0.119, 0.085, 0.149),
    ("grouped", "ranking_only_highscore", "nDCG@100",   0.256, 0.202, 0.323),
    ("grouped", "bce_highscore",          "nDCG@100",   0.246, 0.160, 0.346),

    ("grouped", "full",                   "Recall@100", 0.300, 0.196, 0.407),
    ("grouped", "full_uniform",           "Recall@100", 0.171, 0.134, 0.206),
    ("grouped", "nnpu_only",              "Recall@100", 0.120, 0.091, 0.148),
    ("grouped", "ranking_only_highscore", "Recall@100", 0.195, 0.149, 0.252),
    ("grouped", "bce_highscore",          "Recall@100", 0.200, 0.127, 0.284),

    ("grouped", "full",                   "AUPRC",      0.223, 0.127, 0.331),
    ("grouped", "full_uniform",           "AUPRC",      0.083, 0.061, 0.105),
    ("grouped", "nnpu_only",              "AUPRC",      0.037, 0.025, 0.048),
    ("grouped", "ranking_only_highscore", "AUPRC",      0.112, 0.075, 0.166),
    ("grouped", "bce_highscore",          "AUPRC",      0.113, 0.051, 0.192),

    ("grouped", "full",                   "AUROC",      0.806, 0.766, 0.847),
    ("grouped", "full_uniform",           "AUROC",      0.760, 0.730, 0.786),
    ("grouped", "nnpu_only",              "AUROC",      0.755, 0.727, 0.777),
    ("grouped", "ranking_only_highscore", "AUROC",      0.777, 0.747, 0.805),
    ("grouped", "bce_highscore",          "AUROC",      0.780, 0.741, 0.822),
]

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
    sub = data[(data["split_scheme"] == split_name) & (data["metric"] == metric)].copy()
    sub["condition"] = pd.Categorical(sub["condition"], CONDITION_ORDER, ordered=True)
    sub = sub.sort_values("condition")

    x = np.arange(len(CONDITION_ORDER))
    means = sub["mean"].to_numpy(float)
    lows = sub["ci95_low"].to_numpy(float)
    highs = sub["ci95_high"].to_numpy(float)
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
    df = pd.DataFrame(DATA_ROWS, columns=DATA_COLUMNS)

    fig, axes = plt.subplots(2, 4, figsize=(16.2, 9.0), sharey=False)

    for i, metric in enumerate(METRICS):
        metric_panel(axes[0, i], df, "random", metric, f"({chr(97+i)})")
        metric_panel(axes[1, i], df, "grouped", metric, f"({chr(97+i)})")

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
