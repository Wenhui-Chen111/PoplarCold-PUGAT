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

# Control-condition summaries are preserved from the supplied Fig.4 script.
# The full-model summaries are always read from the same source as Fig.2.
SOURCE = Path(__file__).with_name("Primary_summary.csv")
CONTROL_SOURCE = Path(__file__).with_name("Fig4_plot_data.csv")

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
    df = pd.read_csv(CONTROL_SOURCE, float_precision="round_trip")
    summary = pd.read_csv(SOURCE, float_precision="round_trip")
    for scheme in ["random", "grouped"]:
        row = summary.loc[summary["split_scheme"].eq(scheme)
                          & summary["method"].eq("PoplarCold-PUGAT")].iloc[0]
        for metric in METRICS:
            mask = (df["split_scheme"].eq(scheme) & df["condition"].eq("full")
                    & df["metric"].eq(metric))
            df.loc[mask, ["mean", "ci95_low", "ci95_high"]] = [
                row[f"{metric}_mean"], row[f"{metric}_CI95_boot_low"],
                row[f"{metric}_CI95_boot_high"]]

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
