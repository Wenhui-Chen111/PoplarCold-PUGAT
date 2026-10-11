from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

methods = [
    "PoplarCold\n-PUGAT",
    "XGBoost",
    "R-GCN",
    "EMOGI-style\nvariant",
    "Label\nPropagation",
    "PU-Bagging",
]
metrics = ["nDCG@100", "Recall@100", "AUPRC", "AUROC"]

colors = ["#F9CCE0", "#9BDFDF", "#FFE2CE", "#C4D8E9", "#FFF3CA", "#BEBCDF"]

ERROR_BAR = "ci"          # "ci" for 95% CI, "std" for mean ± SD
SHOW_VALUE_LABELS = True   # User requested numerical labels on bars
VALUE_LABEL_ROTATION = 0   # horizontal value labels as requested

# Means, SDs, and precomputed bootstrap CIs come from one canonical table.
# Do not recompute bootstrap intervals inside a plotting script.
SOURCE = Path(__file__).with_name("Primary_summary.csv")
summary = pd.read_csv(SOURCE, float_precision="round_trip")
METHOD_KEYS = [
    "PoplarCold-PUGAT", "XGBoost", "R-GCN", "EMOGI-style variant",
    "Label Propagation", "PU-Bagging",
]


def load_split_data(split_scheme):
    sub = summary.loc[summary["split_scheme"].eq(split_scheme)].set_index("method")
    sub = sub.loc[METHOD_KEYS]
    fields = {"mean": "mean", "std": "SD", "low": "CI95_boot_low", "high": "CI95_boot_high"}
    return {
        metric: {key: sub[f"{metric}_{suffix}"].to_numpy(float).tolist()
                 for key, suffix in fields.items()}
        for metric in metrics
    }


random_data = load_split_data("random")
grouped_data = load_split_data("grouped")


def make_yerr(d):
    """Return matplotlib-compatible yerr for the selected error-bar mode."""
    mean = np.asarray(d["mean"], dtype=float)
    if ERROR_BAR.lower() == "std":
        return np.asarray(d["std"], dtype=float)
    if ERROR_BAR.lower() == "ci":
        low = np.asarray(d["low"], dtype=float)
        high = np.asarray(d["high"], dtype=float)
        return np.vstack([mean - low, high - mean])
    raise ValueError("ERROR_BAR must be either 'ci' or 'std'.")


def get_upper_err(yerr):
    return yerr if np.ndim(yerr) == 1 else yerr[1]


def nice_upper_limit(max_value):
    """Round subplot upper bound upward to a clean tick-like value."""
    if max_value <= 0.10:
        step = 0.02
    elif max_value <= 0.30:
        step = 0.05
    elif max_value <= 0.80:
        step = 0.10
    else:
        step = 0.20
    return np.ceil(max_value / step) * step


def plot_metric_panel(ax, metric, data, inner_label):
    x = np.arange(len(methods))
    values = np.asarray(data[metric]["mean"], dtype=float)
    yerr = make_yerr(data[metric])
    upper_err = get_upper_err(yerr)

    bars = ax.bar(
        x,
        values,
        yerr=yerr,
        capsize=3,
        width=0.7,
        color=colors,
        edgecolor="none",
        linewidth=0,
        error_kw=dict(ecolor="#444444", elinewidth=1.0, capthick=1.0),
        zorder=3,
    )

    ymax_data = float(np.max(values + upper_err))
    y_top = nice_upper_limit(ymax_data + 0.06)

    if SHOW_VALUE_LABELS:
        label_offset = max(0.008, y_top * 0.02)
        for xi, v, e in zip(x, values, upper_err):
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
    ax.set_xticklabels(methods, rotation=45, ha="right", fontsize=8.5)
    ax.set_ylim(0, y_top)
    ax.grid(False)

    ax.text(
        0.00, 1.04, inner_label,
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=12, fontweight="bold",
        clip_on=False,
    )

    for s in ["top", "right", "bottom", "left"]:
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("black")

    ax.tick_params(axis="y", labelsize=9)


fig, axes = plt.subplots(2, 4, figsize=(16.2, 9.0), sharey=False)

for i, metric in enumerate(metrics):
    plot_metric_panel(axes[0, i], metric, random_data, f"({chr(97 + i)})")

for i, metric in enumerate(metrics):
    plot_metric_panel(axes[1, i], metric, grouped_data, f"({chr(97 + i)})")

axes[0, 0].set_ylabel("Score", fontsize=12)
axes[1, 0].set_ylabel("Score", fontsize=12)

fig.text(0.02, 0.975, "A  Random split", ha="left", va="top", fontsize=15, fontweight="bold")
fig.text(0.02, 0.49,  "B  Grouped split", ha="left", va="top", fontsize=15, fontweight="bold")

legend_handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none") for c in colors
]
fig.legend(
    legend_handles,
    METHOD_KEYS,
    loc="upper center",
    ncol=6,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.53, 0.995),
)

fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.955], h_pad=2.8, w_pad=1.2)

out_png = Path(__file__).with_name("Fig2.png")
out_pdf = Path(__file__).with_name("Fig2.pdf")
fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"saved {out_png}")
print(f"saved {out_pdf}")
print(f"error bars: {'95% CI' if ERROR_BAR.lower() == 'ci' else 'mean ± SD'}")
