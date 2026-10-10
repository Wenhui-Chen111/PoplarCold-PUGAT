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

random_data = {
    "nDCG@100": {
        "mean": [0.550, 0.239, 0.254, 0.107, 0.069, 0.050],
        "std":  [0.180, 0.059, 0.064, 0.041, 0.038, 0.039],
        "low":  [0.444, 0.202, 0.218, 0.085, 0.046, 0.028],
        "high": [0.659, 0.272, 0.292, 0.133, 0.089, 0.073],
    },
    "Recall@100": {
        "mean": [0.464, 0.136, 0.166, 0.087, 0.074, 0.031],
        "std":  [0.183, 0.038, 0.067, 0.023, 0.040, 0.021],
        "low":  [0.359, 0.113, 0.128, 0.074, 0.049, 0.020],
        "high": [0.574, 0.159, 0.205, 0.100, 0.097, 0.044],
    },
    "AUPRC": {
        "mean": [0.391, 0.105, 0.109, 0.034, 0.022, 0.016],
        "std":  [0.209, 0.041, 0.035, 0.020, 0.012, 0.011],
        "low":  [0.272, 0.081, 0.089, 0.024, 0.015, 0.010],
        "high": [0.515, 0.129, 0.130, 0.047, 0.029, 0.023],
    },
    "AUROC": {
        "mean": [0.837, 0.742, 0.604, 0.757, 0.721, 0.726],
        "std":  [0.043, 0.042, 0.199, 0.028, 0.057, 0.034],
        "low":  [0.812, 0.719, 0.489, 0.740, 0.688, 0.705],
        "high": [0.862, 0.767, 0.720, 0.773, 0.753, 0.744],
    },
}

grouped_data = {
    "nDCG@100": {
        "mean": [0.377, 0.127, 0.203, 0.068, 0.018, 0.011],
        "std":  [0.189, 0.097, 0.102, 0.050, 0.010, 0.014],
        "low":  [0.271, 0.074, 0.144, 0.042, 0.012, 0.004],
        "high": [0.490, 0.190, 0.264, 0.100, 0.024, 0.020],
    },
    "Recall@100": {
        "mean": [0.300, 0.076, 0.112, 0.059, 0.025, 0.011],
        "std":  [0.179, 0.056, 0.068, 0.045, 0.014, 0.015],
        "low":  [0.198, 0.047, 0.074, 0.036, 0.016, 0.003],
        "high": [0.407, 0.111, 0.154, 0.089, 0.033, 0.021],
    },
    "AUPRC": {
        "mean": [0.223, 0.050, 0.092, 0.021, 0.008, 0.006],
        "std":  [0.176, 0.050, 0.068, 0.020, 0.002, 0.001],
        "low":  [0.127, 0.025, 0.056, 0.012, 0.007, 0.005],
        "high": [0.331, 0.082, 0.134, 0.034, 0.009, 0.007],
    },
    "AUROC": {
        "mean": [0.806, 0.732, 0.477, 0.675, 0.655, 0.686],
        "std":  [0.068, 0.043, 0.149, 0.085, 0.061, 0.025],
        "low":  [0.768, 0.709, 0.397, 0.620, 0.619, 0.673],
        "high": [0.846, 0.759, 0.572, 0.720, 0.692, 0.702],
    },
}


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
    [m.replace("\n", "") for m in methods],
    loc="upper center",
    ncol=6,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.53, 0.995),
)

fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.955], h_pad=2.8, w_pad=1.2)

out_png = "Fig2.png"
out_pdf = "Fig2.pdf"
fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"saved {out_png}")
print(f"saved {out_pdf}")
print(f"error bars: {'95% CI' if ERROR_BAR.lower() == 'ci' else 'mean ± SD'}")
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

random_data = {
    "nDCG@100": {
        "mean": [0.550, 0.239, 0.254, 0.107, 0.069, 0.050],
        "std":  [0.180, 0.059, 0.064, 0.041, 0.038, 0.039],
        "low":  [0.444, 0.202, 0.218, 0.085, 0.046, 0.028],
        "high": [0.659, 0.272, 0.292, 0.133, 0.089, 0.073],
    },
    "Recall@100": {
        "mean": [0.464, 0.136, 0.166, 0.087, 0.074, 0.031],
        "std":  [0.183, 0.038, 0.067, 0.023, 0.040, 0.021],
        "low":  [0.359, 0.113, 0.128, 0.074, 0.049, 0.020],
        "high": [0.574, 0.159, 0.205, 0.100, 0.097, 0.044],
    },
    "AUPRC": {
        "mean": [0.391, 0.105, 0.109, 0.034, 0.022, 0.016],
        "std":  [0.209, 0.041, 0.035, 0.020, 0.012, 0.011],
        "low":  [0.272, 0.081, 0.089, 0.024, 0.015, 0.010],
        "high": [0.515, 0.129, 0.130, 0.047, 0.029, 0.023],
    },
    "AUROC": {
        "mean": [0.837, 0.742, 0.604, 0.757, 0.721, 0.726],
        "std":  [0.043, 0.042, 0.199, 0.028, 0.057, 0.034],
        "low":  [0.812, 0.719, 0.489, 0.740, 0.688, 0.705],
        "high": [0.862, 0.767, 0.720, 0.773, 0.753, 0.744],
    },
}

grouped_data = {
    "nDCG@100": {
        "mean": [0.377, 0.127, 0.203, 0.068, 0.018, 0.011],
        "std":  [0.189, 0.097, 0.102, 0.050, 0.010, 0.014],
        "low":  [0.271, 0.074, 0.144, 0.042, 0.012, 0.004],
        "high": [0.490, 0.190, 0.264, 0.100, 0.024, 0.020],
    },
    "Recall@100": {
        "mean": [0.300, 0.076, 0.112, 0.059, 0.025, 0.011],
        "std":  [0.179, 0.056, 0.068, 0.045, 0.014, 0.015],
        "low":  [0.198, 0.047, 0.074, 0.036, 0.016, 0.003],
        "high": [0.407, 0.111, 0.154, 0.089, 0.033, 0.021],
    },
    "AUPRC": {
        "mean": [0.223, 0.050, 0.092, 0.021, 0.008, 0.006],
        "std":  [0.176, 0.050, 0.068, 0.020, 0.002, 0.001],
        "low":  [0.127, 0.025, 0.056, 0.012, 0.007, 0.005],
        "high": [0.331, 0.082, 0.134, 0.034, 0.009, 0.007],
    },
    "AUROC": {
        "mean": [0.806, 0.732, 0.477, 0.675, 0.655, 0.686],
        "std":  [0.068, 0.043, 0.149, 0.085, 0.061, 0.025],
        "low":  [0.768, 0.709, 0.397, 0.620, 0.619, 0.673],
        "high": [0.846, 0.759, 0.572, 0.720, 0.692, 0.702],
    },
}


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
    [m.replace("\n", "") for m in methods],
    loc="upper center",
    ncol=6,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.53, 0.995),
)

fig.tight_layout(rect=[0.02, 0.02, 0.995, 0.955], h_pad=2.8, w_pad=1.2)

out_png = "Fig2.png"
out_pdf = "Fig2.pdf"
fig.savefig(out_png, dpi=600, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.close(fig)

print(f"saved {out_png}")
print(f"saved {out_pdf}")
print(f"error bars: {'95% CI' if ERROR_BAR.lower() == 'ci' else 'mean ± SD'}")
