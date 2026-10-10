# -*- coding: utf-8 -*-
"""Plot manuscript FigS1."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "DejaVu Sans"
rcParams["mathtext.fontset"] = "dejavusans"
rcParams["axes.unicode_minus"] = False
rcParams["pdf.fonttype"] = 42
rcParams["ps.fonttype"] = 42
rcParams["svg.fonttype"] = "none"

C_NDCG = "#2C6E9B"
C_REC  = "#E0A030"
C_DEF  = "#C0392B"

panels = [
    (
        r"(a)  Ranking-loss weight $\lambda_r$", r"$\lambda_r$", 1.0,
        [0.0, 0.5, 1.0, 2.0, 4.0],
        [0.014764, 0.397969, 0.550041, 0.603359, 0.626689],
        [0.0, 0.346491, 0.443629, 0.508338, 0.520695],
        [0.035306, 0.455124, 0.658571, 0.689947, 0.717204],
        [0.013115, 0.309836, 0.463934, 0.514754, 0.537705],
        [0.0, 0.263934, 0.359016, 0.419672, 0.437705],
        [0.032787, 0.363934, 0.57377, 0.603279, 0.62623],
    ),
    (
        r"(b)  Class prior $\pi$", r"$\pi$", 0.013,
        [0.005, 0.013, 0.03, 0.05],
        [0.526253295, 0.550041, 0.509562439, 0.510694604],
        [0.434891733, 0.443629, 0.416809656, 0.41401162],
        [0.621558879, 0.658571, 0.60904308, 0.608285295],
        [0.431147541, 0.463934, 0.418032787, 0.414754098],
        [0.344262295, 0.359016, 0.327868852, 0.324590164],
        [0.527868852, 0.57377, 0.516393443, 0.508196721],
    ),
    (
        "(c)  Dropout rate", "dropout", 0.3,
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.371937, 0.430598, 0.550041, 0.625077, 0.634621],
        [0.323313, 0.390676, 0.443629, 0.529638, 0.565513],
        [0.418369, 0.470428, 0.658571, 0.705904, 0.687961],
        [0.270492, 0.322951, 0.463934, 0.539344, 0.570492],
        [0.231148, 0.280287, 0.359016, 0.444262, 0.516393],
        [0.309836, 0.367213, 0.57377, 0.621311, 0.616393],
    ),
    (
        "(d)  Positive train ratio", "train ratio", 0.5,
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.282175, 0.402989, 0.550041, 0.609615, 0.629961],
        [0.251141, 0.348841, 0.443629, 0.527076, 0.532911],
        [0.316071, 0.46376, 0.658571, 0.685005, 0.719132],
        [0.195833, 0.305063, 0.463934, 0.514964, 0.547727],
        [0.172917, 0.25443, 0.359016, 0.426737, 0.447727],
        [0.221875, 0.363291, 0.57377, 0.595593, 0.638636],
    ),
    (
        "(e)  Attention heads", "heads", 4,
        [2, 4, 8],
        [0.58573, 0.550041, 0.502988],
        [0.493318, 0.443629, 0.407617],
        [0.673221, 0.658571, 0.60078],
        [0.491803, 0.463934, 0.409836],
        [0.401639, 0.359016, 0.314754],
        [0.578689, 0.57377, 0.506557],
    ),
    (
        "(f)  Hidden dimension", "hidden dim", 128,
        [64, 128, 256],
        [0.675864, 0.550041, 0.455573],
        [0.635974, 0.443629, 0.397659],
        [0.715647, 0.658571, 0.518953],
        [0.588525, 0.463934, 0.363934],
        [0.540984, 0.359016, 0.304918],
        [0.636066, 0.57377, 0.42623],
    ),
    (
        "(g)  Warm-restart stages", "stages", 4,
        [0, 1, 2, 3, 4],
        [0.226158, 0.426982, 0.48328, 0.52702, 0.550041],
        [0.191195, 0.338089, 0.384918, 0.425398, 0.443629],
        [0.262115, 0.520205, 0.587389, 0.629881, 0.658571],
        [0.203279, 0.354098, 0.4, 0.439344, 0.463934],
        [0.17541, 0.281967, 0.306557, 0.344262, 0.359016],
        [0.229508, 0.434426, 0.501639, 0.542664, 0.57377],
    )
]

fig, axes = plt.subplots(2, 4, figsize=(21, 7.5))
axes = axes.ravel()


def draw_panel(ax, title, xlabel, anchor, vals,
               ndcg_mean, ndcg_low, ndcg_high,
               rec_mean, rec_low, rec_high):

    vals = np.asarray(vals, dtype=float)
    xpos = np.arange(len(vals), dtype=float)

    ndcg_mean = np.asarray(ndcg_mean, dtype=float)
    ndcg_low = np.asarray(ndcg_low, dtype=float)
    ndcg_high = np.asarray(ndcg_high, dtype=float)
    rec_mean = np.asarray(rec_mean, dtype=float)
    rec_low = np.asarray(rec_low, dtype=float)
    rec_high = np.asarray(rec_high, dtype=float)

    ax.fill_between(
        xpos, ndcg_low, ndcg_high,
        color=C_NDCG, alpha=0.14, linewidth=0, zorder=1
    )
    ax.fill_between(
        xpos, rec_low, rec_high,
        color=C_REC, alpha=0.11, linewidth=0, zorder=1
    )

    ax.plot(
        xpos, ndcg_mean, "-o", color=C_NDCG,
        lw=2.2, ms=7, zorder=3, label="nDCG@100"
    )
    ax.plot(
        xpos, rec_mean, "--s", color=C_REC,
        lw=2.2, ms=6.5, zorder=3, label="Recall@100"
    )

    anchor_idx = np.where(np.isclose(vals, anchor))[0]
    if len(anchor_idx) != 1:
        raise ValueError(f"Anchor {anchor} missing/duplicated in {title}")
    ai = int(anchor_idx[0])

    ax.scatter(
        [xpos[ai]], [ndcg_mean[ai]], s=300,
        facecolors="none", edgecolors=C_DEF,
        linewidths=2.5, zorder=6
    )
    ax.scatter(
        [xpos[ai]], [rec_mean[ai]], s=220,
        facecolors="none", edgecolors=C_DEF,
        linewidths=2.0, zorder=6
    )
    ax.annotate(
        "anchor", (xpos[ai], ndcg_mean[ai]),
        textcoords="offset points", xytext=(10, 10),
        color=C_DEF, fontsize=10.5, fontweight="bold", zorder=7
    )

    ax.set_title(title, fontsize=13, fontweight="bold", loc="left", pad=8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_xticks(xpos)
    ax.set_xticklabels([("%g" % v) for v in vals], fontsize=11)

    ax.grid(False)
    ax.set_axisbelow(True)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    ax.tick_params(axis="y", labelsize=10)
    ax.set_ylim(0, 0.78)


for ax, p in zip(axes[:7], panels):
    draw_panel(ax, *p)
    ax.set_ylabel("Score", fontsize=12)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.935),
    ncol=2,
    fontsize=11.5,
    frameon=True,
    framealpha=0.95
)

axes[7].set_visible(False)

fig.suptitle(
    "OAT sensitivity of PoplarCold-PUGAT to key hyperparameters "
    "(one-at-a-time; shaded bands = 95% CI; red circles = anchor)",
    fontsize=16, fontweight="bold", y=0.995
)

fig.tight_layout(rect=[0, 0, 1, 0.94], w_pad=1.0, h_pad=1.5)

fig.savefig("FigS1.png", dpi=300, bbox_inches="tight")
fig.savefig("FigS1.pdf", bbox_inches="tight")
fig.savefig("FigS1.svg", bbox_inches="tight")

print("saved FigS1.png / pdf / svg")
