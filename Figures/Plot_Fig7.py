from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

out_png = Path("Fig7.png")
out_pdf = Path("Fig7.pdf")
out_svg = Path("Fig7.svg")

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

df_a = pd.DataFrame({
    "timepoint": [
        "2+24+168 h union",
        "2 h",
        "24 h",
        "168 h"
    ],
    "fold_enrichment": [1.2964, 1.9829, 1.2066, 1.3312],
    "q_value": [0.030872, 0.271628, 0.271628, 0.030872]
})

rng = np.random.default_rng(42)

background_total = 22549
background_deg = 5798
candidate_n = 171
observed_overlap = 57

perm_overlap = rng.hypergeometric(
    ngood=background_deg,
    nbad=background_total - background_deg,
    nsample=candidate_n,
    size=10000
)
perm_values = perm_overlap / candidate_n

observed_fraction = observed_overlap / candidate_n   # 57/171 = 0.3333
random_mean = background_deg / background_total      # 5798/22549 = 0.2571
p_value = 0.015586

df_c = pd.DataFrame({
    "timepoint": ["2h", "24h", "168h"],
    "up": [2, 8, 31],
    "down": [0, 5, 11]
})

fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

ax = axes[0]

qmin = df_a["q_value"].min()
qmax = df_a["q_value"].max()
if np.isclose(qmin, qmax):
    norm_a = plt.Normalize(vmin=qmin - 1e-6, vmax=qmax + 1e-6)
else:
    norm_a = plt.Normalize(vmin=qmin, vmax=qmax)

bars_a = ax.bar(
    df_a["timepoint"],
    df_a["fold_enrichment"],
    color=plt.cm.Greens(norm_a(df_a["q_value"])),
    edgecolor="none"
)

ax.axhline(1, linestyle="--", linewidth=1, color="gray", alpha=0.8)

for bar, qval in zip(bars_a, df_a["q_value"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.04,
        f"q={qval:.3f}",
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.text(-0.12, 1.05, "(a)", transform=ax.transAxes,
        fontsize=13, fontweight="bold", ha="left", va="bottom")

ax.set_ylabel("Fold enrichment", fontsize=11)
ax.set_ylim(0, 2.15)
ax.tick_params(axis="x", labelsize=10, rotation=30)
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[1]

ax.hist(
    perm_values,
    bins=30,
    color="#76A5D3",
    edgecolor="none",
    alpha=0.9
)

ax.axvline(
    observed_fraction,
    color="#E66101",
    linewidth=1.6,
    label=f"Observed = {observed_fraction:.2f}"
)

ax.axvline(
    random_mean,
    color="gray",
    linestyle="--",
    linewidth=1.2,
    label=f"Random = {random_mean:.2f}"
)

ax.text(-0.12, 1.05, "(b)", transform=ax.transAxes,
        fontsize=13, fontweight="bold", ha="left", va="bottom")

ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_xlabel(f"Fraction of candidate DE in any cold timepoint(p = {p_value:.4f})", fontsize=11)
ax.set_ylabel("Permutation count", fontsize=11)
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

ax = axes[2]

bars_down = ax.bar(
    df_c["timepoint"],
    df_c["down"],
    color="#2166AC",
    label="Down-regulated"
)

bars_up = ax.bar(
    df_c["timepoint"],
    df_c["up"],
    bottom=df_c["down"],
    color="#B2182B",
    label="Up-regulated"
)

for i, row in df_c.iterrows():
    if row["down"] > 0:
        ax.text(
            i,
            row["down"] / 2,
            str(int(row["down"])),
            ha="center",
            va="center",
            fontsize=9,
            color="white"
        )
    if row["up"] > 0:
        ax.text(
            i,
            row["down"] + row["up"] / 2,
            str(int(row["up"])),
            ha="center",
            va="center",
            fontsize=9,
            color="white"
        )
    total = row["up"] + row["down"]
    ax.text(
        i,
        total + 0.6,
        str(int(total)),
        ha="center",
        va="bottom",
        fontsize=9
    )

ax.text(-0.12, 1.05, "(c)", transform=ax.transAxes,
        fontsize=13, fontweight="bold", ha="left", va="bottom")

ax.set_ylabel("Number of candidate cold DEGs", fontsize=11)
ax.set_xlabel("Earliest cold-responsive timepoint", fontsize=11)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.subplots_adjust(
    left=0.06,
    right=0.98,
    top=0.90,
    bottom=0.22,
    wspace=0.28
)

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")
plt.show()
