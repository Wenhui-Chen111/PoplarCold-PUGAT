from pathlib import Path
import csv
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

base = Path(__file__).resolve().parent
csv_path = base / "top200_ranking.csv"

with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

top30 = rows[:30]
ranks = [int(r["rank"]) for r in top30]
scores = [float(r["mean_rank_score"]) for r in top30]
genes = [r["gene_id"] for r in top30]

highlights = {
    "Potri.007G100600": {"label": "PAD4", "group": "Defense signaling"},
    "Potri.013G001300": {"label": "PIF3", "group": "CBF regulation"},
    "Potri.014G135900": {"label": "GID1B", "group": "Hormone signaling"},
    "Potri.005G068700": {"label": "PAD4", "group": "Defense signaling"},
    "Potri.014G135800": {"label": "BSL1", "group": "Hormone signaling"},
}

fig, ax = plt.subplots(figsize=(18, 8))

bg_color = ax._get_lines.get_next_color()
group_colors = {
    "Defense signaling": ax._get_lines.get_next_color(),
    "CBF regulation": ax._get_lines.get_next_color(),
    "Hormone signaling": ax._get_lines.get_next_color(),
}

for rank, score, gene in zip(ranks, scores, genes):
    if gene in highlights:
        continue
    ax.plot([rank, rank], [0, score], linewidth=2.0, alpha=0.28, color=bg_color, zorder=1)
    ax.scatter(rank, score, s=34, alpha=0.55, color=bg_color, zorder=2)

for rank, score, gene in zip(ranks, scores, genes):
    if gene not in highlights:
        continue
    info = highlights[gene]
    c = group_colors[info["group"]]
    ax.plot([rank, rank], [0, score], linewidth=5.0, color=c, zorder=2)
    ax.scatter(rank, score, s=190, color=c, edgecolor="white", linewidth=1.2, zorder=3)

annotation_positions = {
    "Potri.007G100600": (5.5, 0.79),
    "Potri.013G001300": (9.7, 0.815),
    "Potri.014G135900": (13.0, 0.775),
    "Potri.005G068700": (20.0, 0.79),
    "Potri.014G135800": (25.0, 0.76),
}

for rank, score, gene in zip(ranks, scores, genes):
    if gene not in highlights:
        continue
    info = highlights[gene]
    c = group_colors[info["group"]]
    ax.annotate(
        f"{info['label']}\n#{rank}",
        xy=(rank, score),
        xytext=annotation_positions[gene],
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=c,
        arrowprops=dict(arrowstyle="-", color=c, lw=2.2),
    )

ax.set_xlim(0.3, 30.7)
ax.set_ylim(0, 0.86)
ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
ax.set_xlabel("Ensemble rank (top-30 candidates)", fontsize=20)
ax.set_ylabel("Mean transformed ranking score", fontsize=20)
ax.tick_params(axis="both", labelsize=16)

legend_handles = [
    Line2D([0], [0], marker="o", linestyle="None", markersize=11,
           color=group_colors["Defense signaling"], label="Defense signaling"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=11,
           color=group_colors["CBF regulation"], label="CBF regulation"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=11,
           color=group_colors["Hormone signaling"], label="Hormone signaling"),
    Line2D([0], [0], marker="o", linestyle="None", markersize=11,
           color=bg_color, alpha=0.55, label="Other top-30 candidates"),
]
ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
          frameon=False, fontsize=17)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
plt.tight_layout()
plt.savefig(base / "Fig8.png", dpi=300, bbox_inches="tight")
plt.show()
