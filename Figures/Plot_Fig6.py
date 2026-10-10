import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

out_png = Path("Fig6.png")
out_pdf = Path("Fig6.pdf")
out_svg = Path("Fig6.svg")

data = [
    {
        "GO_ID": "GO:0009414",
        "GO_term": "response to water deprivation",
        "candidate_count": "7/161",
        "background_count": "109/14,051",
        "genes": "Potri.001G024300;Potri.001G150200;Potri.004G152800;Potri.006G083000;Potri.007G127500;Potri.007G145800;Potri.011G082800",
        "fold_enrichment": 5.605,
        "p_value": 2.55e-4,
        "q_value": 0.058876,
    },
    {
        "GO_ID": "GO:0009788",
        "GO_term": "negative regulation of abscisic acid-activated signaling pathway",
        "candidate_count": "3/161",
        "background_count": "20/14,051",
        "genes": "Potri.001G458400;Potri.007G127500;Potri.009G037300",
        "fold_enrichment": 13.091,
        "p_value": 0.001459,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0009939",
        "GO_term": "positive regulation of gibberellic acid mediated signaling pathway",
        "candidate_count": "2/161",
        "background_count": "6/14,051",
        "genes": "Potri.009G037300;Potri.013G028700",
        "fold_enrichment": 29.091,
        "p_value": 0.001899,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0006561",
        "GO_term": "proline biosynthetic process",
        "candidate_count": "2/161",
        "background_count": "7/14,051",
        "genes": "Potri.001G024300;Potri.011G082800",
        "fold_enrichment": 24.935,
        "p_value": 0.002639,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0010052",
        "GO_term": "guard cell differentiation",
        "candidate_count": "2/161",
        "background_count": "7/14,051",
        "genes": "Potri.011G123000;Potri.013G136100",
        "fold_enrichment": 24.935,
        "p_value": 0.002639,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0006470",
        "GO_term": "protein dephosphorylation",
        "candidate_count": "4/161",
        "background_count": "51/14,051",
        "genes": "Potri.001G358700;Potri.014G135800;Potri.015G036600;Potri.018G084900",
        "fold_enrichment": 6.845,
        "p_value": 0.002730,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0000054",
        "GO_term": "ribosomal subunit export from nucleus",
        "candidate_count": "2/161",
        "background_count": "8/14,051",
        "genes": "Potri.006G250400;Potri.018G030700",
        "fold_enrichment": 21.818,
        "p_value": 0.003492,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0009625",
        "GO_term": "response to insect",
        "candidate_count": "2/161",
        "background_count": "8/14,051",
        "genes": "Potri.002G203500;Potri.005G068700",
        "fold_enrichment": 21.818,
        "p_value": 0.003492,
        "q_value": 0.089619,
    },
    {
        "GO_ID": "GO:0019563",
        "GO_term": "glycerol catabolic process",
        "candidate_count": "2/161",
        "background_count": "8/14,051",
        "genes": "Potri.003G030900;Potri.010G226700",
        "fold_enrichment": 21.818,
        "p_value": 0.003492,
        "q_value": 0.089619,
    },
]

df = pd.DataFrame(data)

plot_df = df.iloc[::-1].copy()

q = plot_df["q_value"].to_numpy()
norm = plt.Normalize(vmin=q.min(), vmax=q.max())

fig, ax = plt.subplots(figsize=(10.5, 6.5))

bars = ax.barh(
    plot_df["GO_term"],
    plot_df["fold_enrichment"],
    color=plt.cm.Blues_r(norm(plot_df["q_value"])),
    edgecolor="none"
)

ax.axvline(1, linestyle="--", linewidth=1)

for bar, qval in zip(bars, plot_df["q_value"]):
    x = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    ax.text(
        x + 0.35,
        y,
        f"q={qval:.3f}",
        va="center",
        ha="left",
        fontsize=9
    )

sm = plt.cm.ScalarMappable(cmap=plt.cm.Blues_r, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, pad=0.02)
cbar.set_label("q-value (BH-FDR)", fontsize=10)

ax.set_title(
    "GO Biological Process enrichment",
    fontsize=13,
    weight="bold",
    pad=12
)

ax.set_xlabel("Fold enrichment", fontsize=11)
ax.set_ylabel("")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="y", labelsize=10)
ax.tick_params(axis="x", labelsize=10)

xmax = plot_df["fold_enrichment"].max()
ax.set_xlim(0, xmax + 6)

plt.tight_layout()

fig.savefig(out_png, dpi=300, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_svg, bbox_inches="tight")

print(f"saved: {out_png}")
print(f"saved: {out_pdf}")
print(f"saved: {out_svg}")
