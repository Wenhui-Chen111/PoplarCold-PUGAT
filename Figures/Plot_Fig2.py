"""Plot Fig. 2 using embedded, verified numerical summaries.

Run: python Plot_Fig2.py
Requires: numpy, matplotlib. No external data files are required.
Writes Fig2.png (600 dpi) and Fig2.pdf next to this script.
Default error bars: precomputed bootstrap 95% confidence intervals.
"""

from pathlib import Path
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

# All numerical values are embedded below. No CSV, Excel, or other input file
# is read. Entries follow METHOD_KEYS. Means, SDs, and bootstrap 95% CIs are
# copied from Supplementary Data 3, Primary_summary, without recomputation.
METHOD_KEYS = [
    "PoplarCold-PUGAT", "XGBoost", "R-GCN", "EMOGI-style variant",
    "Label Propagation", "PU-Bagging",
]

RANDOM_DATA = {'nDCG@100': {'mean': [0.5500413253605877,
                       0.23914992809032984,
                       0.25416344847188366,
                       0.10684104846957058,
                       0.0685147284142742,
                       0.04950489415254064],
              'std': [0.18026862281947487,
                      0.05873993375869682,
                      0.06364538294077365,
                      0.04082575501243823,
                      0.037689529167938225,
                      0.03904853484363033],
              'low': [0.4436291928184023,
                      0.20214107631062656,
                      0.21725593625375705,
                      0.08430028199272406,
                      0.04573787310471799,
                      0.027689493502993816],
              'high': [0.6585708543951956,
                       0.27186052607967515,
                       0.2909211206093215,
                       0.13279017712438743,
                       0.08954730524398463,
                       0.07295913206627493]},
 'Recall@100': {'mean': [0.4639344262295082,
                         0.1360655737704918,
                         0.16557377049180327,
                         0.08688524590163935,
                         0.07377049180327869,
                         0.031147540983606566],
                'std': [0.18304787496838862,
                        0.037898450020545794,
                        0.06668010256079387,
                        0.023248139219871747,
                        0.04034104661265956,
                        0.021093179306687193],
                'low': [0.3590163934426229,
                        0.11311475409836065,
                        0.1262295081967213,
                        0.07377049180327869,
                        0.04918032786885246,
                        0.019672131147540982],
                'high': [0.5737704918032787,
                         0.15737704918032785,
                         0.20655737704918034,
                         0.1,
                         0.09672131147540983,
                         0.04426229508196722]},
 'AUPRC': {'mean': [0.3914248275953733,
                    0.10481860441617583,
                    0.10904628809813444,
                    0.034395390455369566,
                    0.02189411390495103,
                    0.01585429266906464],
           'std': [0.20945830923809086,
                   0.040738391425674765,
                   0.035079711942845045,
                   0.01958586010536394,
                   0.011750688843577972,
                   0.011109851042848916],
           'low': [0.2721186642453343,
                   0.08115912900344527,
                   0.08867785113359121,
                   0.024296567588380294,
                   0.015238557226021387,
                   0.009833013629810044],
           'high': [0.5148796942171421,
                    0.1289233967053006,
                    0.12988986485886742,
                    0.046993506542604604,
                    0.028917502485278326,
                    0.022837319731985543]},
 'AUROC': {'mean': [0.8371468547093575,
                    0.7415955511983111,
                    0.6043487987932089,
                    0.7565337982899194,
                    0.7206119032746898,
                    0.7257082245745826],
           'std': [0.04330655268412715,
                   0.04166812087081243,
                   0.1992445679801072,
                   0.028027619954287118,
                   0.057029547530013415,
                   0.034286478710958905],
           'low': [0.811757670045725,
                   0.7188478656396807,
                   0.4887135511679759,
                   0.7397260296402928,
                   0.6876905501572607,
                   0.7045775299939744],
           'high': [0.8623046994491186,
                    0.7667276887714086,
                    0.7203828514855307,
                    0.7726083468118345,
                    0.7529073437493538,
                    0.7440583991401336]}}

GROUPED_DATA = {'nDCG@100': {'mean': [0.37674863093720645,
                       0.12729587393609954,
                       0.2027709766057694,
                       0.067553870409617,
                       0.018020007256838423,
                       0.011196121907255933],
              'std': [0.1887183029335253,
                      0.09724074264412003,
                      0.10164809797633416,
                      0.04995828024486349,
                      0.009932615197409475,
                      0.014310504859414094],
              'low': [0.271076847968,
                      0.07453588919842646,
                      0.1440201065073694,
                      0.04202393239304461,
                      0.012096127686741178,
                      0.0034083174741591582],
              'high': [0.49019846141733264,
                       0.18785348515557324,
                       0.264052777502128,
                       0.10039513742131413,
                       0.023860485213948377,
                       0.020193666435927203]},
 'Recall@100': {'mean': [0.29975727313455536,
                         0.07590310615667122,
                         0.11171913544284004,
                         0.059237983840023836,
                         0.02457249745228485,
                         0.011403056575281834],
                'std': [0.17872044638227377,
                        0.055668174130457944,
                        0.06802770764011434,
                        0.0445735831328356,
                        0.014034552594936524,
                        0.015166672939911554],
                'low': [0.19766011747730458,
                        0.04712581871535972,
                        0.07471811788619101,
                        0.03608170728428148,
                        0.016346646185355862,
                        0.003333333333333333],
                'high': [0.4070882151795308,
                         0.1118184760957238,
                         0.15536137054199642,
                         0.087728521086033,
                         0.032883275041411475,
                         0.021008878127522192]},
 'AUPRC': {'mean': [0.222693190781637,
                    0.04990823263746698,
                    0.09192676697264816,
                    0.020877620240914642,
                    0.00767605068098966,
                    0.006163629928523719],
           'std': [0.17569847271761518,
                   0.049746654803423426,
                   0.06771170968525378,
                   0.01978979439325055,
                   0.00170625082872407,
                   0.0014761856121072684],
           'low': [0.1274661454759037,
                   0.025043450734738112,
                   0.05565102575462929,
                   0.011656342815656378,
                   0.006691898943458447,
                   0.0053430467347395415],
           'high': [0.33111163483923484,
                    0.08232888787476868,
                    0.1343790313783888,
                    0.034285944859486564,
                    0.008695086049368214,
                    0.007059730761372461]},
 'AUROC': {'mean': [0.8061873972890408,
                    0.7321585930956158,
                    0.47656649785105243,
                    0.6751579387182509,
                    0.6550407297408147,
                    0.6863461500266446],
           'std': [0.06758267471618933,
                   0.04341559819331328,
                   0.1489581078830269,
                   0.0853969769603554,
                   0.06140236841050076,
                   0.02502843175202959],
           'low': [0.767783047275086,
                   0.7085575434142841,
                   0.39709667691644546,
                   0.6202960312989925,
                   0.6189238412929433,
                   0.6729252034027544],
           'high': [0.8460743329902483,
                    0.7592891654666815,
                    0.5717228933118655,
                    0.7200609910129202,
                    0.6915297137507493,
                    0.7018400861999997]}}

random_data = RANDOM_DATA
grouped_data = GROUPED_DATA


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


def main():
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


if __name__ == "__main__":
    main()
