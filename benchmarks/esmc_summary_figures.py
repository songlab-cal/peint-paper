"""Conservation JSD boxplot comparing ESM-C PEINT (rev2) against the revision-1 models, CPU-only.

Written to a NON-TRACKED folder (figures/output/esmc_summary/): mean JSD-vs-real per family
for WAG, LG, PEINT (ESM2), PEINT (ESM-C), Real (eval subtree). JSD-vs-real is computed FRESH
via collect_family_jsd (it was never run for ESM-C); rev1 models use rev1's aligned MSAs, ESM-C
uses rev2's. The Real (other-split) floor is derived inside family_jsd and is ~identical across
revisions (same real data).

The leaf-pLDDT ECDFs (OmegaFold + AF2Rank) that used to live here are now the canonical
figures.figure3_structure_metrics module (one source of truth, colors matched to
figure3_conservation).
"""

import json
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from paper.jsd import REAL, REAL_OTHER_SPLIT
from paper.model_style import model_colors
from figures.figure3_conservation import collect_family_jsd
import paper_config as cfg

R1 = str(cfg.RESULTS_R1_DIR)
R2 = str(cfg.RESULTS_R2_DIR)
FAM_JSON = str(cfg.HELDOUT_FAMILIES_JSON)
OUT = str(cfg.FIGURES_DIR / "esmc_summary")
THRESH = 0.7

# Colors from the shared canonical map so PEINT ESM-C matches the ECDFs /
# figure3_conservation.
ORDER = ["WAG", "LG", "PEINT ESM2", "PEINT ESM-C", "Real"]
COLORS = model_colors()


def build_jsd(families):
    d1 = {REAL: f"{R1}/mafft_add/old_sequences",
          "WAG": f"{R1}/simulations/wag", "LG": f"{R1}/simulations/lg",
          "PEINT (Progressive)": f"{R1}/mafft_add/peint_progressive_dir"}
    d2 = {REAL: f"{R2}/mafft_add/old_sequences",
          "PEINT (Progressive)": f"{R2}/mafft_add/peint_progressive_dir"}
    j1 = collect_family_jsd(families, d1, THRESH)
    j2 = collect_family_jsd(families, d2, THRESH)
    df = pd.DataFrame({
        "WAG": j1["WAG"], "LG": j1["LG"],
        "PEINT ESM2": j1["PEINT (Progressive)"],
        "PEINT ESM-C": j2["PEINT (Progressive)"],
        "Real": j1[REAL_OTHER_SPLIT],
    })
    return df


def jsd_boxplot(df):
    fig, ax = plt.subplots(figsize=(2.6, 3))
    sns.boxplot(data=df[ORDER], ax=ax, palette=[COLORS[m] for m in ORDER],
                showfliers=True,
                flierprops=dict(marker="o", markersize=2, markerfacecolor="black",
                                markeredgecolor="none", alpha=0.5),
                width=0.75, linewidth=0.5)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Mean JSD vs. real", fontsize=8)
    ax.yaxis.grid(True, linestyle=":", color="gray", linewidth=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    fig.savefig(f"{OUT}/conservation_jsd_boxplot.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/conservation_jsd_boxplot.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("\nJSD median vs real:")
    print(df[ORDER].median().round(3).to_dict())


def main():
    os.makedirs(OUT, exist_ok=True)
    families = json.load(open(FAM_JSON))["families"]
    print(f"{len(families)} families -> {OUT}")
    print("=== JSD (computing fresh) ===")
    jdf = build_jsd(families)
    jdf.to_csv(f"{OUT}/conservation_jsd.csv")
    jsd_boxplot(jdf)
    print("\nDONE_SUMMARY_FIGS")


if __name__ == "__main__":
    main()
