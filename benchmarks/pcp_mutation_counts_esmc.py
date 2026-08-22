"""PCP (parent-child pair) per-branch substitution counts: Real vs ESM2 PEINT (rev1) vs
ESM-C PEINT (rev2), on the already-historian'd reconstructions.

Reuses the aggregation from figures/figure3_pcp_mutation_counts.py (get_branches +
match_branches) — SKIPS the AliSim classical-simulator setup. For each family it matches each
model's reconstruction branches to the real subtree branches by topology (so every model is
scored on the SAME branches) and records per-branch (per-model branch length, #substitutions,
#sites). Figure style matches figure3: grouped box plots of fraction-mutated-sites over
branch-length bins of 0.1 in the 0-1.1 range.

Inputs (historian reconstructions):
  Real  = rev1 real_msa_historian + real_subtree_historian
  ESM2  = rev1 peint_msa_historian                (ESM2-150 PEINT)
  ESM-C = rev2 refine remove_dummy reconstruction (ESM-C PEINT, matched -refine option)
Trees for the PEINT recons: the shared cherryml sim trees (SIM/trees); name-matching verified.
"""

import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from ete3 import Tree

from protevo.io import read_tree
from protevo.utils import read_msa
from paper.historian import esmc_historian_dirs
from paper.model_style import model_colors
import paper_config as cfg

from figures.figure3_pcp_mutation_counts import get_branches, match_branches

_MC = model_colors()
R1 = str(cfg.RESULTS_R1_DIR / "simulations")
R2 = str(cfg.RESULTS_R2_DIR)
SIM_TREES = str(cfg.TREE_DIR)
FAM_JSON = str(cfg.HELDOUT_FAMILIES_JSON)
FIG_OUT = str(cfg.FIGURES_DIR)
# The "refine" ESM-C Historian run, same one every other ESM-C indel panel uses.
# Resolved inside main() so importing this module never requires the rev2 data.
COLORS = {k: _MC[k] for k in ("Real", "PEINT ESM2 (rev1)", "PEINT ESM-C")}
MODEL_ORDER = ["PEINT ESM2 (rev1)", "PEINT ESM-C", "Real"]


def main():
    esmc_recon = esmc_historian_dirs(R2, "refine")["reconstructions"]
    families = json.load(open(FAM_JSON))["families"]
    rows = []
    for i, fam in enumerate(families):
        rp = f"{R1}/real_subtree_historian/{fam}.txt"
        rm = f"{R1}/real_msa_historian/{fam}.txt"
        if not (os.path.exists(rp) and os.path.exists(rm)):
            continue
        real_tree = Tree(rp, format=1)
        real_msa = read_msa(rm)
        eval_leaf_set = set(n.name for n in real_tree.get_leaves())
        real_br = get_branches(real_tree, real_msa)

        # Real: its own inferred substitutions; branch length = its own (historian) length.
        for lk, v in real_br.items():
            if v["num_sites"] > 0:
                rows.append((fam, "Real", v["branch_length"], v["mutations"], v["num_sites"]))

        # ESM2 / ESM-C: matched to real branches; branch length = each model's OWN length
        # (sim_branch_length), matching figure3's per-model binning.
        for label, recon in [("PEINT ESM2 (rev1)", f"{R1}/peint_msa_historian"),
                             ("PEINT ESM-C", esmc_recon)]:
            fp = f"{recon}/{fam}.txt"
            if not os.path.exists(fp):
                continue
            matched = match_branches(real_br,
                                     get_branches(read_tree(f"{SIM_TREES}/{fam}.txt").to_ete3(),
                                                  read_msa(fp)),
                                     eval_leaf_set)["matched"]
            for lk, v in matched.items():
                if v["sim_sites"] > 0:
                    rows.append((fam, label, v["sim_branch_length"],
                                 v["sim_mutations"], v["sim_sites"]))
        if (i + 1) % 50 == 0:
            print(f"...{i+1}/{len(families)} families")

    df = pd.DataFrame(rows, columns=["family", "model", "branch_length", "mutations", "sites"])
    df["rate"] = df["mutations"] / df["sites"]
    os.makedirs(FIG_OUT, exist_ok=True)
    df.to_csv(f"{FIG_OUT}/pcp_mutation_counts_esmc.csv", index=False)

    print("\n=== overall substitutions/site (total muts / total sites) ===")
    for m in MODEL_ORDER:
        d = df[df["model"] == m]
        print(f"  {m:20s} n_branches={len(d):7d}  subs/site={d['mutations'].sum()/d['sites'].sum():.4f}")

    # ---- figure3-style grouped boxplot: fraction mutated sites over 0.1 branch-length bins ----
    df["bl_q"] = df["branch_length"].apply(lambda x: math.floor(x * 10) / 10)
    sub = df[df["branch_length"] < 1.1]
    bins = sorted(sub["bl_q"].unique())
    n = len(MODEL_ORDER)
    positions, boxdata, boxcolors = [], [], []
    for i, bl in enumerate(bins):
        for j, m in enumerate(MODEL_ORDER):
            d = sub[(sub["bl_q"] == bl) & (sub["model"] == m)]["rate"]
            if len(d) > 0:
                positions.append(i + (j - n / 2 + 0.5) * 0.2)
                boxdata.append(d.values)
                boxcolors.append(COLORS[m])

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(boxdata, positions=positions, widths=0.15, patch_artist=True,
                    boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5),
                    flierprops={"marker": "o", "markersize": 0.3},
                    medianprops=dict(linestyle="--", color="black", linewidth=0.5))
    for patch, c in zip(bp["boxes"], boxcolors):
        patch.set_facecolor(c)
    ax.legend(handles=[Patch(facecolor=COLORS[m], edgecolor="black", linewidth=0.5, label=m)
                       for m in MODEL_ORDER], title="Model", fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(bins)))
    ax.set_xticklabels([f"{bl:.1f}" for bl in bins])
    ax.set_xlabel("Branch length (per-model)", fontsize=12)
    ax.set_ylabel("Fraction mutated sites", fontsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_OUT}/pcp_mutation_counts_esmc.{ext}", dpi=150, bbox_inches="tight")
    print(f"\nsaved: {FIG_OUT}/pcp_mutation_counts_esmc.{{png,pdf,csv}}  ({len(df)} branch rows)")


if __name__ == "__main__":
    main()
