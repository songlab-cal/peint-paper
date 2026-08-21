"""Figure 3 — structure metrics: how foldable each model's simulated leaves are.

Two leaf-pLDDT ECDF panels comparing every simulator against real, selectable from
the CLI:

``omegafold``  de-novo OmegaFold pLDDT of gap-stripped leaves.
``af2``        AF2Rank pLDDT (each leaf threaded onto the experimental template).

Models: WAG, LG, LG4X, LG+C60, LG+S256, PEINT (ESM2), PEINT (ESM-C), Real. The
classical/mixture baselines, PEINT (ESM2) and Real come from the revision-1 result
tables; PEINT (ESM-C) is parsed fresh from the revision-2 ESM-C run the SAME way
(per-family median leaf pLDDT). Colors reuse ``figure3_conservation``'s shared
seaborn-palette indices so the two figures are consistent; PEINT (ESM-C) takes a
free palette slot (cyan) so it pairs with the green PEINT (ESM2) and never collides
with LG+C60's brown.

Run from the repo root, e.g.::

    python -m figures.figure3_structure_metrics --panels omegafold af2
"""

import argparse
import glob
import os

import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper.plot_style import _set_publication_style
from paper.model_style import model_colors
import paper_config as cfg

mpl.rcParams["pdf.fonttype"] = 42

# Revision-1 holds the classical/mixture baselines + PEINT-ESM2 + real; revision-2
# holds the fresh ESM-C run (per the user, both live under DATA_ROOT/local_data).
R1 = cfg.DATA_ROOT / "local_data" / "results_revision1"
R2 = cfg.DATA_ROOT / "local_data" / "results_revision2_esmc"
OUT = cfg.FIGURES_DIR

MODEL_ORDER = [
    "WAG", "LG", "LG4X", "LG+C60", "LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real",
]
# Released result-table model names -> our display labels.
NAME_MAP = {
    "WAG": "WAG", "LG": "LG", "LG4X": "LG4X", "LG+C60": "LG+C60", "LG+S256": "LG+S256",
    "PEINT (Progressive)": "PEINT (ESM2)", "Real (other split)": "Real",
}


def _struct_mean_plddt(pdb):
    """Per-structure pLDDT = mean over ALL atom B-factors (matches the rev1 table)."""
    v = [float(l[60:66]) for l in open(pdb) if l.startswith("ATOM")]
    return float(np.mean(v)) if v else np.nan


def _rev1_df(csv_name, scale100=False):
    """Per-(family, model) pLDDT from a rev1 result table, relabeled to MODEL_ORDER."""
    df = pd.read_csv(f"{R1}/{csv_name}")
    df = df[df["model"].isin(NAME_MAP)].copy()
    if scale100 and df["plddt"].max() <= 1.5:  # AF2Rank stores 0-1; show on 0-100
        df["plddt"] = df["plddt"] * 100.0
    df["model"] = df["model"].map(NAME_MAP)
    return df[["family", "model", "plddt"]]


def _esmc_omegafold():
    """Per-family median OmegaFold leaf pLDDT for PEINT (ESM-C), parsed like rev1
    (per-structure all-atom mean B-factor, median over the ~30 folded leaves)."""
    rows = []
    for d in sorted(glob.glob(f"{R2}/omegafold/*/PEINT (Progressive)/structures")):
        fam = d.split("/omegafold/")[1].split("/")[0]
        if not os.path.exists(f"{d}/result.success"):
            continue
        per = [_struct_mean_plddt(p) for p in glob.glob(f"{d}/*.pdb")]
        per = [x for x in per if not np.isnan(x)]
        if per:
            rows.append({"family": fam, "model": "PEINT (ESM-C)", "plddt": float(np.median(per))})
    return pd.DataFrame(rows)


def _esmc_af2():
    """Per-family median AF2Rank pLDDT (0-1 -> x100) for PEINT (ESM-C), matching the
    rev1 table's per-family median over the 30 leaves."""
    rows = []
    for p in sorted(glob.glob(f"{R2}/af2/*/PEINT (Progressive)/scores/result.txt")):
        fam = p.split("/af2/")[1].split("/")[0]
        d = pd.read_csv(p)
        v = d["plddt"].dropna() if "plddt" in d.columns else pd.Series(dtype=float)
        if not v.empty:
            rows.append({"family": fam, "model": "PEINT (ESM-C)", "plddt": float(v.median()) * 100.0})
    return pd.DataFrame(rows)


def _ecdf(df, xlabel, stem):
    """Leaf-pLDDT ECDF over MODEL_ORDER; writes <stem>.{png,pdf,csv} to OUT."""
    colors = model_colors()
    df = df[df["model"].isin(MODEL_ORDER)].copy()
    present = [m for m in MODEL_ORDER if m in set(df["model"])]

    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    sns.ecdfplot(
        data=df, x="plddt", hue="model", hue_order=present, ax=ax,
        palette=[colors[m] for m in present], linewidth=1.0, legend=False,
    )
    ax.axvline(80, color="0.4", linestyle="--", linewidth=0.75, alpha=0.6)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Cumulative density", fontsize=10)
    ax.set_xlim(20, 100)
    ax.set_ylim(0, 1)
    handles = [mlines.Line2D([], [], color=colors[m], linewidth=2, label=m) for m in present]
    ax.legend(handles=handles, title="Model", fontsize=7, title_fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    sns.despine(ax=ax, top=True, right=True)
    for s in ax.spines.values():
        s.set_linewidth(0.5)

    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{stem}.png", bbox_inches="tight", dpi=300)
    df.to_csv(f"{OUT}/{stem}.csv", index=False)
    plt.close(fig)
    med = df.groupby("model")["plddt"].median().reindex(present).round(1)
    n = df.groupby("model")["family"].nunique().reindex(present)
    print(f"{stem}: median leaf pLDDT (n families)")
    for m in present:
        print(f"  {m:16s} {med[m]:5.1f}  (n={int(n[m])})")
    print(f"  -> {OUT}/{stem}.png")


def make_omegafold_ecdf():
    _ecdf(pd.concat([_rev1_df("omegafold_plddt.csv"), _esmc_omegafold()], ignore_index=True),
          "OmegaFold leaf pLDDT", "figure3_omegafold_plddt_ecdf")


def make_af2rank_ecdf():
    _ecdf(pd.concat([_rev1_df("af2rank_comparisons.csv", scale100=True), _esmc_af2()], ignore_index=True),
          "AF2Rank pLDDT", "figure3_af2rank_plddt_ecdf")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panels", nargs="+", choices=["omegafold", "af2"],
                    default=["omegafold", "af2"])
    args = ap.parse_args()
    _set_publication_style()
    if "omegafold" in args.panels:
        make_omegafold_ecdf()
    if "af2" in args.panels:
        make_af2rank_ecdf()


if __name__ == "__main__":
    main()
