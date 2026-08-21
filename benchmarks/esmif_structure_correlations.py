"""Per-(family, model) correlation of the three structure metrics.

Two scatter panels written to figures/output/esmif/:
  * ESM-IF log-likelihood (Approach 1, seq1-structure) vs AF2Rank pLDDT
  * ESM-IF log-likelihood                                vs OmegaFold pLDDT

Each point is one (family, model) pair, colored by model via the shared paper.model_style
palette. Overall Pearson/Spearman are annotated; per-model correlations (within a model, across
families — i.e. do the metrics agree at the family level once the model is fixed) are printed and
saved to a stats CSV. Reads the per-(family, model) CSVs the other figures already wrote:
  esmif/esmif_gt_likelihood.csv, figure3_af2rank_plddt_ecdf.csv, figure3_omegafold_plddt_ecdf.csv
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.stats import pearsonr, spearmanr

from paper.model_style import model_colors
import paper_config as cfg

O = str(cfg.FIGURES_DIR)
OUT = os.path.join(O, "esmif")
MODEL_ORDER = ["WAG", "LG", "LG4X", "LG+C60", "LG+S256",
               "PEINT (ESM2)", "PEINT (ESM-C)", "Real"]


def _load():
    ll = pd.read_csv(f"{OUT}/esmif_gt_likelihood.csv")[["family", "model", "ll"]]
    af = pd.read_csv(f"{O}/figure3_af2rank_plddt_ecdf.csv").rename(columns={"plddt": "af2"})
    om = pd.read_csv(f"{O}/figure3_omegafold_plddt_ecdf.csv").rename(columns={"plddt": "omega"})
    df = (ll.merge(af[["family", "model", "af2"]], on=["family", "model"])
            .merge(om[["family", "model", "omega"]], on=["family", "model"]))
    return df


def _panel(ax, df, ycol, ylabel, colors):
    for m in MODEL_ORDER:
        d = df[df["model"] == m]
        if not len(d):
            continue
        ax.scatter(d["ll"], d[ycol], s=8, alpha=0.5, linewidths=0,
                   color=colors[m], label=m)
    x, y = df["ll"].to_numpy(float), df[ycol].to_numpy(float)
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y)[0]
    ax.set_xlabel("ESM-IF log-likelihood (seq1 structure)", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.text(0.04, 0.96, f"Pearson r = {pr:.2f}\nSpearman ρ = {sr:.2f}\nn = {len(df)}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.7", lw=0.5, alpha=0.85))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    return pr, sr


def _per_model_stats(df):
    rows = []
    for m in MODEL_ORDER:
        d = df[df["model"] == m]
        if len(d) < 3:
            continue
        rows.append({
            "model": m, "n": len(d),
            "pearson_af2": pearsonr(d["ll"], d["af2"])[0],
            "spearman_af2": spearmanr(d["ll"], d["af2"])[0],
            "pearson_omega": pearsonr(d["ll"], d["omega"])[0],
            "spearman_omega": spearmanr(d["ll"], d["omega"])[0],
        })
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = _load()
    colors = model_colors()
    print(f"{len(df)} (family, model) rows over {df['family'].nunique()} families")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    pa = _panel(axes[0], df, "af2", "AF2Rank pLDDT", colors)
    po = _panel(axes[1], df, "omega", "OmegaFold pLDDT", colors)
    handles = [mlines.Line2D([], [], marker="o", linestyle="", color=colors[m], label=m)
               for m in MODEL_ORDER]
    axes[1].legend(handles=handles, title="Model", fontsize=7.5, title_fontsize=8,
                   loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    fig.suptitle("ESM-IF likelihood vs folding pLDDT (per family × model)", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/esmif_vs_plddt_correlations.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    df.to_csv(f"{OUT}/esmif_vs_plddt_merged.csv", index=False)
    stats = _per_model_stats(df)
    stats.to_csv(f"{OUT}/esmif_vs_plddt_per_model_corr.csv", index=False)
    print(f"\noverall: ESM-IF vs AF2  Pearson {pa[0]:.3f} Spearman {pa[1]:.3f}")
    print(f"overall: ESM-IF vs OmegaFold  Pearson {po[0]:.3f} Spearman {po[1]:.3f}")
    print("\nper-model (within model, across families):")
    print(stats.round(3).to_string(index=False))
    print(f"\nsaved -> {OUT}/esmif_vs_plddt_correlations.{{png,pdf}} (+ merged/per-model CSVs)")


if __name__ == "__main__":
    main()
