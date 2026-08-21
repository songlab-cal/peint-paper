"""Real vs Real(other-subtree) historian comparison — the within-real variation baseline.

  (A) Indel-length CDF with Real(other subtree) added as a gray dotted line, alongside
      Real / PEINT ESM-C / PEINT ESM2 (all -refine).
  (B) Direct Real vs Real(other) 2-"model" scatter: per-family total indel events, with
      Pearson/Spearman — how much two subtrees of the SAME real family differ.

Indel events for Real (eval subtree), ESM2, and Real(other) are computed with the same
root-safe counter; ESM-C(refine) events are read from the refine run's cache.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from protevo import caching as pc
from paper.historian import get_all_evolutionary_counts_from_historian_output as gac

R1 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision1/simulations"
R2 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc"
ROS = f"{R1}/real_other_subtree"
FAM_JSON = "/scratch/users/akoehl/protein-evolution/local_data/final_sim_held_out_family.json"
FIG = "/scratch/users/akoehl/peint-paper/figures/output"


def esmc_refine_events():
    """The ESM-C -refine events dir (the 545-family one that is NOT the no-refine 15ef dir)."""
    cands = glob.glob(f"{R2}/simulations/historian_progressive/_cache/"
                      "get_all_evolutionary_counts_from_historian_output/*/*/*/*/output_events_dir")
    cands = [d for d in cands if len(glob.glob(f"{d}/*.txt")) >= 500 and "15ef65" not in d]
    return sorted(cands, key=lambda d: os.path.getmtime(d))[-1]


def indel_total(ev, f):
    p = f"{ev}/{f}.txt"
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, usecols=["event_type"])
    return int((d.event_type == "insertion").sum() + (d.event_type == "deletion").sum())


def indel_lengths(ev, fams):
    out = []
    for f in fams:
        p = f"{ev}/{f}.txt"
        if not os.path.exists(p):
            continue
        d = pd.read_csv(p, usecols=["event_type", "length"])
        out.extend(d.loc[d.event_type.isin(["insertion", "deletion"]), "length"].values)
    return np.sort(np.asarray(out, dtype=float))


def main():
    families = json.load(open(FAM_JSON))["families"]

    # Real (eval) + ESM2 counts: historian_compare cache (cache-hits from earlier runs).
    pc.set_cache_dir(f"{R2}/simulations/historian_compare/_cache"); pc.set_dir_levels(3)
    real = gac(sequences_dir=f"{R1}/real_msa_historian", tree_dir=f"{R1}/real_subtree_historian",
               families=families, num_processes=16)["output_events_dir"]
    esm2 = gac(sequences_dir=f"{R1}/peint_msa_historian", tree_dir=f"{R1}/real_subtree_historian",
               families=families, num_processes=16)["output_events_dir"]
    # Real(other subtree) counts: its own cache (different recon + tree).
    pc.set_cache_dir(f"{ROS}/_counts_cache"); pc.set_dir_levels(3)
    realo = gac(sequences_dir=f"{ROS}/real_other_subtree_historian", tree_dir=f"{ROS}/_prep/trees",
                families=families, num_processes=16)["output_events_dir"]
    esmc = esmc_refine_events()

    # ---- Figure A: indel-length CDF with Real(other) as gray dotted ----
    series = [("Real (eval subtree)", real, "purple", "-"),
              ("PEINT ESM-C", esmc, "#55a868", "-"),
              ("PEINT ESM2", esm2, "#d95f02", "-"),
              ("Real (other subtree)", realo, "gray", ":")]
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    print("indel length (median / P90 / n):")
    for name, ev, c, ls in series:
        L = indel_lengths(ev, families)
        cdf = np.arange(1, len(L) + 1) / len(L)
        ax.plot(L, cdf, label=f"{name} (med {np.median(L):.0f})", color=c, ls=ls, lw=1.5)
        print(f"  {name:22s} med={np.median(L):.0f} P90={np.percentile(L,90):.0f} n={len(L)}")
    ax.set_xlim(0, 50); ax.set_ylim(0, 1.02)
    ax.set_xlabel("Indel length (residues)"); ax.set_ylabel("Cumulative density")
    ax.set_title("Indel length CDF (+ real other-subtree)")
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    ax.grid(True, ls="--", lw=0.3)
    fig.tight_layout()
    for e in ("png", "pdf"):
        fig.savefig(f"{FIG}/historian_indel_length_cdf_with_realother.{e}", dpi=150)

    # ---- Figure B: Real vs Real(other) direct 2-model scatter (per-family indel events) ----
    rows = []
    for f in families:
        a, b = indel_total(real, f), indel_total(realo, f)
        if a is not None and b is not None:
            rows.append((f, a, b))
    df = pd.DataFrame(rows, columns=["family", "real", "real_other"])
    df.to_csv(f"{FIG}/real_vs_realother_indels.csv", index=False)
    x, y = df["real"].to_numpy(float), df["real_other"].to_numpy(float)
    pr, sr = pearsonr(x, y)[0], spearmanr(x, y)[0]
    fig2, ax2 = plt.subplots(figsize=(3.4, 3.4))
    ax2.scatter(x, y, s=14, alpha=0.6, edgecolor="black", linewidth=0.2, color="gray")
    lo = min(x.min(), y.min()) * 0.8
    hi = max(x.max(), y.max()) * 1.25
    ax2.plot([lo, hi], [lo, hi], "--", color="black", lw=0.7)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlim(lo, hi); ax2.set_ylim(lo, hi); ax2.set_aspect("equal")
    ax2.set_xlabel("Real eval-subtree indel events"); ax2.set_ylabel("Real other-subtree indel events")
    ax2.set_title("Real vs Real (two subtrees)")
    ax2.text(0.05, 0.95, f"Pearson {pr:.2f}\nSpearman {sr:.2f}\nn={len(df)}",
             transform=ax2.transAxes, va="top", fontsize=9)
    fig2.tight_layout()
    for e in ("png", "pdf"):
        fig2.savefig(f"{FIG}/real_vs_realother_indels.{e}", dpi=150)

    print(f"\nReal vs Real(other): Pearson {pr:.3f}  Spearman {sr:.3f}  n={len(df)}")
    print(f"median indel events/family: real(eval)={df.real.median():.0f}  real(other)={df.real_other.median():.0f}")
    print("DONE_REALOTHER_FIGS")


if __name__ == "__main__":
    main()
