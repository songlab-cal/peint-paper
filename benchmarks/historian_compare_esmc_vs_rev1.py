"""Compare Historian-inferred indel events: ESM-C PEINT (revision 2) vs the original
ESM2 PEINT (revision 1) vs real.

ESM-C event tables come from ``historian_esmc_progressive.py`` (revision-2 job). ESM2 and
real counts are derived here from revision 1's *reconstructions* (they were never persisted
as counts) with the same root-safe counter, paired with rev1's eval-subtree trees
(``real_subtree_historian`` — topologically identical for real & simulated subtrees, node
names verified to match the reconstructions exactly).

Per family the metric is total indel *events* = #insertions + #deletions (gap opens), as in
figures/figure3_indels.py. Reports Pearson/Spearman for real-vs-ESM-C, real-vs-ESM2, and
ESM2-vs-ESM-C, and saves a 3-panel scatter.
"""

import argparse
import json
import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
# Keep text as editable TrueType (not outlines) so labels are editable in Illustrator.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

from protevo import caching as protevo_caching
from paper.historian import get_all_evolutionary_counts_from_historian_output

R1 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision1/simulations"
R2 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc"
FAM_JSON = "/scratch/users/akoehl/protein-evolution/local_data/final_sim_held_out_family.json"
FIG_OUT = "/scratch/users/akoehl/peint-paper/figures/output"

# ESM-C per-family event tables (revision-2 historian job output).
ESMC_EVENTS = sorted(glob.glob(
    f"{R2}/simulations/historian_progressive/_cache/"
    "get_all_evolutionary_counts_from_historian_output/*/*/*/*/output_events_dir"))[0]


def indel_events(events_dir, fam):
    """Total indel events (#insertions + #deletions) for a family, or None if missing."""
    p = os.path.join(events_dir, f"{fam}.txt")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    vc = df["event_type"].value_counts()
    return int(vc.get("insertion", 0) + vc.get("deletion", 0))


def _annotate(ax, x, y, xlabel, ylabel):
    x = np.asarray(x); y = np.asarray(y)
    pr = pearsonr(x, y)[0]; sr = spearmanr(x, y)[0]
    ax.scatter(x, y, s=14, alpha=0.6, edgecolor="black", linewidth=0.2)
    lo = min(x.min(), y.min()) * 0.8
    hi = max(x.max(), y.max()) * 1.25
    ax.plot([lo, hi], [lo, hi], "--", color="grey", linewidth=0.7)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.text(0.05, 0.95, f"Pearson {pr:.2f}\nSpearman {sr:.2f}",
            transform=ax.transAxes, va="top", fontsize=9)
    return pr, sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--esmc_events", default=None,
                    help="ESM-C events dir. Default: no-refine (sorted glob [0]). "
                         "Pass the refine dir for the matched-option comparison.")
    ap.add_argument("--out_suffix", default="",
                    help="Suffix for output files, e.g. '_refine'.")
    ap.add_argument("--esmc_label", default="PEINT ESM-C")
    args = ap.parse_args()
    esmc_events = args.esmc_events or ESMC_EVENTS
    sfx = args.out_suffix
    lbl = args.esmc_label

    families = json.load(open(FAM_JSON))["families"]
    protevo_caching.set_cache_dir(f"{R2}/simulations/historian_compare/_cache")
    protevo_caching.set_dir_levels(3)

    # ESM2 + real counts from rev1 reconstructions (paired with rev1's subtree trees).
    esm2 = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=f"{R1}/peint_msa_historian",
        tree_dir=f"{R1}/real_subtree_historian",
        families=families, num_processes=16,
    )["output_events_dir"]
    real = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=f"{R1}/real_msa_historian",
        tree_dir=f"{R1}/real_subtree_historian",
        families=families, num_processes=16,
    )["output_events_dir"]

    rows = []
    for f in families:
        e = indel_events(esmc_events, f)
        m = indel_events(esm2, f)
        r = indel_events(real, f)
        if None not in (e, m, r):
            rows.append((f, e, m, r))
    df = pd.DataFrame(rows, columns=["family", "esmc", "esm2", "real"])
    os.makedirs(FIG_OUT, exist_ok=True)
    df.to_csv(f"{FIG_OUT}/historian_indel_esmc_vs_rev1{sfx}.csv", index=False)
    print(f"{len(df)} families with counts in all three sources")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    s1 = _annotate(axes[0], df["real"], df["esmc"], "Real indel events", lbl)
    s2 = _annotate(axes[1], df["real"], df["esm2"], "Real indel events", "PEINT ESM2 (rev1)")
    s3 = _annotate(axes[2], df["esm2"], df["esmc"], "PEINT ESM2 (rev1)", lbl)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIG_OUT}/historian_indel_esmc_vs_rev1{sfx}.{ext}", dpi=150)

    # ---- Indel length CDF (reuses figure3_indels.py's CDF construction) ----
    def _indel_lengths(events_dir):
        out = []
        for f in families:
            p = os.path.join(events_dir, f"{f}.txt")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p, usecols=["event_type", "length"])
            out.extend(d.loc[d["event_type"].isin(["insertion", "deletion"]), "length"].values)
        return np.sort(np.asarray(out, dtype=float))

    series = [("Real (inferred)", _indel_lengths(real), "purple"),
              (lbl, _indel_lengths(esmc_events), "green"),
              ("PEINT ESM2 (rev1)", _indel_lengths(esm2), "orange")]
    figc, axc = plt.subplots(1, 1, figsize=(4, 3.2))
    for name, L, c in series:
        cdf = np.arange(1, len(L) + 1) / len(L)
        axc.plot(L, cdf, label=f"{name} (median {np.median(L):.0f})", color=c, linewidth=1.5)
    axc.set_xlabel("Indel length (residues)")
    axc.set_ylabel("Cumulative density")
    axc.set_title("Indel length CDF")
    axc.set_xlim(0, 50); axc.set_ylim(0, 1.02)
    axc.legend(loc="lower right", fontsize=8, frameon=False)
    axc.grid(True, which="both", linestyle="--", linewidth=0.3)
    figc.tight_layout()
    for ext in ("png", "pdf"):
        figc.savefig(f"{FIG_OUT}/historian_indel_length_cdf{sfx}.{ext}", dpi=150)

    print("\n=== indel length (residues): median / P90 / P99 / max ===")
    for name, L, _ in series:
        print(f"{name:20s} n={len(L):8d}  median={np.median(L):.0f}  "
              f"P90={np.percentile(L,90):.0f}  P99={np.percentile(L,99):.0f}  max={L.max():.0f}")

    print("\n=== Indel-event correlations (per family, n=%d) ===" % len(df))
    print(f"real vs ESM-C : Pearson {s1[0]:.3f}  Spearman {s1[1]:.3f}")
    print(f"real vs ESM2  : Pearson {s2[0]:.3f}  Spearman {s2[1]:.3f}")
    print(f"ESM2 vs ESM-C : Pearson {s3[0]:.3f}  Spearman {s3[1]:.3f}")
    print("\nmedian indel events/family:  real %.0f  ESM-C %.0f  ESM2 %.0f"
          % (df["real"].median(), df["esmc"].median(), df["esm2"].median()))
    print(f"\nsaved: {FIG_OUT}/historian_indel_esmc_vs_rev1.{{png,pdf,csv}}")


if __name__ == "__main__":
    main()
