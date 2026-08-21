"""Indel events vs median sequence length — 3 panels (Real, PEINT ESM2, PEINT ESM-C).

Companion panel to ``historian_compare_esmc_vs_rev1.py``. Each panel scatters one model's
per-family total indel *events* (#insertions + #deletions from the Historian reconstruction,
same counter as that script) against the median ungapped length of that SAME model's
reconstruction leaves (``seq*`` nodes; historian internal nodes ``internal-*`` are excluded).
Pearson/Spearman annotated per panel; each panel uses its own model's lengths (real→real
lengths, PEINT→PEINT lengths, ESM-C→ESM-C lengths).

Sources (all overridable via CLI) match historian_compare_esmc_vs_rev1.py:
  * event tables: real/ESM2 recomputed via ``get_all_evolutionary_counts_from_historian_output``
    (cached), ESM-C from the rev2 historian job's cached events dir;
  * reconstruction sequences: rev1 ``real_msa_historian`` / ``peint_msa_historian`` and the rev2
    ESM-C remove-dummy reconstruction (the same one figure3_pcp_mutation_counts uses).

Run from the repo root (env with protevo, e.g. protevo-env)::

    python -m benchmarks.historian_indel_vs_length
"""

import argparse
import glob
import json
import os

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
from protevo.utils import read_msa
from paper.historian import get_all_evolutionary_counts_from_historian_output
from paper.model_style import model_colors

R1 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision1/simulations"
R2 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc"
FAM_JSON = "/scratch/users/akoehl/protein-evolution/local_data/final_sim_held_out_family.json"
FIG_OUT = "/scratch/users/akoehl/peint-paper/figures/output"

# ESM-C cached event tables (rev2 historian job), sorted-glob [0] as in historian_compare.
ESMC_EVENTS_DEFAULT = sorted(glob.glob(
    f"{R2}/simulations/historian_progressive/_cache/"
    "get_all_evolutionary_counts_from_historian_output/*/*/*/*/output_events_dir"))[0]
# ESM-C remove-dummy reconstruction sequences (== figure3_pcp_mutation_counts ESMC_RECON).
ESMC_RECON_DEFAULT = (
    f"{R2}/simulations/historian_progressive/_cache/remove_dummy_nodes_from_historian_output/"
    "0/e/2/7848affe284d9acd102977c0d0f2f3dc441092610c895020ca28319a8da332aaa86e43d5d31b8b064e99a9d97e163d28c7b1f7997d8fc7dc2924a88db175a/"
    "output_sequences_dir")


def indel_events(events_dir, fam):
    """Total indel events (#insertions + #deletions) for a family, or None if missing."""
    p = os.path.join(events_dir, f"{fam}.txt")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    vc = df["event_type"].value_counts()
    return int(vc.get("insertion", 0) + vc.get("deletion", 0))


def median_leaf_length(recon_dir, fam):
    """Median ungapped length of a family's reconstruction LEAVES (seq* nodes), or None."""
    p = os.path.join(recon_dir, f"{fam}.txt")
    if not os.path.exists(p):
        return None
    msa = read_msa(p)
    lengths = [len(seq.replace("-", "")) for name, seq in msa.items() if name.startswith("seq")]
    if not lengths:
        return None
    return float(np.median(lengths))


def _panel(ax, x, y, xlabel, title, color):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    pr = pearsonr(x, y)[0]
    sr = spearmanr(x, y)[0]
    ax.scatter(x, y, s=14, alpha=0.6, color=color, edgecolor="black", linewidth=0.2)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color=color, linewidth=1.0)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("Indel events", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.text(0.05, 0.95, f"Pearson {pr:.2f}\nSpearman {sr:.2f}\nn={len(x)}",
            transform=ax.transAxes, va="top", fontsize=8)
    ax.tick_params(width=0.5, length=2, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    return pr, sr


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real-recon", default=f"{R1}/real_msa_historian")
    ap.add_argument("--esm2-recon", default=f"{R1}/peint_msa_historian")
    ap.add_argument("--esmc-recon", default=ESMC_RECON_DEFAULT)
    ap.add_argument("--tree-dir", default=f"{R1}/real_subtree_historian",
                    help="Eval-subtree trees for the real/ESM2 event recomputation.")
    ap.add_argument("--esmc-events", default=ESMC_EVENTS_DEFAULT)
    ap.add_argument("--families-path", default=FAM_JSON)
    ap.add_argument("--out-dir", default=FIG_OUT)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--num-processes", type=int, default=16)
    args = ap.parse_args()

    families = json.load(open(args.families_path))["families"]
    protevo_caching.set_cache_dir(f"{R2}/simulations/historian_compare/_cache")
    protevo_caching.set_dir_levels(3)

    # Real + ESM2 events from rev1 reconstructions (cached; same call as historian_compare).
    real_events = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=args.real_recon, tree_dir=args.tree_dir,
        families=families, num_processes=args.num_processes,
    )["output_events_dir"]
    esm2_events = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=args.esm2_recon, tree_dir=args.tree_dir,
        families=families, num_processes=args.num_processes,
    )["output_events_dir"]

    # Per model: (events dir, reconstruction dir, display, color).
    mc = model_colors()
    models = [
        ("real", real_events, args.real_recon, "Real (inferred)", mc["Real"]),
        ("esm2", esm2_events, args.esm2_recon, "PEINT (ESM2)", mc["PEINT (ESM2)"]),
        ("esmc", args.esmc_events, args.esmc_recon, "PEINT (ESM-C)", mc["PEINT (ESM-C)"]),
    ]

    # Keep families present (events AND length) for ALL three models, so the panels share a set.
    rows = []
    for fam in families:
        vals = {}
        ok = True
        for key, ev_dir, recon_dir, _, _ in models:
            e = indel_events(ev_dir, fam)
            L = median_leaf_length(recon_dir, fam)
            if e is None or L is None:
                ok = False
                break
            vals[f"{key}_events"] = e
            vals[f"{key}_med_len"] = L
        if ok:
            vals["family"] = fam
            rows.append(vals)
    df = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"historian_indel_vs_length{args.out_suffix}.csv")
    df.to_csv(csv_path, index=False)
    print(f"{len(df)} families with events + median length in all three models")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    stats = {}
    for ax, (key, _, _, display, color) in zip(axes, models):
        stats[key] = _panel(
            ax, df[f"{key}_med_len"], df[f"{key}_events"],
            "Median sequence length (residues)", display, color,
        )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out_dir, f"historian_indel_vs_length{args.out_suffix}.{ext}"),
                    dpi=150, bbox_inches="tight")

    print("\n=== indel events vs median seq length (per family) ===")
    for key, _, _, display, _ in models:
        pr, sr = stats[key]
        print(f"{display:18s} Pearson {pr:.3f}  Spearman {sr:.3f}  "
              f"median events={df[f'{key}_events'].median():.0f}  "
              f"median len={df[f'{key}_med_len'].median():.0f}")
    print(f"\nsaved: {csv_path}")
    print(f"       historian_indel_vs_length{args.out_suffix}.{{png,pdf}}")


if __name__ == "__main__":
    main()
