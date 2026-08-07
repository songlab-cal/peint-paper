"""ESM-IF inverse-folding validation of the structural metrics (reviewer response).

A reviewer asked us to corroborate the structural metrics with an inverse-folding model. We
score, under ``esm_if1_gvp4_t16_142M_UR50``, two readouts:

* **Approach 1 — GT-structure likelihood.** Thread each model's simulated leaf onto the
  experimental (ground-truth) PDB structure and score ``p(leaf | structure)``. Leaves are the
  **non-root split** of the tree (the half NOT containing the simulation root; ``seq1`` is only
  the alignment anchor and is excluded), and the **same leaf set is used for every model** so the
  comparison is balanced position-for-position. Threading uses the true reference frame
  (``old_sequences/seq1``); see ``paper.esmif`` for the seq1-labelling gotcha.
* **Approach 2 — self-consistency.** Score each OmegaFold-predicted structure against its own
  sequence (template-free companion; reuses structures already on disk).

INTERPRETATION (confounds established with the user — they decide which comparisons are valid):

* **Memorization.** ESM-IF trained on UniRef50 structures ⇒ the empirical ("Real") leaves are
  in-distribution and score inflated; PEINT / classical-simulator leaves are synthetic and OOD.
  So the **memorization-free axis is PEINT vs the classical simulators** (WAG/LG/LG4X/LG+C60/
  LG+S256), and the PEINT-vs-Real gap is an **upper bound** on the true gap (bias runs against
  PEINT), making "PEINT tracks Real" conservative.
* **Divergence.** Approach-1 LL on a fixed reference also tracks identity-to-``seq1``, so we report
  a divergence-controlled view (LL vs identity), not raw LL alone.
* **Native is not maximally designable** — we score the actual sequence's likelihood, not
  design→refold scTM "designability" (native backbones are only ~0.5–0.7 recoverable; higher is
  not better).

Run::

    python -m benchmarks.esmif_validation                 # both approaches, all families
    python -m benchmarks.esmif_validation --limit 8       # quick smoke test
    python -m benchmarks.esmif_validation --approach 1    # only Approach 1
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from paper import esmif
from paper.generalization import eval_families
from paper.splits import generate_tree_split, SPLIT_A, SPLIT_B
from protevo.utils import read_msa, gap_character
from figures.figure3_conservation import model_msa_dirs

# --- models: same order/labels/colors as the JSD figure (figure3_conservation.BOXPLOT_MODELS /
#     BOXPLOT_LABELS): classical simulators first (increasing complexity), then PEINT, then Real.
#     PEINT (Progressive) is relabelled "PEINT"; PEINT (Single Shot) is dropped. ---
MODEL_ORDER = ["WAG", "LG", "LG4X", "LG+C60", "LG+S256", "PEINT", "Real"]
# display label -> directory key used by model_msa_dirs()/omegafold
DIRKEY = {
    "Real": "Real", "PEINT": "PEINT (Progressive)", "WAG": "WAG", "LG": "LG",
    "LG4X": "LG4X", "LG+C60": "LG+C60", "LG+S256": "LG+S256",
}
# x-axis labels matching the JSD figure; "Real" here is the eval-subtree (non-root split) real data.
LABELS = {"Real": "Real (eval subtree)"}
LEAF = re.compile(r"^seq\d+$")
LEAF_CAP = 60       # non-root leaves per family (evenly spaced); logged, not silent
STRUCT_CAP = 25     # OmegaFold structures per family/model for Approach 2
OUT_DIR = Path(cfg.FIGURES_DIR) / "esmif"


def model_colors():
    """Same seaborn-default palette as figure3_pcp_mutation_counts, extended for LG4X/LG+C60."""
    import seaborn as sns
    p = sns.color_palette()
    return {"Real": p[4], "PEINT": p[2], "WAG": p[0], "LG": p[1],
            "LG4X": p[3], "LG+C60": p[5], "LG+S256": p[6]}


# ======================================================================================
# leaf selection
# ======================================================================================
def _has_inputs(fam: str) -> bool:
    return all(os.path.exists(p) for p in (
        os.path.join(str(cfg.TREE_DIR), f"{fam}.txt"),
        os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"),
        os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt"),
    ))


def nonroot_leaves(fam: str) -> list:
    """Leaf names on the split NOT containing the simulation root; ``seq1`` excluded, capped.

    The root sequence (from ``root_sequences/``) can be on either side, so we find its side per
    family. Even spacing keeps the family-level mean stable at ``LEAF_CAP`` without bias.
    """
    root_name = list(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")))[0]
    ts = generate_tree_split(str(cfg.TREE_DIR), fam)
    a, b = set(ts[SPLIT_A]), set(ts[SPLIT_B])
    nonroot = b if root_name in a else a
    leaves = sorted(n for n in nonroot if LEAF.match(n) and n != "seq1")
    if len(leaves) > LEAF_CAP:
        idx = np.unique(np.linspace(0, len(leaves) - 1, LEAF_CAP).astype(int))
        leaves = [leaves[i] for i in idx]
    return leaves


def _thread_and_identity(leaf_gapped: str, keep, gt_seq: str):
    """Thread a leaf onto the GT frame and also return its identity to seq1 (divergence covariate)."""
    target, mask = esmif.thread_via_reference_frame(leaf_gapped, keep, gap_character)
    matches = sum(1 for r in np.where(mask)[0] if target[r] == gt_seq[r])
    identity = matches / max(int(mask.sum()), 1)
    return target, mask, identity


# ======================================================================================
# Approach 1 — GT-structure likelihood
# ======================================================================================
def run_approach1(families):
    """Return (family_df, perleaf_df). One row per (family, model[, leaf])."""
    import esm.inverse_folding as invf  # esmif import already ran the biotite shim
    esmif.load_model()
    mm = model_msa_dirs()
    fam_rows, leaf_rows = [], []
    n_ok = 0
    for i, fam in enumerate(families):
        if not _has_inputs(fam):
            continue
        try:
            leaves = nonroot_leaves(fam)
            old = read_msa(os.path.join(mm["Real"], f"{fam}.txt"))
            s1 = old["seq1"]
            keep = esmif.reference_columns(s1, gap_character)
            coords, gt_seq = invf.util.load_coords(
                os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"), fam.split("_")[-1]
            )
        except Exception as e:  # pragma: no cover - skip unusable families, keep going
            print(f"  [A1] skip {fam}: {e}")
            continue
        ncol = len(s1)
        for disp in MODEL_ORDER:
            try:
                path = os.path.join(mm[DIRKEY[disp]], f"{fam}.txt")
                if not os.path.exists(path):
                    continue
                msa = read_msa(path)
                if len(next(iter(msa.values()))) != ncol:  # not in the shared reference frame
                    continue
                seqs, masks, idents, names = [], [], [], []
                for n in leaves:
                    if n not in msa:
                        continue
                    t, m, idn = _thread_and_identity(msa[n], keep, gt_seq)
                    seqs.append(t); masks.append(m); idents.append(idn); names.append(n)
                if not seqs:
                    continue
                scored = esmif.score_batch(coords, seqs, masks)
            except Exception as e:  # pragma: no cover - skip (e.g. transient OOM) and keep going
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"  [A1] skip {fam}/{disp}: {e}")
                continue
            lls = np.array([d["ll"] for d in scored], dtype=float)
            for n, idn, d in zip(names, idents, scored):
                leaf_rows.append({"family": fam, "model": disp, "leaf": n,
                                  "ll": d["ll"], "identity": idn})
            fam_rows.append({"family": fam, "model": disp, "ll": float(np.nanmean(lls)),
                             "identity": float(np.mean(idents)), "n_leaves": len(seqs)})
        n_ok += 1
        if (i + 1) % 25 == 0:
            print(f"  [A1] {i + 1}/{len(families)} families ({n_ok} scored)")
    return pd.DataFrame(fam_rows), pd.DataFrame(leaf_rows)


# ======================================================================================
# Approach 2 — self-consistency on OmegaFold structures
# ======================================================================================
def run_approach2(families):
    esmif.load_model()
    rows = []
    for i, fam in enumerate(families):
        for disp in MODEL_ORDER:
            sdir = Path(cfg.RESULTS_DIR) / "omegafold" / fam / DIRKEY[disp] / "structures"
            pdbs = sorted(glob.glob(str(sdir / "seq*.pdb")))
            if not pdbs:
                continue
            if len(pdbs) > STRUCT_CAP:
                idx = np.unique(np.linspace(0, len(pdbs) - 1, STRUCT_CAP).astype(int))
                pdbs = [pdbs[j] for j in idx]
            lls, recs = [], []
            for pdb in pdbs:
                try:
                    r = esmif.score_pdb_self(pdb)
                except Exception:
                    continue
                lls.append(r["ll"]); recs.append(r["recovery"])
            if lls:
                rows.append({"family": fam, "model": disp, "ll": float(np.nanmean(lls)),
                             "recovery": float(np.nanmean(recs)), "n_struct": len(lls)})
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if (i + 1) % 25 == 0:
            print(f"  [A2] {i + 1}/{len(families)} families")
    return pd.DataFrame(rows)


# ======================================================================================
# figures + writeup
# ======================================================================================
def _boxplot(df, value_col, ylabel, title, out_stem):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style
    _set_publication_style()
    order = [m for m in MODEL_ORDER if m in set(df["model"])]
    colors = model_colors()
    fig, ax = plt.subplots(figsize=(0.85 * len(order) + 1, 3.2))
    sns.boxplot(data=df, x="model", y=value_col, order=order,
                palette={m: colors[m] for m in order}, showfliers=False,
                width=0.65, linewidth=0.6, ax=ax)
    sns.stripplot(data=df, x="model", y=value_col, order=order, color="0.25",
                  size=1.3, alpha=0.25, ax=ax)
    ax.set_xlabel(""); ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=10)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([LABELS.get(m, m) for m in order], rotation=30, ha="right", fontsize=9)
    sns.despine(ax=ax)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _divergence_plot(leaf_df, out_stem):
    """LL vs identity-to-seq1 per model — shows PEINT is not merely winning by staying conservative."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style
    _set_publication_style()
    colors = model_colors()
    bins = np.linspace(0.0, 0.6, 7)
    mid = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    for m in [x for x in MODEL_ORDER if x in set(leaf_df["model"])]:
        d = leaf_df[leaf_df["model"] == m]
        idx = np.digitize(d["identity"].values, bins) - 1
        ys = [np.nanmean(d["ll"].values[idx == k]) if (idx == k).any() else np.nan
              for k in range(len(mid))]
        ax.plot(mid, ys, "-o", ms=3, lw=1, color=colors[m], label=LABELS.get(m, m))
    ax.set_xlabel("identity to seq1", fontsize=10)
    ax.set_ylabel("ESM-IF log-likelihood", fontsize=10)
    ax.set_title("Divergence-controlled likelihood", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    sns.despine(ax=ax)
    fig.savefig(OUT_DIR / f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{out_stem}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def _median_table(df, value_col):
    med = df.groupby("model")[value_col].median()
    return [(m, float(med[m])) for m in MODEL_ORDER if m in med.index]


def write_rebuttal(a1, a2, strat):
    lines = ["# ESM-IF inverse-folding validation\n",
             "Inverse-folding corroboration of the structural metrics, per reviewer request. ",
             "Two confounds shape interpretation: ESM-IF's training on natural (UniRef50) sequences ",
             "inflates *Real* (so the memorization-free comparison is PEINT vs the classical ",
             "simulators, and PEINT-vs-Real is a conservative upper bound), and Approach-1 LL on a ",
             "fixed reference also tracks divergence from `seq1` (hence the divergence-controlled view).\n",
             "\n## Approach 1 — likelihood on the ground-truth structure (median LL per model)\n"]
    for m, v in _median_table(a1, "ll"):
        tag = "  <- PEINT" if m == "PEINT" else ("  <- Real (memorization-inflated ceiling)" if m == "Real" else "")
        lines.append(f"- {m:<10} {v:+.3f}{tag}")
    if a2 is not None and len(a2):
        lines.append("\n## Approach 2 — self-consistency on OmegaFold structures (median recovery per model)\n")
        for m, v in _median_table(a2, "recovery"):
            lines.append(f"- {m:<10} {v:.3f}")
    lines.append("\n## Novel vs seen (generalization)\n")
    lines.append("See `esmif_gt_likelihood_stratified_*` " + ("(written)." if strat else "(skipped: annotations unavailable)."))
    lines.append("\n_Leaves: non-root split, matched across models, capped at "
                 f"{LEAF_CAP}/family. Structures: up to {STRUCT_CAP}/family/model._\n")
    (OUT_DIR / "REBUTTAL_esmif.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N eval families")
    ap.add_argument("--approach", choices=["1", "2", "both"], default="both")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families = eval_families()
    if args.limit:
        families = families[: args.limit]
    print(f"ESM-IF validation over {len(families)} families; models={MODEL_ORDER}")
    print(f"output -> {OUT_DIR}")

    a1 = a2 = None
    if args.approach in ("1", "both"):
        print("[Approach 1] GT-structure likelihood ...")
        a1, a1_leaf = run_approach1(families)
        a1.to_csv(OUT_DIR / "esmif_gt_likelihood.csv", index=False)
        a1_leaf.to_csv(OUT_DIR / "esmif_gt_likelihood_perleaf.csv", index=False)
        _boxplot(a1, "ll", "ESM-IF log-likelihood", "Likelihood on the ground-truth structure",
                 "esmif_approach1_gt_likelihood")
        _divergence_plot(a1_leaf, "esmif_approach1_divergence_controlled")
        print("  medians:", _median_table(a1, "ll"))

    if args.approach in ("2", "both"):
        print("[Approach 2] self-consistency ...")
        a2 = run_approach2(families)
        if len(a2):
            a2.to_csv(OUT_DIR / "esmif_selfconsistency.csv", index=False)
            _boxplot(a2, "recovery", "ESM-IF sequence recovery",
                     "Self-consistency on OmegaFold structures",
                     "esmif_approach2_selfconsistency")
            print("  medians:", _median_table(a2, "recovery"))

    strat = False
    if a1 is not None and len(a1):
        try:
            from paper import generalization as gen
            gen.ensure_annotations()
            gen.report_stratified(
                a1, "ll", "esmif_gt_likelihood_stratified",
                focus_models=[m for m in ["LG+S256", "PEINT", "Real"] if m in set(a1["model"])],
                value_label="ESM-IF LL (GT structure)", out_dir=str(OUT_DIR),
            )
            strat = True
        except Exception as e:  # pragma: no cover
            print(f"  [stratify] skipped: {e}")

    write_rebuttal(a1 if a1 is not None else pd.DataFrame(),
                   a2, strat)
    print(f"Done. Wrote CSVs + figures + REBUTTAL_esmif.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
