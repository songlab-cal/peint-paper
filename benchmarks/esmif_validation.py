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
from collections import Counter
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
MODEL_ORDER = ["WAG", "LG", "LG4X", "LG+C60", "LG+S256",
               "PEINT (ESM2)", "PEINT (ESM-C)", "Real"]
# display label -> directory key used by model_msa_dirs()/omegafold. Both PEINT models use
# the "PEINT (Progressive)" subdir key; they are disambiguated by MODEL_RESULTS_DIR / MSA dirs
# (ESM2 = rev1, ESM-C = rev2), so ESM-C is a first-class model, not a merged one-off.
DIRKEY = {
    "Real": "Real", "PEINT (ESM2)": "PEINT (Progressive)",
    "PEINT (ESM-C)": "PEINT (Progressive)", "WAG": "WAG", "LG": "LG",
    "LG4X": "LG4X", "LG+C60": "LG+C60", "LG+S256": "LG+S256",
}
# Rev1 models read cfg.RESULTS_DIR; ESM-C reads the rev2 ESM-C results (its own alignment
# frame for A1, its own OmegaFold structures for A2). A1 threads each model through ITS
# revision's seq1 frame, which is revision-independent once threaded -> comparable.
_R2 = os.environ.get(
    "PEINT_PAPER_ESMC_RESULTS_DIR",
    str(cfg.RESULTS_R2_DIR),
)
MODEL_RESULTS_DIR = {m: str(cfg.RESULTS_DIR) for m in MODEL_ORDER}
MODEL_RESULTS_DIR["PEINT (ESM-C)"] = _R2
# Old CSVs (and the rev1 run) label the ESM2 PEINT simply "PEINT"; map it forward.
LEGACY_MODEL_RENAME = {"PEINT": "PEINT (ESM2)"}
# x-axis labels matching the JSD figure; "Real" here is the eval-subtree (non-root split) real data.
LABELS = {"Real": "Real (eval subtree)"}
LEAF = re.compile(r"^seq\d+$")
LEAF_CAP = 60       # non-root leaves per family (evenly spaced); logged, not silent
STRUCT_CAP = 25     # OmegaFold structures per family/model for Approach 2
OUT_DIR = Path(cfg.FIGURES_DIR) / "esmif"


def model_colors():
    """Canonical colors from paper.model_style (shared with the JSD / ECDF / PCP figures)."""
    from paper.model_style import model_colors as shared
    return shared()


def _a1_dirs():
    """(msa_dir, real_dir) per model for Approach 1. Rev1 models come from the shared
    model_msa_dirs(); ESM-C overrides to its rev2 mafft-add frame (its own real seq1)."""
    mm = model_msa_dirs()
    msa = {m: mm[DIRKEY[m]] for m in MODEL_ORDER if DIRKEY[m] in mm}
    real = {m: mm["Real"] for m in MODEL_ORDER}
    esmc_mafft = os.path.join(_R2, "mafft_add")
    msa["PEINT (ESM-C)"] = os.path.join(esmc_mafft, "peint_progressive_dir")
    real["PEINT (ESM-C)"] = os.path.join(esmc_mafft, "old_sequences")
    return msa, real


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
def run_approach1(families, existing=None, existing_leaf=None):
    """Return (family_df, perleaf_df). One row per (family, model[, leaf]).

    ``existing``/``existing_leaf`` (from a prior run's CSVs) are reused verbatim; only the
    (family, model) pairs not already present are computed. Each model is threaded through
    ITS OWN revision's seq1 frame (rev1 vs ESM-C rev2), so ESM-C is a first-class model.
    """
    import esm.inverse_folding as invf  # esmif import already ran the biotite shim
    msa_dirs, real_dirs = _a1_dirs()
    fam_rows = existing.to_dict("records") if existing is not None and len(existing) else []
    leaf_rows = existing_leaf.to_dict("records") if existing_leaf is not None and len(existing_leaf) else []
    # A pair counts as done only if BOTH its family-level row and its per-leaf rows exist.
    # Keying on the family frame alone let a truncated per-leaf table look complete, so the
    # divergence-controlled panel silently kept plotting one leaf per model.
    done_fam = {(r["family"], r["model"]) for r in fam_rows}
    # Count leaf rows per pair and compare against the n_leaves the family row recorded, so a
    # pair that was truncated to a single leaf is not mistaken for a complete one.
    leaf_counts = Counter((r["family"], r["model"]) for r in leaf_rows)
    expected = {(r["family"], r["model"]): r.get("n_leaves") for r in fam_rows}

    def _complete(key):
        want = expected.get(key)
        want = int(want) if want == want and want else 1   # NaN-safe
        return leaf_counts.get(key, 0) >= want

    done = {k for k in done_fam if _complete(k)}
    stale = done_fam - done
    if stale:
        print(f"  [A1] {len(stale)} (family, model) pairs have incomplete per-leaf rows; recomputing")
        # Drop the stale rows from BOTH frames. Leaving the partial leaf rows in place would
        # duplicate whichever leaves the recompute reproduces, double-weighting them.
        fam_rows = [r for r in fam_rows if (r["family"], r["model"]) not in stale]
        leaf_rows = [r for r in leaf_rows if (r["family"], r["model"]) not in stale]
    todo = [(f, m) for f in families for m in MODEL_ORDER if (f, m) not in done]
    if not todo:
        print(f"  [A1] all ({len(families)}x{len(MODEL_ORDER)}) pairs reused; nothing to compute")
        return pd.DataFrame(fam_rows), pd.DataFrame(leaf_rows)
    print(f"  [A1] computing {len(todo)} (family, model) pairs; reusing {len(done)}")
    esmif.load_model()
    n_ok = 0
    for i, fam in enumerate(families):
        pend = [m for m in MODEL_ORDER if (fam, m) not in done]
        if not pend or not _has_inputs(fam):
            continue
        try:
            leaves = nonroot_leaves(fam)
            coords, gt_seq = invf.util.load_coords(
                os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"), fam.split("_")[-1]
            )
        except Exception as e:  # pragma: no cover - skip unusable families, keep going
            print(f"  [A1] skip {fam}: {e}")
            continue
        frame_cache = {}  # real_dir -> (keep, ncol)
        for disp in pend:
            try:
                real_dir = real_dirs[disp]
                if real_dir not in frame_cache:
                    s1 = read_msa(os.path.join(real_dir, f"{fam}.txt"))["seq1"]
                    frame_cache[real_dir] = (esmif.reference_columns(s1, gap_character), len(s1))
                keep, ncol = frame_cache[real_dir]
                path = os.path.join(msa_dirs[disp], f"{fam}.txt")
                if not os.path.exists(path):
                    continue
                msa = read_msa(path)
                if len(next(iter(msa.values()))) != ncol:  # not in that model's reference frame
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
def a2_model_coverage(df):
    """(scored, missing) model arms for Approach 2, in MODEL_ORDER."""
    have = set(df["model"]) if df is not None and len(df) else set()
    return ([m for m in MODEL_ORDER if m in have],
            [m for m in MODEL_ORDER if m not in have])


def warn_a2_incomplete(df, skipped=None, skipped_root=None):
    """Say loudly which model arms Approach 2 could not score, and why.

    A2 reads each model's OmegaFold structures from ITS revision's results dir. The seven
    rev1 arms live in the ``r1_omegafold`` deposit role, which ``MANIFEST.toml`` marks
    ``panels = []`` and ``REPRODUCING.md`` tells readers they may skip -- so the common
    failure is that EVERY rev1 arm yields no structures and the panel quietly collapses to
    the ESM-C arm alone. Left unreported that looks like a finished eight-model comparison.
    """
    scored, missing = a2_model_coverage(df)
    if not missing:
        return scored, missing
    bar = "  " + "=" * 74
    print("\n" + bar)
    print(f"  WARNING: Approach 2 scored {len(scored)} of {len(MODEL_ORDER)} model arms -- "
          f"ED5e is INCOMPLETE.")
    print(f"    scored : {', '.join(scored) or '(none)'}")
    print(f"    SKIPPED: {', '.join(missing)}")
    for m in missing:
        n = (skipped or {}).get(m)
        root = (skipped_root or {}).get(m)
        if n:
            print(f"      {m:<16} no structures for {n} families under {root}/<family>/...")
    print("    The rev1 arms come from the `r1_omegafold` deposit role. MANIFEST.toml marks it")
    print("    `panels = []`, which is wrong: this panel needs it. Fetch it with")
    print("      python scripts/fetch_local_data.py --tier full")
    print("    (or unpack r1_omegafold.tar.zst into local_data/r1/), then re-run.")
    print(bar + "\n")
    return scored, missing


def run_approach2(families, existing=None):
    """Self-consistency recovery per (family, model). ``existing`` rows are reused; only the
    missing (family, model) pairs are computed. Each model's OmegaFold structures come from
    ITS revision's results dir (rev1 vs ESM-C rev2), so ESM-C is a first-class model."""
    rows = existing.to_dict("records") if existing is not None and len(existing) else []
    done = {(r["family"], r["model"]) for r in rows}
    todo = [(f, m) for f in families for m in MODEL_ORDER if (f, m) not in done]
    if not todo:
        print(f"  [A2] all pairs reused; nothing to compute")
        df = pd.DataFrame(rows)
        warn_a2_incomplete(df)
        return df
    print(f"  [A2] computing {len(todo)} (family, model) pairs; reusing {len(done)}")
    skipped, skipped_root = {}, {}
    esmif.load_model()
    for i, fam in enumerate(families):
        for disp in MODEL_ORDER:
            if (fam, disp) in done:
                continue
            sdir = Path(MODEL_RESULTS_DIR[disp]) / "omegafold" / fam / DIRKEY[disp] / "structures"
            pdbs = sorted(glob.glob(str(sdir / "seq*.pdb")))
            if not pdbs:
                # Not a per-family quirk when it happens to a whole arm: the rev1 models read
                # the r1_omegafold role, which the deposit marks `panels = []`. Record the
                # skip so it can be reported instead of silently shrinking the figure.
                skipped[disp] = skipped.get(disp, 0) + 1
                skipped_root.setdefault(disp, str(sdir.parents[1]))
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
    df = pd.DataFrame(rows)
    warn_a2_incomplete(df, skipped, skipped_root)
    return df


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


def write_rebuttal(a1, a2, strat, a2_missing=()):
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
        if a2_missing:
            lines.append(f"> **INCOMPLETE — {len(MODEL_ORDER) - len(a2_missing)} of "
                         f"{len(MODEL_ORDER)} model arms.** Not scored: "
                         f"{', '.join(a2_missing)}. Their OmegaFold structures come from the "
                         f"`r1_omegafold` deposit role, which is absent here. The values below "
                         f"are NOT a model comparison and must not be read as one.\n")
        for m, v in _median_table(a2, "recovery"):
            lines.append(f"- {m:<10} {v:.3f}")
    lines.append("\n## Novel vs seen (generalization)\n")
    lines.append("See `esmif_gt_likelihood_stratified_*` " + ("(written)." if strat else "(skipped: annotations unavailable)."))
    lines.append("\n_Leaves: non-root split, matched across models, capped at "
                 f"{LEAF_CAP}/family. Structures: up to {STRUCT_CAP}/family/model._\n")
    (OUT_DIR / "REBUTTAL_esmif.md").write_text("\n".join(lines))


def _load_prior(path, rename=None, drop_models=()):
    """Load a prior ESM-IF CSV, optionally relabeling model names and dropping models.

    Falls back to the shipped copy under FIGURE_DATA_DIR. These three tables are the only
    thing standing between the ESM-IF panels and a two-hour GPU run against ground-truth
    structures that the deposit does not redistribute, so a fresh checkout has to be able to
    find them without being told. Same preference order as figure3_pcp's _find_table.
    """
    path = Path(path)
    if not path.exists():
        shipped = Path(cfg.FIGURE_DATA_DIR) / "esmif" / path.name
        if not shipped.exists():
            return pd.DataFrame()
        path = shipped
    d = pd.read_csv(path)
    if rename:
        d["model"] = d["model"].replace(rename)
    if drop_models:
        d = d[~d["model"].isin(list(drop_models))].copy()
    return d


def _concat(frames, keys=("family", "model")):
    """Concatenate prior frames, keeping the first row per key.

    ``keys`` matters: the family-level table has one row per (family, model), but the
    per-leaf table has ~60. Deduplicating the per-leaf table on (family, model) collapses
    it to a single leaf per model and silently throws away 59/60 of the data, which is what
    previously truncated esmif_gt_likelihood_perleaf.csv — pass the leaf key for that one.
    """
    frames = [f for f in frames if f is not None and len(f)]
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates(list(keys), keep="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N eval families")
    ap.add_argument("--approach", choices=["1", "2", "both"], default="both")
    ap.add_argument("--no-reuse", action="store_true",
                    help="recompute every model from scratch (default reuses prior CSVs, "
                         "computing only the missing ESM-C columns)")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    families = eval_families()
    if args.limit:
        families = families[: args.limit]
    print(f"ESM-IF validation over {len(families)} families; models={MODEL_ORDER}")
    print(f"output -> {OUT_DIR}")

    # Reuse prior results: rev1 CSVs supply the 7 established models (PEINT->"PEINT (ESM2)");
    # the wrapper's ESM-C A1 CSV supplies PEINT (ESM-C). Only ESM-C A2 is left to compute.
    r1, rc = {"PEINT": "PEINT (ESM2)"}, {"PEINT": "PEINT (ESM-C)"}
    FIG = Path(cfg.FIGURES_DIR)
    a1x = a1lx = a2x = None
    if not args.no_reuse:
        a1x = _concat([
            _load_prior(OUT_DIR / "esmif_gt_likelihood.csv", r1),
            _load_prior(FIG / "esmif_a1_esmc_rev2.csv", rc, drop_models=["Real"]),
        ])
        a1lx = _concat([
            _load_prior(OUT_DIR / "esmif_gt_likelihood_perleaf.csv", r1),
            _load_prior(FIG / "esmif_a1_esm2_rev1_perleaf.csv", r1),
            _load_prior(FIG / "esmif_a1_esmc_rev2_perleaf.csv", rc, drop_models=["Real"]),
        ], keys=("family", "model", "leaf"))
        a2x = _load_prior(OUT_DIR / "esmif_selfconsistency.csv", r1)

    a1 = a2 = None
    if args.approach in ("1", "both"):
        print("[Approach 1] GT-structure likelihood ...")
        a1, a1_leaf = run_approach1(families, existing=a1x, existing_leaf=a1lx)
        a1.to_csv(OUT_DIR / "esmif_gt_likelihood.csv", index=False)
        a1_leaf.to_csv(OUT_DIR / "esmif_gt_likelihood_perleaf.csv", index=False)
        _boxplot(a1, "ll", "ESM-IF log-likelihood", "Likelihood on Seq1 Structure",
                 "esmif_approach1_gt_likelihood")
        _divergence_plot(a1_leaf, "esmif_approach1_divergence_controlled")
        print("  medians:", _median_table(a1, "ll"))

    a2_missing = ()
    if args.approach in ("2", "both"):
        print("[Approach 2] self-consistency ...")
        a2 = run_approach2(families, existing=a2x)
        if len(a2):
            a2.to_csv(OUT_DIR / "esmif_selfconsistency.csv", index=False)
            a2_scored, a2_missing = a2_model_coverage(a2)
            a2_title = "Self-consistency on OmegaFold structures"
            if a2_missing:
                a2_title += (f"\nINCOMPLETE: {len(a2_scored)} of {len(MODEL_ORDER)} model "
                             f"arms (missing r1_omegafold)")
            _boxplot(a2, "recovery", "ESM-IF sequence recovery", a2_title,
                     "esmif_approach2_selfconsistency")
            print("  medians:", _median_table(a2, "recovery"))

    strat = False
    if a1 is not None and len(a1):
        try:
            from paper import generalization as gen
            gen.ensure_annotations()
            gen.report_stratified(
                a1, "ll", "esmif_gt_likelihood_stratified",
                focus_models=[m for m in ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"]
                              if m in set(a1["model"])],
                value_label="ESM-IF LL (GT structure)", out_dir=str(OUT_DIR),
            )
            strat = True
        except Exception as e:  # pragma: no cover
            print(f"  [stratify] skipped: {e}")

    write_rebuttal(a1 if a1 is not None else pd.DataFrame(),
                   a2, strat, a2_missing=a2_missing)
    print(f"Done. Wrote CSVs + figures + REBUTTAL_esmif.md to {OUT_DIR}")


if __name__ == "__main__":
    main()
