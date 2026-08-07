"""Standalone evals on the ESM2-MCMC simulated MSAs (reviewer response) — situating PEINT vs.
protein LMs. Kept SEPARATE from the main paper figures: everything lands under
``figures/output/esm_mcmc/eval/`` and no figure's model list is touched.

Reuses the existing metric machinery so ESM-MCMC is scored identically to the other models, on
the SAME families:

* **ESM-IF Approach 1** — GT-structure log-likelihood (``paper.esmif``). The other models'
  per-family LL is pulled straight from ``figures/output/esmif/esmif_gt_likelihood.csv`` (no
  recompute); only ESM-MCMC is scored here, on the same non-root leaf set (``nonroot_leaves``).
* **JSD conservation** — ``paper.jsd.family_jsd``, recomputed for ALL models on these families so
  ESM-MCMC sits in an apples-to-apples column.

Frame placement (the substitution-only shortcut): ESM-MCMC leaves are 1:1 positional with the
root, so each leaf drops into the shared ``old_sequences`` column frame via the root row's gap
pattern — no ``mafft --add``, and exact (the root row ungaps to ``root_seq``). That single frame
MSA feeds both metrics unchanged. (This does not generalise to indel models.)

Run from the repo root::

    python -m benchmarks.esm_mcmc_eval                       # 8 default families
    python -m benchmarks.esm_mcmc_eval --families 1a2t_1_A
    python -m benchmarks.esm_mcmc_eval --seq-dir figures/output/esm_mcmc/hamming/sequences
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from protevo.utils import read_msa, write_msa, gap_character
from paper.splits import generate_tree_split

MCMC_MODEL = "ESM-MCMC"
DEFAULT_SEQ_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "hamming" / "sequences"
OUT_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "eval"
FRAME_DIR = OUT_DIR / "frame_msa"
ESMIF_CSV = Path(cfg.FIGURES_DIR) / "esmif" / "esmif_gt_likelihood.csv"
# Column order for the comparison tables: other models (as in the JSD/esmif figures) then ESM-MCMC.
OTHER_MODELS = ["WAG", "LG", "LG4X", "LG+C60", "LG+S256", "PEINT", "Real"]
JSD_THRESHOLD = 0.8   # figure3_conservation.DEFAULT_CONSERVATION_THRESHOLD


# ======================================================================================
# frame placement — the substitution-only shortcut
# ======================================================================================
def _old_sequences_dir() -> str:
    return os.path.join(str(cfg.MAFFT_ADD_DIR), "old_sequences")


def place_in_reference_frame(mcmc_msa: dict, old_msa: dict, root_label: str,
                             gap: str = gap_character) -> dict:
    """Drop substitution-only MCMC leaves into the shared ``old_sequences`` column frame.

    Each MCMC record has length ``L`` and is 1:1 positional with the root, so residue ``i`` of a
    leaf belongs at the alignment column of the root's ``i``-th residue. We read those columns off
    the root's gapped row in ``old_msa`` (which ungaps exactly to ``root_seq``), then rebuild every
    leaf as a gapped row of the frame width. Records whose length != L (the root span) are skipped.
    """
    root_row = old_msa[root_label]
    ncol = len(root_row)
    root_cols = [i for i, ch in enumerate(root_row) if ch != gap and ch != "."]
    L = len(root_cols)
    framed = {}
    for name, seq in mcmc_msa.items():
        if len(seq) != L:
            continue
        row = [gap] * ncol
        for i, col in enumerate(root_cols):
            row[col] = seq[i]
        framed[name] = "".join(row)
    return framed


def build_frame_msas(families, seq_dir: Path) -> Path:
    """Write each family's ESM-MCMC MSA in the old_sequences frame; return the frame dir."""
    FRAME_DIR.mkdir(parents=True, exist_ok=True)
    old_dir = _old_sequences_dir()
    for fam in families:
        mcmc = read_msa(str(seq_dir / f"{fam}.txt"))
        old = read_msa(os.path.join(old_dir, f"{fam}.txt"))
        root_label = list(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")))[0]
        framed = place_in_reference_frame(mcmc, old, root_label)
        write_msa(framed, str(FRAME_DIR / f"{fam}.txt"))
    return FRAME_DIR


# ======================================================================================
# ESM-IF Approach 1 — GT-structure likelihood (ESM-MCMC only; others pulled from the CSV)
# ======================================================================================
def run_esmif_a1(families, frame_dir: Path) -> pd.DataFrame:
    """Per-family mean ESM-IF LL of ESM-MCMC leaves threaded onto the GT structure.

    Mirrors ``benchmarks.esmif_validation.run_approach1`` exactly (same reference-frame threading
    via ``old_sequences/seq1``, same non-root leaf set), but for the ESM-MCMC frame MSA only.
    """
    from paper import esmif  # applies the biotite<->fair-esm shim, must precede invf import
    import esm.inverse_folding as invf
    from benchmarks.esmif_validation import nonroot_leaves

    esmif.load_model()
    old_dir = _old_sequences_dir()
    rows = []
    for fam in families:
        try:
            old = read_msa(os.path.join(old_dir, f"{fam}.txt"))
            keep = esmif.reference_columns(old["seq1"], gap_character)
            coords, gt_seq = invf.util.load_coords(
                os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"), fam.split("_")[-1]
            )
            mcmc = read_msa(str(frame_dir / f"{fam}.txt"))
        except Exception as e:  # pragma: no cover
            print(f"  [A1] skip {fam}: {e}")
            continue
        seqs, masks = [], []
        for n in nonroot_leaves(fam):
            if n in mcmc:
                t, m = esmif.thread_via_reference_frame(mcmc[n], keep, gap_character)
                seqs.append(t); masks.append(m)
        if not seqs:
            continue
        scored = esmif.score_batch(coords, seqs, masks)
        lls = np.array([d["ll"] for d in scored], dtype=float)
        rows.append({"family": fam, "model": MCMC_MODEL, "ll": float(np.nanmean(lls)),
                     "n_leaves": len(seqs)})
        print(f"  [A1] {fam}: ESM-MCMC LL {rows[-1]['ll']:+.3f} over {len(seqs)} leaves")
    return pd.DataFrame(rows)


def esmif_comparison(a1_mcmc: pd.DataFrame, families) -> pd.DataFrame:
    """Merge ESM-MCMC LL with the other models' cached per-family LL into one wide table."""
    other = pd.read_csv(ESMIF_CSV)
    other = other[other.family.isin(families)][["family", "model", "ll"]]
    both = pd.concat([other, a1_mcmc[["family", "model", "ll"]]], ignore_index=True)
    return both.pivot_table(index="family", columns="model", values="ll")


# ======================================================================================
# JSD conservation — all models on these families (ESM-MCMC via the frame dir)
# ======================================================================================
def run_jsd(families, frame_dir: Path, threshold: float = JSD_THRESHOLD) -> pd.DataFrame:
    from paper.jsd import family_jsd
    from figures.figure3_conservation import model_msa_dirs

    msa_dirs = dict(model_msa_dirs())
    msa_dirs[MCMC_MODEL] = str(frame_dir)
    rows = []
    for fam in families:
        try:
            ts = generate_tree_split(str(cfg.TREE_DIR), fam)
            per_model, _ = family_jsd(msa_dirs, fam, ts, threshold)
        except Exception as e:  # pragma: no cover
            print(f"  [JSD] skip {fam}: {e}")
            continue
        for model, val in per_model.items():
            rows.append({"family": fam, "model": model, "jsd": float(val)})
        print(f"  [JSD] {fam}: ESM-MCMC {per_model.get(MCMC_MODEL, float('nan')):.3f}")
    return pd.DataFrame(rows)


# ======================================================================================
# report
# ======================================================================================
def write_report(a1_wide, a1_mcmc, jsd_df, families):
    lines = ["# ESM2-MCMC vs. PEINT / classical / Real — standalone evals\n",
             "Evals on the ESM2-MCMC simulated MSAs (Bitbol Phylogeny-ESM2 / ESM lm-design "
             "algorithm), kept separate from the main figures. Same families, same metric "
             "machinery, same non-root leaf set as the PEINT comparison.\n",
             f"_Families ({len(families)}): {', '.join(families)}._\n"]

    # ESM-IF Approach 1
    models_a1 = [m for m in OTHER_MODELS if m in a1_wide.columns] + [MCMC_MODEL]
    lines.append("\n## ESM-IF Approach 1 — GT-structure log-likelihood (higher = more compatible)\n")
    lines.append("Per-family LL (other models from the cached esmif run; ESM-MCMC scored here):\n")
    lines.append("| family | " + " | ".join(models_a1) + " |")
    lines.append("|" + "---|" * (len(models_a1) + 1))
    for fam in families:
        if fam not in a1_wide.index:
            continue
        cells = [f"{a1_wide.loc[fam, m]:+.2f}" if m in a1_wide.columns and not pd.isna(a1_wide.loc[fam, m]) else "—"
                 for m in models_a1]
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    a1_long = a1_wide.reset_index().melt(id_vars="family", var_name="model", value_name="ll")
    lines.append("| **median** | " + " | ".join(
        f"**{v:+.2f}**" if (v := a1_long[a1_long.model == m]["ll"].median()) == v else "—"
        for m in models_a1) + " |")

    # JSD
    lines.append("\n## JSD conservation vs. Real (lower = closer to real conservation)\n")
    jsd_models = [m for m in OTHER_MODELS if m != "Real"] + ["Real (other split)", MCMC_MODEL]
    jsd_models = [m for m in jsd_models if m in set(jsd_df.model)]
    lines.append("| family | " + " | ".join(jsd_models) + " |")
    lines.append("|" + "---|" * (len(jsd_models) + 1))
    jpv = jsd_df.pivot_table(index="family", columns="model", values="jsd")
    for fam in families:
        if fam not in jpv.index:
            continue
        cells = [f"{jpv.loc[fam, m]:.3f}" if m in jpv.columns and not pd.isna(jpv.loc[fam, m]) else "—"
                 for m in jsd_models]
        lines.append(f"| {fam} | " + " | ".join(cells) + " |")
    lines.append("| **median** | " + " | ".join(
        f"**{jsd_df[jsd_df.model == m]['jsd'].median():.3f}**" if m in set(jsd_df.model) else "—"
        for m in jsd_models) + " |")

    lines.append("\n_Frame placement: ESM-MCMC leaves dropped into the old_sequences column frame "
                 "via the root gap pattern (substitution-only; exact, no mafft --add). "
                 "JSD threshold "
                 f"{JSD_THRESHOLD}; conserved sites lie within the root span for these families._\n")
    (OUT_DIR / "REBUTTAL_esm_mcmc_eval.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-dir", default=str(DEFAULT_SEQ_DIR), help="ESM-MCMC sequences dir")
    ap.add_argument("--families", nargs="*", default=None, help="explicit family list")
    ap.add_argument("--skip-esmif", action="store_true", help="skip Approach-1 (GPU) — JSD only")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seq_dir = Path(args.seq_dir)

    families = args.families or sorted(
        f[:-4] for f in os.listdir(seq_dir) if f.endswith(".txt")
    )
    print(f"ESM-MCMC evals over {len(families)} families: {', '.join(families)}")
    print(f"seq-dir={seq_dir}\noutput -> {OUT_DIR}")

    frame_dir = build_frame_msas(families, seq_dir)
    print(f"frame MSAs -> {frame_dir}")

    a1_wide = a1_mcmc = None
    if not args.skip_esmif:
        print("[ESM-IF Approach 1] scoring ESM-MCMC on GT structures ...")
        a1_mcmc = run_esmif_a1(families, frame_dir)
        a1_mcmc.to_csv(OUT_DIR / "esmif_a1_esm_mcmc.csv", index=False)
        a1_wide = esmif_comparison(a1_mcmc, families)
        a1_wide.to_csv(OUT_DIR / "esmif_a1_comparison.csv")
        print(a1_wide.round(2).to_string())

    print("[JSD] computing conservation JSD for all models ...")
    jsd_df = run_jsd(families, frame_dir)
    jsd_df.to_csv(OUT_DIR / "jsd_comparison.csv", index=False)
    print(jsd_df.pivot_table(index="family", columns="model", values="jsd").round(3).to_string())

    if a1_wide is not None:
        write_report(a1_wide, a1_mcmc, jsd_df, families)
        print(f"Wrote report -> {OUT_DIR / 'REBUTTAL_esm_mcmc_eval.md'}")


if __name__ == "__main__":
    main()
