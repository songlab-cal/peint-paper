"""OmegaFold pLDDT on the ESM2-MCMC (Bitbol-style) simulated leaves — a template-free, non-ESM-IF
structural check (reviewer response).

The ESM-IF result flattered ESM-MCMC (it beat every classical simulator and edged PEINT), but ESM-IF
shares ESM2's training universe, so that metric may be too ESM-centric. OmegaFold folds each sequence
de novo and reports pLDDT — an independent structural-quality signal. We fold ESM-MCMC's leaves on the
**same leaf set** already folded for PEINT / Real / the classical simulators (their exact structure
names are copied from the OmegaFold cache), so every model is scored on the same tree positions.

For a clean apples-to-apples we recompute the mean pLDDT for ALL models identically (mean CA B-factor
per structure, averaged over the shared leaf set), rather than mixing in the cached CSV.

Run::  python -m benchmarks.esm_mcmc_omegafold [--families ...]
Output -> figures/output/esm_mcmc/eval/omegafold_plddt.csv (+ printed table).
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import biotite.structure.io as bsio

import paper_config as cfg
from protevo import caching as protevo_caching
from cherryml import caching as cherryml_caching
from protevo.utils import read_msa, write_msa
from paper.structure_prediction import generate_omegafold_predictions

OG = Path(cfg.RESULTS_DIR) / "omegafold"
SEQ_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "hamming" / "sequences"
OUT_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "eval"
IN_DIR = OUT_DIR / "omegafold_input"
MCMC = "ESM-MCMC"
REF_MODEL = "Real"   # its cache defines the shared leaf set (all models share the same names)
# cache dir name -> display label (JSD/esmif figure order); others (prior-anchored, single-shot) skipped
MODELS = {"WAG": "WAG", "LG": "LG", "LG4X": "LG4X", "LG+C60": "LG+C60", "LG+S256": "LG+S256",
          "PEINT (Progressive)": "PEINT", "Real": "Real", "Real (other split)": "Real (other split)"}
ORDER = ["WAG", "LG", "LG4X", "LG+C60", "LG+S256", "PEINT", MCMC, "Real", "Real (other split)"]


def mean_plddt(pdb_path: str) -> float:
    """Mean pLDDT = mean B-factor (OmegaFold writes per-residue pLDDT into the B-factor)."""
    s = bsio.load_structure(pdb_path, extra_fields=["b_factor"])
    return float(s.b_factor.mean())


def folded_leaf_names(fam: str) -> list:
    """The exact leaf names folded for the reference model (shared across all models)."""
    ref = OG / fam / REF_MODEL / "structures"
    return sorted(os.path.basename(p)[:-4] for p in glob.glob(str(ref / "seq*.pdb")))


def cached_model_plddt(fam: str, cache_dir: str, names) -> float:
    """Mean pLDDT over the shared leaf set from a model's cached OmegaFold structures."""
    sdir = OG / fam / cache_dir / "structures"
    vals = [mean_plddt(str(sdir / f"{n}.pdb")) for n in names if (sdir / f"{n}.pdb").exists()]
    return float(np.mean(vals)) if vals else float("nan")


def fold_esm_mcmc(fam: str, names) -> float:
    """Fold ESM-MCMC's sequences for the shared leaf set and return their mean pLDDT."""
    mcmc = read_msa(str(SEQ_DIR / f"{fam}.txt"))
    filtered = {n: mcmc[n] for n in names if n in mcmc}
    IN_DIR.mkdir(parents=True, exist_ok=True)
    write_msa(filtered, str(IN_DIR / f"{fam}.txt"))
    res = generate_omegafold_predictions(
        sequences_dir=str(IN_DIR), family=fam, input_filename=f"{fam}.txt", keep_prefix="seq")
    sdir = res["output_structures_dir"]
    vals = [mean_plddt(p) for p in glob.glob(os.path.join(sdir, "seq*.pdb"))]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=None)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cherryml_caching.set_cache_dir("_cache_cherryml"); cherryml_caching.set_read_only(False)
    protevo_caching.set_cache_dir("_cache_protevo"); protevo_caching.set_read_only(False)

    families = args.families or sorted(f[:-4] for f in os.listdir(SEQ_DIR) if f.endswith(".txt"))
    print(f"OmegaFold pLDDT on ESM-MCMC leaves | {len(families)} families | shared leaf set from '{REF_MODEL}'")

    rows = []
    for i, fam in enumerate(families, 1):
        names = folded_leaf_names(fam)
        if not names:
            print(f"[{i}/{len(families)}] {fam}: no reference structures, skip"); continue
        rec = {"family": fam, "n_leaves": len(names)}
        try:
            rec[MCMC] = fold_esm_mcmc(fam, names)
        except Exception as e:  # pragma: no cover
            print(f"[{i}/{len(families)}] {fam}: ESM-MCMC fold failed ({e})"); rec[MCMC] = float("nan")
        for cache_dir, disp in MODELS.items():
            rec[disp] = cached_model_plddt(fam, cache_dir, names)
        rows.append(rec)
        cols = [m for m in ORDER if m in rec and not np.isnan(rec.get(m, float("nan")))]
        print(f"[{i}/{len(families)}] {fam}: " +
              " ".join(f"{m} {rec[m]:.1f}" for m in cols))
        pd.DataFrame(rows).to_csv(OUT_DIR / "omegafold_plddt.csv", index=False)

    df = pd.DataFrame(rows)
    print("\n=== median pLDDT over families (higher = more confidently folded) ===")
    for m in ORDER:
        if m in df.columns:
            print(f"  {m:20s} {df[m].median():.1f}")
    print(f"\nwrote {OUT_DIR/'omegafold_plddt.csv'}")


if __name__ == "__main__":
    main()
