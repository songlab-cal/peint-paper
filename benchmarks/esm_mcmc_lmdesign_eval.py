"""Eval the lm-design-simulator leaves the same way as PEINT/Bitbol: JSD + ESM-IF (+ divergence).

Reuses benchmarks.esm_mcmc_eval, monkeypatched to read the lm-design leaves and label them
'lm-design'. Structural evals (ESM-IF) are partly CIRCULAR for lm-design (its energy optimizes the
GT structure) — JSD (conservation) and divergence are the clean axes.
"""
import os, sys
from pathlib import Path
import numpy as np
import pandas as pd

import paper_config as cfg
from benchmarks import esm_mcmc_eval as ev

OUT = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "lmdesign" / "eval"
LMSEQ = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "lmdesign" / "sequences"


def main():
    families = sorted(f[:-4] for f in os.listdir(LMSEQ) if f.endswith(".txt"))
    print(f"lm-design eval over {len(families)} families: {', '.join(families)}")

    # relabel + redirect the reused eval module at the lm-design leaves
    ev.MCMC_MODEL = "lm-design"
    ev.FRAME_DIR = OUT / "frame_msa"
    frame = ev.build_frame_msas(families, LMSEQ)     # substitution-only frame placement (exact)

    print("[JSD] ...")
    jsd = ev.run_jsd(families, frame)                # all models incl 'lm-design'
    jp = jsd.pivot_table(index="family", columns="model", values="jsd")
    print("JSD medians (lower=closer to real conservation):")
    for m in ["Real (other split)", "PEINT (Progressive)", "lm-design", "LG+S256", "LG", "WAG"]:
        if m in jp.columns:
            print(f"  {m:22s} {jp[m].median():.3f}")

    print("[ESM-IF A1] ...  (CIRCULAR for lm-design — reads with caution)")
    a1 = ev.run_esmif_a1(families, frame)
    wide = ev.esmif_comparison(a1, families)
    print("ESM-IF A1 medians (higher=more compatible):")
    a1_long = wide.reset_index().melt(id_vars="family", var_name="model", value_name="ll")
    for m in ["Real", "lm-design", "PEINT", "LG+S256", "LG", "WAG"]:
        v = a1_long[a1_long.model == m]["ll"].median()
        if v == v:
            print(f"  {m:12s} {v:+.2f}")

    out = OUT
    out.mkdir(parents=True, exist_ok=True)
    jsd.to_csv(out / "jsd.csv", index=False)
    wide.to_csv(out / "esmif_a1.csv")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
