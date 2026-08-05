"""Generalization test on OmegaFold pLDDT: de-novo folding quality on unseen protein classes.

Unlike AF2Rank (which threads onto the experimental structure), OmegaFold folds each simulated
sequence from scratch, so its pLDDT reads out whether the simulated sequence encodes a foldable
protein on its own. This is the natural place to look at the *structural* holdout — families whose
superfamily was never seen in training — so we report both the Pfam-family and the structural
(ECOD, SCOPe) splits. As in the AF2Rank test, "Real (other split)" is the difficulty control.

Assumes the OmegaFold benchmark has already been run (``omegafold_plddt.csv``). Fetches the
annotation data itself. Run from the repo root::

    python -m benchmarks.generalization_omegafold
"""

import pandas as pd

import paper_config as cfg
from paper import generalization as gen


def load_omegafold() -> pd.DataFrame:
    """Long-form OmegaFold pLDDT (0-100): one row per (family, model)."""
    return pd.read_csv(cfg.require(cfg.RESULTS_DIR / "omegafold_plddt.csv"))


def main() -> None:
    gen.ensure_annotations()
    df = load_omegafold()

    print(f"OmegaFold: {df['family'].nunique()} families, {df['model'].nunique()} models")
    # Pfam family is the powered test; ECOD + SCOPe are the (underpowered) structural holdouts.
    stats = gen.report_stratified(
        df, "plddt", "generalization_omegafold_plddt",
        schemes=("pfam_family", "ecod_hgroup", "scop_superfamily"), value_label="OmegaFold pLDDT",
    )

    from paper.splits import REAL_OTHER_SPLIT
    print("\nNovel vs seen OmegaFold pLDDT (median), per model:")
    for scheme in stats["scheme"].unique():
        print(f"\n  [{scheme}]")
        for _, r in stats[stats["scheme"] == scheme].iterrows():
            flag = "  <-- PEINT" if r["model"].startswith("PEINT") else (
                "  <-- Real baseline" if r["model"] == REAL_OTHER_SPLIT else "")
            print(f"    {r['model']:<28} novel={r.get('median_novel', float('nan')):.1f} "
                  f"seen={r.get('median_seen', float('nan')):.1f} "
                  f"p={r['p']:.3g} p_matched={r['p_matched']:.3g} "
                  f"delta={r['cliffs_delta']:+.2f}{flag}")
    print(f"\nWrote tables + figure to {cfg.GENERALIZATION_DIR}")


if __name__ == "__main__":
    main()
