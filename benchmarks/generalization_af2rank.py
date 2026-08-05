"""Generalization test on AF2Rank pLDDT: does structure quality hold on unseen protein classes?

AF2Rank threads each simulated sequence onto the experimental structure and reports how confident
AlphaFold is in the result (pLDDT) — a measure of how "protein-like" the simulated sequence is in
its native fold. Here we re-stratify the already-cached AF2Rank scores by whether the held-out
family's Pfam family / structural superfamily (SCOPe, ECOD) was ever seen in training (see
``paper.generalization``),
and test novel vs seen for every model. The empirical "Real (other split)" row is the difficulty
control: if novel families are intrinsically harder, Real drops on novel too, and only a
PEINT-specific novel/seen gap would indicate a generalization failure.

Assumes the AF2Rank benchmark has already been run (``af2rank_comparisons.csv``). Fetches the
annotation data itself. Run from the repo root::

    python -m benchmarks.generalization_af2rank
"""

import pandas as pd

import paper_config as cfg
from paper import generalization as gen


def load_af2rank() -> pd.DataFrame:
    """Long-form AF2Rank scores: one row per (family, model) with tm / plddt / composite."""
    df = pd.read_csv(cfg.require(cfg.RESULTS_DIR / "af2rank_comparisons.csv"))
    # pLDDT is cached on a 0-1 scale here; put it on the familiar 0-100 scale.
    if df["plddt"].max() <= 1.5:
        df["plddt"] = df["plddt"] * 100.0
    return df


def main() -> None:
    gen.ensure_annotations()
    df = load_af2rank()

    print(f"AF2Rank: {df['family'].nunique()} families, {df['model'].nunique()} models")
    stats = gen.report_stratified(
        df, "plddt", "generalization_af2rank_plddt", value_label="AF2Rank pLDDT",
    )
    # Secondary quality measures, tables only (no separate figure).
    for col, stem in [("tm", "generalization_af2rank_tm"),
                      ("composite", "generalization_af2rank_composite")]:
        gen.report_stratified(df, col, stem, value_label=f"AF2Rank {col}")

    _print_headline(stats)


def _print_headline(stats: pd.DataFrame) -> None:
    from paper.splits import REAL_OTHER_SPLIT

    print("\nNovel vs seen AF2Rank pLDDT (median), per model:")
    for scheme in stats["scheme"].unique():
        print(f"\n  [{scheme}]")
        sub = stats[stats["scheme"] == scheme]
        for _, r in sub.iterrows():
            flag = "  <-- PEINT" if r["model"].startswith("PEINT") else (
                "  <-- Real baseline" if r["model"] == REAL_OTHER_SPLIT else "")
            print(f"    {r['model']:<28} novel={r.get('median_novel', float('nan')):.1f} "
                  f"seen={r.get('median_seen', float('nan')):.1f} "
                  f"p={r['p']:.3g} p_matched={r['p_matched']:.3g} "
                  f"delta={r['cliffs_delta']:+.2f}{flag}")
    print(f"\nWrote tables + figure to {cfg.GENERALIZATION_DIR}")


if __name__ == "__main__":
    main()
