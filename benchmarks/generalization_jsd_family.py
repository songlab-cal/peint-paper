"""Generalization test on conservation (JSD): does per-site fidelity hold on unseen protein classes?

The conservation benchmark measures, per family, the mean Jensen-Shannon divergence between each
model's per-site residue distributions and the real data (lower = the model reproduces the real
conservation pattern better). Here we recompute that per-family JSD for the held-out families
(via ``paper.jsd``, the same definition the paper uses) and stratify by Pfam-family / structural
(ECOD, SCOPe) novelty. "Real (other split)" is the floor: the JSD between two halves of the real
data, i.e. the best any model could do given finite sampling.

This is the family-level (approach #1) view of conservation; ``generalization_jsd_domain`` restricts
the same JSD to the residues of the *novel domain* itself (approach #2).

Assumes the alignment pipeline has been run (``mafft_add`` + simulation dirs exist). The per-family
JSD table is cached under ``cfg.GENERALIZATION_DIR``. Run from the repo root::

    python -m benchmarks.generalization_jsd_family
"""

from pathlib import Path

import pandas as pd

import paper_config as cfg
from paper import generalization as gen
from figures.figure3_conservation import DEFAULT_CONSERVATION_THRESHOLD, model_msa_dirs


def compute_or_load_family_jsd(force: bool = False) -> pd.DataFrame:
    """Per-family mean JSD (rows=family, cols=model) for the held-out families, cached to CSV."""
    cache = Path(cfg.GENERALIZATION_DIR) / "family_jsd_heldout.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache, index_col=0)
    cache.parent.mkdir(parents=True, exist_ok=True)
    means, _, skipped = gen.collect_family_jsd_fast(
        gen.eval_families(), model_msa_dirs(), DEFAULT_CONSERVATION_THRESHOLD
    )
    if skipped:
        print(f"(skipped {skipped} families with no computable JSD)")
    jsd = pd.DataFrame.from_dict(means, orient="index")
    jsd.to_csv(cache)
    return jsd


def main() -> None:
    gen.ensure_annotations()
    jsd = compute_or_load_family_jsd()
    long = jsd.reset_index(names="family").melt(
        id_vars="family", var_name="model", value_name="jsd"
    ).dropna(subset=["jsd"])

    print(f"Family JSD: {long['family'].nunique()} families, {long['model'].nunique()} models")
    stats = gen.report_stratified(long, "jsd", "generalization_jsd_family", value_label="Mean JSD vs. real")

    from paper.splits import REAL_OTHER_SPLIT
    print("\nNovel vs seen mean-JSD (median; lower is better), per model:")
    for scheme in stats["scheme"].unique():
        print(f"\n  [{scheme}]")
        for _, r in stats[stats["scheme"] == scheme].iterrows():
            flag = "  <-- PEINT" if r["model"].startswith("PEINT") else (
                "  <-- Real baseline" if r["model"] == REAL_OTHER_SPLIT else "")
            print(f"    {r['model']:<28} novel={r.get('median_novel', float('nan')):.3f} "
                  f"seen={r.get('median_seen', float('nan')):.3f} "
                  f"p={r['p']:.3g} p_matched={r['p_matched']:.3g} "
                  f"delta={r['cliffs_delta']:+.2f}{flag}")
    print(f"\nWrote tables + figure to {cfg.GENERALIZATION_DIR}")


if __name__ == "__main__":
    main()
