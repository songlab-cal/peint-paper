"""Generalization test on OmegaFold pLDDT: de-novo folding quality on unseen protein classes.

Unlike AF2Rank (which threads onto the experimental structure), OmegaFold folds each simulated
sequence from scratch, so its pLDDT reads out whether the simulated sequence encodes a foldable
protein on its own — split by whether the held-out family's Pfam family was seen in training.

Reads the per-(family, model) pLDDT table that figure3_structure_metrics writes
(``figure3_omegafold_plddt_ecdf.csv``), which already carries every model — WAG/LG/mixtures,
PEINT (ESM2), **PEINT (ESM-C)** (parsed from its own rev2 OmegaFold structures), and Real — under
one uniform per-family aggregation, so ESM-C is a first-class model here, not a merged one-off.
Fetches the annotation data itself. Run from the repo root::

    python -m benchmarks.generalization_omegafold
"""

import pathlib

import pandas as pd

import paper_config as cfg
from paper import generalization as gen

# Plotted models + order (Antoine): LG+S256, PEINT (ESM2), PEINT (ESM-C), Real.
FOCUS_MODELS = ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"]


def load_omegafold() -> pd.DataFrame:
    """Long-form OmegaFold pLDDT (0-100): one row per (family, model), all models incl ESM-C.

    The table ships with the data release; look there first so a fresh checkout works
    without having produced it, then fall back to a locally produced copy.
    """
    name = "figure3_omegafold_plddt_ecdf.csv"
    for d in (cfg.FIGURE_DATA_DIR, cfg.FIGURES_DIR):
        p = pathlib.Path(d) / name
        if p.exists():
            print(f"  reading {p}")
            return pd.read_csv(p)
    raise SystemExit(
        f"{name} not found in the release (local_data/figure_data) or figures/output. "
        f"Fetch the figure_data tier, or run figures.figure3_structure_metrics first."
    )


def main() -> None:
    # Novelty here is family-level: a family counts as novel only when every one of its Pfam
    # domains is novel to the held-out set. That partition is derived from the annotation
    # files, not from the per-domain table used by the sequence-conservation panel -- the two
    # define different novel sets, and substituting one for the other shifts the novel arm by
    # several pLDDT points while leaving the seen arm apparently correct.
    gen.ensure_annotations()
    df = load_omegafold()
    print(f"OmegaFold: {df['family'].nunique()} families, models={sorted(df['model'].unique())}")

    plotted = gen.plot_grouped_by_novelty(
        df, "plddt", "generalization_omegafold_plddt",
        model_order=FOCUS_MODELS, scheme="pfam_family",
        value_label="OmegaFold pLDDT",
        title="Novel vs seen Pfam family (held-out families)",
    )
    med = (plotted.groupby(["_stratum", "model"], observed=True)["plddt"].median()
           .unstack("_stratum"))
    print("\nMedian OmegaFold pLDDT (seen vs novel Pfam family), per model:")
    print(med.reindex(FOCUS_MODELS).round(1).to_string())
    print(f"\nWrote figure to {cfg.GENERALIZATION_DIR}")


if __name__ == "__main__":
    main()
