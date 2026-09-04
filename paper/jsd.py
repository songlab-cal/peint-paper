"""Jensen-Shannon divergence between per-site residue distributions.

Single source of truth for the conservation analysis. The model repo carried four
near-identical copies of this computation: ``compute_jsd`` and ``compute_kld`` in
``benchmarks/sequence.py`` shared ~70 lines verbatim (and ``compute_kld(use_jsd=True)``
was a third copy of the JSD loop), with a fourth inlined in ``fig4_conservation.ipynb``.
Everything here is that logic, deduplicated.

Amino acids and 3Di structural states share this module: 3Di states are letters over
the same 20-symbol alphabet, so one vocabulary and one frequency routine serve both.

KL divergence is deliberately not provided. It is unbounded whenever a conserved
residue's replacement falls outside a model's support, which made it uninformative
for this comparison. JSD is the reported statistic.
"""

import os
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from peint.utils import amino_acids, gap_character, read_msa

from paper.splits import REAL, REAL_OTHER_SPLIT, SPLIT_A, SPLIT_B, filter_msa_based_on_split

# Rows of every frequency table: the 20 residues plus the gap. 3Di states reuse this
# alphabet, which is why the same code serves both vocabularies.
VOCAB: Tuple[str, ...] = tuple(amino_acids) + (gap_character,)
RESIDUES: Tuple[str, ...] = tuple(amino_acids)



def msa_path(msa_dir, family: str, foldseek_states: bool = False) -> str:
    """Path to a family's aligned MSA. 3Di runs nest theirs under a sub-key."""
    if foldseek_states:
        return os.path.join(msa_dir["output_3di_dir"], f"{family}_aligned.txt")
    return os.path.join(msa_dir, f"{family}.txt")


def site_frequencies(msa: Dict[str, str]) -> pd.DataFrame:
    """Per-site vocabulary frequencies: rows are ``VOCAB``, columns are 1-based sites."""
    if not msa:
        raise ValueError("Cannot compute site frequencies from an empty MSA.")

    columns = pd.DataFrame([list(seq) for seq in msa.values()])
    frequencies = columns.apply(
        lambda col: col.value_counts(normalize=True).reindex(VOCAB, fill_value=0.0)
    )
    frequencies.columns = [str(i + 1) for i in range(frequencies.shape[1])]
    return frequencies


def residue_distributions(frequencies: pd.DataFrame) -> pd.DataFrame:
    """Drop the gap row and renormalize, making each column a distribution over residues."""
    residues = frequencies.drop(gap_character)
    # All-gap columns divide by zero; they carry no residue signal, so zero them out.
    return residues.div(residues.sum(axis=0), axis=1).fillna(0.0)


def conserved_sites(frequencies: pd.DataFrame, threshold: float) -> List[str]:
    """Sites where some non-gap residue exceeds ``threshold`` frequency."""
    residues = frequencies.drop(gap_character)
    return [site for site in frequencies.columns if (residues[site] > threshold).any()]


def jsd(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    """Jensen-Shannon distance between two residue distributions."""
    return float(jensenshannon(_regularize(p, eps), _regularize(q, eps)))


def family_site_distributions(
    msa_dirs: Dict[str, str],
    family: str,
    tree_split: Dict[str, List[str]],
    conservation_threshold: float,
    foldseek_states: bool = False,
) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    """Per-model residue distributions on a family's conserved sites.

    Every model is restricted to split A; the real data is read a second time on
    split B as the ``Real (other split)`` baseline. Conserved sites are defined
    empirically, as the union of sites conserved in *either* real split, so all
    models are scored on the same set of columns.
    """
    if REAL not in msa_dirs:
        raise KeyError(f"msa_dirs must contain a {REAL!r} entry; got {sorted(msa_dirs)}.")

    msas = {
        model: read_msa(msa_path(msa_dir, family, foldseek_states))
        for model, msa_dir in msa_dirs.items()
    }
    msas[REAL_OTHER_SPLIT] = read_msa(msa_path(msa_dirs[REAL], family, foldseek_states))

    msas = {
        model: filter_msa_based_on_split(
            msa, tree_split, SPLIT_B if model == REAL_OTHER_SPLIT else SPLIT_A
        )
        for model, msa in msas.items()
    }

    frequencies = {model: site_frequencies(msa) for model, msa in msas.items()}
    sites = sorted(
        set(conserved_sites(frequencies[REAL], conservation_threshold))
        | set(conserved_sites(frequencies[REAL_OTHER_SPLIT], conservation_threshold)),
        key=int,
    )
    if not sites:
        raise ValueError(
            f"No conserved sites for {family} at threshold {conservation_threshold}."
        )

    distributions = {
        model: residue_distributions(freqs)[sites] for model, freqs in frequencies.items()
    }
    return distributions, sites


def family_jsd(
    msa_dirs: Dict[str, str],
    family: str,
    tree_split: Dict[str, List[str]],
    conservation_threshold: float,
    foldseek_states: bool = False,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Mean JSD-vs-real per model over a family's conserved sites, and the per-site table."""
    distributions, sites = family_site_distributions(
        msa_dirs, family, tree_split, conservation_threshold, foldseek_states
    )
    real = distributions[REAL]

    per_site = pd.DataFrame(
        {
            model: [jsd(real[site].values, dist[site].values) for site in sites]
            for model, dist in distributions.items()
            if model != REAL
        },
        index=sites,
    )
    return per_site.mean().to_dict(), per_site


def signed_residue_contributions(
    real: np.ndarray,
    model: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """Per-residue signed divergence contributions for logo plots.

    Takes and returns ``(sites, residues)``. The sign shows which way the model moved
    relative to the real data; the magnitude is the site's divergence scaled by how far
    that residue's frequency shifted.

    The weighting uses the JS *divergence* rather than the square-root distance returned
    by ``jsd()`` — this is what sets the published panel's logo heights.
    """
    p = _regularize_rows(np.asarray(real, dtype=float), eps)
    q = _regularize_rows(np.asarray(model, dtype=float), eps)

    divergence = _js_divergence_rows(p, q).reshape(-1, 1)
    frequency_shift = q - p
    return np.sign(frequency_shift) * divergence * np.abs(frequency_shift)


def _regularize(v: Sequence[float], eps: float) -> np.ndarray:
    v = np.asarray(v, dtype=float) + eps
    return v / v.sum()


def _regularize_rows(m: np.ndarray, eps: float) -> np.ndarray:
    m = m + eps
    return m / m.sum(axis=1, keepdims=True)


def _kl_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.sum(np.where(p != 0, p * np.log(p / q), 0), axis=1)


def _js_divergence_rows(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    m = 0.5 * (p + q)
    return 0.5 * (_kl_rows(p, m) + _kl_rows(q, m))
