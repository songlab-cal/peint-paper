"""Per-site conservation heatmaps, and the 3Di certainty histogram.

Supplementary panels ported from the model repo's ``benchmarks/sequence.py``: for one
family, each model's per-site residue usage on the conserved columns, shown raw,
gap-renormalized, and as a difference from the real data.

The divergence machinery that used to live alongside these plots is gone. ``paper.jsd``
is the single source of truth for site frequencies, conserved-site selection and JSD, and
the aggregate per-model boxplot is ``figures.figure3_conservation.plot_jsd_boxplot``. What
remains here is plotting, plus the one input ``paper.jsd`` does not hand back: the raw
frequency tables, gap row included, that the un-normalized heatmap displays.

Amino acids and 3Di structural states share every function via ``foldseek_states``; the
two differ only in where their MSAs live (see ``paper.jsd.msa_path``).
"""

import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from protevo.utils import read_msa

from paper.jsd import (
    REAL,
    REAL_OTHER_SPLIT,
    conserved_sites,
    family_site_distributions,
    msa_path,
    site_frequencies,
)
from paper.splits import SPLIT_A, SPLIT_B, filter_msa_based_on_split

# Canonical panel order for the heatmap grids. Models absent from a run are skipped.
MODEL_ORDER = [
    REAL,
    REAL_OTHER_SPLIT,
    "PEINT (Progressive)",
    "PEINT (Single Shot)",
    "WAG",
    "LG",
    "LG4X",
    "LG+C60",
    "LG+C60 (prior_anchored)",
    "LG+S256",
    "LG+S256 (prior_anchored)",
]

# The difference panel is a fixed 3x2 grid against the real data, so it shows the five
# models the published version showed rather than growing with the run.
DIFFERENCE_MODEL_ORDER = [
    REAL_OTHER_SPLIT,
    "PEINT (Progressive)",
    "PEINT (Single Shot)",
    "WAG",
    "LG",
]

_BASE_RCPARAMS = {
    "xtick.bottom": True,
    "ytick.left": True,
    "ytick.minor.left": True,
    "grid.linewidth": 0.5,
    "font.size": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "pdf.fonttype": 42,
}

_TICK_RCPARAMS = {
    "xtick.major.size": 2,
    "xtick.minor.size": 1,
    "ytick.major.size": 2,
    "ytick.minor.size": 1,
    "xtick.major.width": 0.5,
    "xtick.minor.width": 0.5,
    "ytick.major.width": 0.5,
    "ytick.minor.width": 0.5,
}


def _apply_style(extra: Dict = None) -> None:
    """Seaborn theme first — it resets rcParams — then our overrides."""
    sns.set_theme(style="white")
    mpl.rcParams.update(_BASE_RCPARAMS)
    if extra:
        mpl.rcParams.update(extra)


def _vocabulary_labels(foldseek_states: bool) -> str:
    return "3Di State" if foldseek_states else "Amino Acid"


def _suptitle(family: str, in_family: bool, conservation_threshold: float, foldseek_states: bool) -> str:
    training_fam_status = "In family" if in_family else "Out family"
    vocabulary = "3Di" if foldseek_states else "AA"
    return (
        f"Per-site {vocabulary} frequencies for {family} ({training_fam_status}), "
        f"{int(conservation_threshold * 100)}% Conserved Sites Only"
    )


def family_frequencies(
    msa_dirs: Dict[str, str],
    family: str,
    tree_split: Dict[str, List[str]],
    foldseek_states: bool = False,
) -> Dict[str, pd.DataFrame]:
    """Raw per-site frequencies per model, gap row included, over every site.

    ``paper.jsd.family_site_distributions`` drops the gap row, renormalizes, and restricts
    to conserved sites — right for divergences, but it loses the gap occupancy the
    un-normalized heatmap shows and the per-model conserved-site counts annotated on each
    panel. Split assignment is deliberately identical to that function's.
    """
    if REAL not in msa_dirs:
        raise KeyError(f"msa_dirs must contain a {REAL!r} entry; got {sorted(msa_dirs)}.")

    msas = {
        model: read_msa(msa_path(msa_dir, family, foldseek_states))
        for model, msa_dir in msa_dirs.items()
    }
    msas[REAL_OTHER_SPLIT] = read_msa(msa_path(msa_dirs[REAL], family, foldseek_states))

    return {
        model: site_frequencies(
            filter_msa_based_on_split(
                msa, tree_split, SPLIT_B if model == REAL_OTHER_SPLIT else SPLIT_A
            )
        )
        for model, msa in msas.items()
    }


def plot_conservation_differences(
    filtered_maps: Dict[str, pd.DataFrame],
    num_conserved_sites: Dict[str, int],
    out_path: str,
    in_family: bool,
    family: str,
    conservation_threshold: float,
    foldseek_states: bool = False,
) -> None:
    """Heatmaps of each model's per-site frequencies minus the real data's."""
    if REAL not in filtered_maps:
        raise KeyError(
            f"filtered_maps must contain a {REAL!r} entry to difference against; "
            f"got {sorted(filtered_maps)}."
        )

    _apply_style(_TICK_RCPARAMS)

    fig, axs = plt.subplots(3, 2, figsize=(8, 8), constrained_layout=True)
    ax_positions = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)]
    sorted_items = [
        (title, filtered_maps[title]) for title in DIFFERENCE_MODEL_ORDER if title in filtered_maps
    ]

    reference = filtered_maps[REAL]

    for (title, site_freqs_df), ax_pos in zip(sorted_items, ax_positions):
        axs[ax_pos].set_title(f"{title}")

        if site_freqs_df.empty:
            continue

        difference_map = site_freqs_df.subtract(reference, fill_value=0)

        sns.heatmap(
            difference_map,
            ax=axs[ax_pos],
            cmap="vlag_r",
            cbar=False,
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=difference_map.columns,
            yticklabels=difference_map.index,
        )

        axs[ax_pos].text(
            0.98,
            0.98,
            f"{num_conserved_sites[title]} Conserved Sites",
            horizontalalignment="right",
            verticalalignment="top",
            transform=axs[ax_pos].transAxes,
            color="black",
            fontsize=8,
        )

    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="vlag_r"), ax=axs, shrink=0.7, pad=0.01)

    fig.suptitle(_suptitle(family, in_family, conservation_threshold, foldseek_states), fontsize=6)
    fig.supylabel(_vocabulary_labels(foldseek_states), fontsize=12)
    fig.supxlabel("Site", fontsize=12)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_conservation(
    filtered_maps: Dict[str, pd.DataFrame],
    num_conserved_sites: Dict[str, int],
    out_path: str,
    in_family: bool,
    family: str,
    conservation_threshold: float,
    foldseek_states: bool = False,
) -> None:
    """Heatmaps of per-site vocabulary frequencies, one panel per model."""
    _apply_style()

    # Canonical order for the models we know; anything else the caller passed goes last.
    known = [t for t in MODEL_ORDER if t in filtered_maps]
    unknown = [t for t in filtered_maps if t not in MODEL_ORDER]
    sorted_items = [(t, filtered_maps[t]) for t in known + unknown]

    n_rows = (len(sorted_items) + 1) // 2
    fig, axs = plt.subplots(
        n_rows, 2, figsize=(8, 8 / 3 * n_rows), constrained_layout=True, squeeze=False
    )
    ax_positions = [(r, c) for r in range(n_rows) for c in range(2)]
    # An odd number of models leaves trailing slots; blank them so the grid does not show an
    # empty framed axes next to the last panel.
    for pos in ax_positions[len(sorted_items):]:
        axs[pos].axis("off")

    for (title, site_freqs_df), ax_pos in zip(sorted_items, ax_positions):
        axs[ax_pos].set_title(f"{title}")

        if site_freqs_df.empty:
            continue

        sns.heatmap(
            site_freqs_df,
            ax=axs[ax_pos],
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=1,
        )

        axs[ax_pos].set_xticks(
            np.arange(len(site_freqs_df.columns)) + 0.5,
            labels=site_freqs_df.columns,
            rotation=90,
            ha="right",
        )
        axs[ax_pos].set_yticks(
            np.arange(len(site_freqs_df.index)) + 0.5,
            labels=site_freqs_df.index,
            rotation=0,
            ha="right",
        )

        axs[ax_pos].text(
            0.98,
            0.98,
            f"{num_conserved_sites[title]} Conserved Sites",
            horizontalalignment="right",
            verticalalignment="top",
            transform=axs[ax_pos].transAxes,
            color="black",
            fontsize=10,
        )

        axs[ax_pos].tick_params(axis="both", which="both", width=0.5, length=2)

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap="Blues"), ax=axs, shrink=0.5, pad=0.01)

    fig.suptitle(_suptitle(family, in_family, conservation_threshold, foldseek_states), fontsize=10)
    fig.supylabel(_vocabulary_labels(foldseek_states), fontsize=10)
    fig.supxlabel("Site", fontsize=10)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)
    plt.close()


def generate_conservation_plots(
    msa_dirs: Dict[str, str],
    family: str,
    tree_split: Dict[str, List[str]],
    conservation_threshold: float,
    in_family: bool,
    out_path: str,
    foldseek_states: bool = False,
) -> None:
    """Write one family's three conservation heatmaps under ``out_path``.

    Raw frequencies, gap-renormalized frequencies, and the normalized difference from the
    real data. Sites and normalized distributions come from ``paper.jsd``, so these panels
    are drawn on exactly the columns the reported JSD is scored on.
    """
    distributions, sites = family_site_distributions(
        msa_dirs, family, tree_split, conservation_threshold, foldseek_states
    )
    frequencies = family_frequencies(msa_dirs, family, tree_split, foldseek_states)

    raw_maps = {model: freqs[sites] for model, freqs in frequencies.items()}
    # Annotated per panel: how many sites that model conserves overall, not how many of
    # the plotted (real-derived) columns it conserves.
    num_conserved_sites = {
        model: len(conserved_sites(freqs, conservation_threshold))
        for model, freqs in frequencies.items()
    }

    prefix = "3di" if foldseek_states else "aa"
    conservation_out_path = f"{out_path}/conservation/{prefix}_{conservation_threshold}.pdf"
    normalized_conservation_out_path = (
        f"{out_path}/conservation/{prefix}_{conservation_threshold}_normalized.pdf"
    )
    normalized_differences_out_path = (
        f"{out_path}/conservation_differences/{prefix}_{conservation_threshold}.pdf"
    )

    plot_conservation(
        filtered_maps=raw_maps,
        num_conserved_sites=num_conserved_sites,
        out_path=conservation_out_path,
        in_family=in_family,
        family=family,
        conservation_threshold=conservation_threshold,
        foldseek_states=foldseek_states,
    )

    plot_conservation_differences(
        filtered_maps=distributions,
        num_conserved_sites=num_conserved_sites,
        out_path=normalized_differences_out_path,
        in_family=in_family,
        family=family,
        conservation_threshold=conservation_threshold,
        foldseek_states=foldseek_states,
    )

    plot_conservation(
        filtered_maps=distributions,
        num_conserved_sites=num_conserved_sites,
        out_path=normalized_conservation_out_path,
        in_family=in_family,
        family=family,
        conservation_threshold=conservation_threshold,
        foldseek_states=foldseek_states,
    )


def generate_3di_certainty_plots(
    probabilities_dirs: Dict[str, Dict[str, str]],
    family: str,
    in_family: bool,
    out_path: str,
) -> None:
    """Histogram of ProstT5's per-sequence mean top-token probability, per model.

    A diagnostic for the 3Di benchmark: it says how confident the 3Di annotation itself is
    on each model's sequences, which bounds how much the 3Di conservation panels can mean.
    """
    color_map = {
        REAL: "blue",
        "PEINT (Progressive)": "green",
        "PEINT (Single Shot)": "orange",
        "WAG": "red",
        "LG": "purple",
        "LG4X": "gold",
        "LG+C60": "brown",
        "LG+C60 (prior_anchored)": "sienna",
        "LG+S256": "pink",
        "LG+S256 (prior_anchored)": "hotpink",
    }
    labels = [m for m in color_map if m in probabilities_dirs]
    if not labels:
        raise KeyError(
            f"None of the known models {sorted(color_map)} are in probabilities_dirs "
            f"{sorted(probabilities_dirs)}."
        )

    dfs = []
    for model in labels:
        probabilities_path = os.path.join(
            probabilities_dirs[model]["output_probabilities_dir"], family + ".txt"
        )
        probs = pd.read_csv(probabilities_path, header=None, names=["Sequence", "Probability"])
        dfs.append(probs["Probability"])

    plt.figure(figsize=(10, 6))
    for df, label in zip(dfs, labels):
        sns.histplot(
            df, kde=False, label=label, color=color_map[label], alpha=0.45, bins=40, binrange=(0, 100)
        )

    training_fam_status = "In family" if in_family else "Out family"
    plt.title(
        rf"{family} ({training_fam_status}) Distribution of "
        rf"$p$(most likely token per-site) averaged across sequence"
    )
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.legend(title="Sequence Type")

    plt.tight_layout()
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(f"{out_path}/3di_probabilities.png", dpi=300)
    plt.close()
