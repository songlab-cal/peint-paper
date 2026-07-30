"""Figure 3 — conservation: how well each model reproduces real per-site residue usage.

Two panels, either or both selectable from the CLI:

``logo``    a 2x2 sequence-logo grid for one family. The first panel is the real
            residue distribution on conserved sites; the rest show each model's
            signed divergence from it (see ``paper.jsd``).
``boxplot`` mean JSD-vs-real per family, across all families, one box per model.

All divergences come from ``paper.jsd`` — there is exactly one JSD definition, used
for both panels (and for 3Di states, when that benchmark migrates). KL divergence is
not computed; it is unbounded when a conserved residue's replacement falls outside a
model's support.

Run from the repo root, e.g.::

    python -m figures.figure3_conservation --panels logo boxplot
"""

import argparse
import os
from typing import Dict, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import logomaker
from tqdm import tqdm

from paper.jsd import (
    REAL,
    REAL_OTHER_SPLIT,
    RESIDUES,
    family_jsd,
    family_site_distributions,
    signed_residue_contributions,
)
from paper.splits import generate_tree_split
import paper_config as cfg

mpl.rcParams["pdf.fonttype"] = 42

# The family carrying the structural panel in the paper.
DEFAULT_LOGO_FAMILY = "4wff_1_B"
DEFAULT_CONSERVATION_THRESHOLD = 0.8

# Models shown in the logo grid, after the real distribution itself.
LOGO_MODELS = [REAL_OTHER_SPLIT, "PEINT (Progressive)", "LG+S256"]
LOGO_LABELS = {
    REAL_OTHER_SPLIT: "Real (eval subtree)",
    "PEINT (Progressive)": "PEINT",
    "LG+S256": "LG+S256",
}

BOXPLOT_MODELS = [
    "WAG",
    "LG",
    "LG4X",
    "LG+C60",
    "LG+S256",
    "PEINT (Progressive)",
    REAL_OTHER_SPLIT,
]
BOXPLOT_LABELS = {
    "PEINT (Progressive)": "PEINT",
    REAL_OTHER_SPLIT: "Real (eval subtree)",
}


def model_msa_dirs() -> Dict[str, str]:
    """Aligned-MSA directory per model, all in the reference alignment frame."""
    return {
        REAL: str(cfg.MAFFT_ADD_DIR / "old_sequences"),
        "PEINT (Progressive)": str(cfg.MAFFT_ADD_DIR / "peint_progressive_dir"),
        "PEINT (Single Shot)": str(cfg.MAFFT_ADD_DIR / "peint_single_shot_dir"),
        "WAG": str(cfg.SIMULATIONS_DIR / "wag"),
        "LG": str(cfg.SIMULATIONS_DIR / "lg"),
        "LG4X": str(cfg.SIMULATIONS_DIR / "lg4x"),
        "LG+C60": str(cfg.SIMULATIONS_DIR / "lg_c60"),
        "LG+S256": str(cfg.SIMULATIONS_DIR / "lg_s256"),
    }


def _hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i:i + 2], 16) / 255 for i in (0, 2, 4))


# Residues coloured by physicochemical class.
AA_COLORS = {
    **{aa: _hex_to_rgb("fdd686") for aa in "AILMFWV"},   # hydrophobic
    **{aa: _hex_to_rgb("6da0cd") for aa in "RHK"},       # positive
    **{aa: _hex_to_rgb("b25e7e") for aa in "DE"},        # negative
    **{aa: _hex_to_rgb("5cb25d") for aa in "NQST"},      # polar uncharged
    "C": _hex_to_rgb("af93d7"),
    "P": _hex_to_rgb("af93d7"),
    "G": _hex_to_rgb("ffc5d3"),
    "Y": _hex_to_rgb("0f9015"),
}


def _logo_frame(matrix: np.ndarray, num_sites: int) -> pd.DataFrame:
    """(residues, sites) array -> the (sites, residues) frame logomaker expects."""
    return pd.DataFrame(matrix, index=list(RESIDUES), columns=np.arange(num_sites)).T


def plot_conservation_logos(
    family: str,
    msa_dirs: Dict[str, str],
    conservation_threshold: float,
    output_dir: str,
) -> pd.DataFrame:
    """2x2 logo grid for one family: real distribution + per-model signed divergence."""
    tree_split = generate_tree_split(str(cfg.require(cfg.TREE_DIR)), family)
    distributions, sites = family_site_distributions(
        msa_dirs, family, tree_split, conservation_threshold
    )
    mean_jsd, per_site = family_jsd(
        msa_dirs, family, tree_split, conservation_threshold
    )

    real = distributions[REAL]
    panels = [_logo_frame(real.values, len(sites))]
    titles = [REAL]
    for model in LOGO_MODELS:
        signed = signed_residue_contributions(real.values.T, distributions[model].values.T)
        panels.append(_logo_frame(signed.T, len(sites)))
        titles.append(f"{LOGO_LABELS[model]} (JS-Div: {mean_jsd[model]:.2f})")

    fig, axs = plt.subplots(2, 2, figsize=(4.8, 3))
    plt.subplots_adjust(left=0, bottom=0.0, right=1, top=1, wspace=0.2, hspace=0.2)

    for i, (ax, panel, title) in enumerate(zip(axs.flatten(), panels, titles)):
        logomaker.Logo(
            panel,
            ax=ax,
            color_scheme=AA_COLORS,
            fade_below=0.1,
            flip_below=(i == 0),
        )
        sns.despine(ax=ax, left=True, bottom=False)
        ax.set_xticks(np.arange(0, len(sites), 5), labels=[])
        ax.set_xticks(np.arange(len(sites)), labels=[], minor=True)
        ax.set_yticks([], labels=[])
        ax.set_title(title, fontsize=8)

        # The real panel is a probability distribution; the rest are signed
        # divergences on a shared symmetric scale.
        if i > 0:
            ax.set_ylim(-0.7, 0.7)
            ax.set_yticks([-0.6, 0, 0.6], labels=["-0.6", "0", "0.6"], fontsize=6)
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(
        os.path.join(output_dir, "figure3_conservation_lg_s256.pdf"), bbox_inches="tight"
    )
    fig.savefig(
        os.path.join(output_dir, "figure3_conservation_lg_s256.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

    per_site.to_csv(os.path.join(output_dir, f"figure3_conservation_jsd_{family}.csv"))
    return per_site


def collect_family_jsd(
    families: List[str],
    msa_dirs: Dict[str, str],
    conservation_threshold: float,
) -> pd.DataFrame:
    """Mean JSD per model for every family; rows are families, columns are models."""
    tree_dir = str(cfg.require(cfg.TREE_DIR))
    rows = {}
    skipped = {}

    for family in tqdm(families, desc="Computing JSD"):
        try:
            tree_split = generate_tree_split(tree_dir, family)
            mean_jsd, _ = family_jsd(
                msa_dirs, family, tree_split, conservation_threshold
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            skipped[family] = f"{type(exc).__name__}: {exc}"
            continue
        rows[family] = mean_jsd

    if not rows:
        raise RuntimeError(
            f"No family produced a JSD score ({len(skipped)} skipped). "
            f"First failure: {next(iter(skipped.items()), None)}"
        )
    if skipped:
        print(f"Skipped {len(skipped)}/{len(families)} families; first few:")
        for family, reason in list(skipped.items())[:5]:
            print(f"  {family}: {reason}")

    return pd.DataFrame.from_dict(rows, orient="index")


def plot_jsd_boxplot(
    jsd_df: pd.DataFrame,
    output_dir: str,
    filename_stem: str = "figure3_conservation_jsd_boxplot",
) -> None:
    """Distribution of per-family mean JSD, one box per model.

    ``filename_stem`` lets callers write several slices of the same plot; the
    generate_all_results driver uses it for the all / in-family / held-out breakdown.
    """
    palette = sns.color_palette()
    colors = {
        "WAG": palette[0],
        "LG": palette[1],
        "LG4X": palette[3],
        "LG+C60": palette[5],
        "LG+S256": palette[6],
        "PEINT (Progressive)": palette[2],
        REAL_OTHER_SPLIT: palette[4],
    }

    models = [m for m in BOXPLOT_MODELS if m in jsd_df.columns]
    missing = [m for m in BOXPLOT_MODELS if m not in jsd_df.columns]
    if missing:
        print(f"Note: no JSD columns for {missing}; omitting from the boxplot.")
    plot_df = jsd_df[models].rename(columns=BOXPLOT_LABELS)

    fig, ax = plt.subplots(figsize=(2, 3))
    sns.boxplot(
        data=plot_df,
        ax=ax,
        palette=[colors[m] for m in models],
        showfliers=True,
        flierprops=dict(
            marker="o", markersize=2, markerfacecolor="black", markeredgecolor=None, alpha=0.5
        ),
        width=0.75,
        linewidth=0.5,
    )

    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_ylabel("Mean JSD vs. real", fontsize=8)
    ax.yaxis.grid(True, linestyle=":", color="gray", linewidth=0.25)
    ax.xaxis.grid(True, linestyle=":", color="gray", linewidth=0.25)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{filename_stem}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(output_dir, f"{filename_stem}.png"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def discover_families(msa_dirs: Dict[str, str]) -> List[str]:
    real_dir = cfg.require(msa_dirs[REAL])
    return sorted(
        f[: -len(".txt")] for f in os.listdir(real_dir) if f.endswith(".txt")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panels",
        nargs="+",
        choices=["logo", "boxplot"],
        default=["logo", "boxplot"],
        help="Which panels to generate.",
    )
    parser.add_argument(
        "--family", default=DEFAULT_LOGO_FAMILY, help="Family for the logo panel."
    )
    parser.add_argument(
        "--conservation-threshold",
        type=float,
        default=DEFAULT_CONSERVATION_THRESHOLD,
        help="Residue frequency above which a site counts as conserved.",
    )
    parser.add_argument(
        "--max-families",
        type=int,
        default=None,
        help="Cap the number of families in the boxplot (for smoke runs).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(cfg.FIGURES_DIR),
        help="Where figures and CSVs are written.",
    )
    args = parser.parse_args()

    msa_dirs = model_msa_dirs()

    if "logo" in args.panels:
        plot_conservation_logos(
            family=args.family,
            msa_dirs=msa_dirs,
            conservation_threshold=args.conservation_threshold,
            output_dir=args.output_dir,
        )
        print(f"Wrote logo panel for {args.family} to {args.output_dir}")

    if "boxplot" in args.panels:
        families = discover_families(msa_dirs)
        if args.max_families is not None:
            families = families[: args.max_families]
        jsd_df = collect_family_jsd(families, msa_dirs, args.conservation_threshold)

        os.makedirs(args.output_dir, exist_ok=True)
        jsd_df.to_csv(os.path.join(args.output_dir, "figure3_conservation_jsd.csv"))
        plot_jsd_boxplot(jsd_df, args.output_dir)
        print(f"Wrote JSD boxplot over {len(jsd_df)} families to {args.output_dir}")


if __name__ == "__main__":
    main()
