"""Shared publication plotting style for VEP figures.

Centralizes all rcParams so figure modules do not redefine styles. Call
``_set_publication_style()`` before plotting. Was previously defined in _vep_utils.
"""

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt


def _set_publication_style():
    """Apply Illustrator-friendly, publication-ready Matplotlib/Seaborn styles."""
    sns.set_theme(style="white")
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.minor.left"] = True
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams["axes.linewidth"] = 0.5
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 12,
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 12,
            "legend.title_fontsize": 12,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "xtick.major.size": 2,
            "ytick.major.size": 2,
        }
    )
