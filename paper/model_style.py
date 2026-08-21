"""Canonical model -> color for ALL paper figures — one source of truth.

Every figure (conservation JSD, structure-metrics ECDFs, PCP mutation counts, ...)
must color a given model identically. Historically each figure built its own palette
and they drifted; this module centralizes the mapping so they cannot.

Colors are seaborn-default-palette *indices* (read at call time, not hardcoded hexes),
matching the mapping figure3_conservation has always used:

    WAG=0  LG=1  PEINT(ESM2)=2  LG4X=3  Real=4  LG+C60=5  LG+S256=6

PEINT (ESM-C) is the one addition (rev2); it takes palette[9] (cyan) so it pairs with
the green PEINT (ESM2) and never collides with LG+C60's brown. Keys cover every alias a
figure might use (internal run names, revision suffixes, display variants).
"""

import seaborn as sns

# Canonical model name -> seaborn default-palette index.
_INDEX = {
    "WAG": 0,
    "LG": 1,
    "PEINT (ESM2)": 2,
    "LG4X": 3,
    "Real": 4,
    "LG+C60": 5,
    "LG+S256": 6,
    "PEINT (ESM-C)": 9,
}

# Aliases a figure might key on -> canonical name above.
_ALIAS = {
    "PEINT (Progressive)": "PEINT (ESM2)",  # internal run name for ESM2-150 PEINT
    "PEINT ESM2": "PEINT (ESM2)",
    "PEINT ESM2 (rev1)": "PEINT (ESM2)",
    "PEINT (ESM2-150M)": "PEINT (ESM2)",
    "PEINT ESM-C": "PEINT (ESM-C)",
    "PEINT (ESM-C 300M)": "PEINT (ESM-C)",
    "Real (other split)": "Real",
    "Real (eval subtree)": "Real",
    "Real (Inferred)": "Real",
    "LG_S256": "LG+S256",
    "LG+S256 (prior_anchored)": "LG+S256",
    "LG+C60 (prior_anchored)": "LG+C60",
}


def model_colors(palette=None):
    """Return {model_name -> RGB} for every canonical name and alias, using the
    seaborn palette resolved at call time (pass an explicit ``palette`` to override)."""
    pal = palette if palette is not None else sns.color_palette()
    colors = {name: pal[i] for name, i in _INDEX.items()}
    for alias, canon in _ALIAS.items():
        colors[alias] = colors[canon]
    return colors


def color_for(model, palette=None):
    """Color for a single model name (canonical or alias)."""
    return model_colors(palette)[model]
