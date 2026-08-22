"""Canonical model -> color for ALL paper figures — one source of truth.

Every figure (conservation JSD, structure-metrics ECDFs, PCP mutation counts, the VEP
bar charts, ...) must color a given model identically. Historically each figure built
its own palette and they drifted; this module centralizes the mapping so they cannot.

Two families of entry:

* ``_INDEX`` — the simulation baselines, kept as seaborn ``deep`` palette *indices* so
  they match every figure published so far::

      WAG=0  LG=1  PEINT(ESM2)=2  LG4X=3  Real=4  LG+C60=5  LG+S256=6

* ``_HEX`` — the models added in revision 2 and the VEP comparison. These are explicit
  hexes because they need a deliberate light/dark relationship that palette indices
  cannot express: PEINT's two backbones read as one green family (ESM2-150M deep,
  ESM-C lighter) and the 650M tier is orange so the largest model stands apart.

The palette is resolved as ``deep`` by default rather than via a bare
``sns.color_palette()``. Seaborn does not set a palette on import, so a bare call
returns matplotlib's ``tab10`` unless something already ran ``sns.set_theme()`` — which
half the figure scripts do and half do not, so the same model came out in two different
shades depending on which script drew it.
"""

import seaborn as sns

# Canonical model name -> seaborn "deep" palette index.
_INDEX = {
    "WAG": 0,
    "LG": 1,
    "PEINT (ESM2)": 2,      # == PEINT on the ESM2-150M backbone; deep green #55a868
    "LG4X": 3,
    "Real": 4,
    "LG+C60": 5,
    "LG+S256": 6,
}

# Canonical model name -> explicit hex. The three base pLMs are the light partner of
# their PEINT model, so a base/PEINT pair reads as one hue at two lightnesses.
#
# The two PEINT backbones are both green (they are the same model on different
# encoders), so they are separated by HUE and not only by lightness: ESM2-150M keeps the
# canonical blue-leaning green, ESM-C takes a lighter yellow-green. Separating them by
# lightness alone was tried first and failed — in the six-bar VEP chart the light
# partner of one green is indistinguishable from the dark member of the other.
_HEX = {
    "PEINT (ESM-C)": "#8bc34a",       # lighter yellow-green beside PEINT (ESM2)'s #55a868
    "ESM2-150M": "#cfe8d5",           # light partner of PEINT (ESM2)
    "ESM-C 300M": "#e4f0c4",          # light partner of PEINT (ESM-C)
    "ESM2-650M": "#fdd0a2",           # light partner of PEINT (ESM2-650M)
    "PEINT (ESM2-650M)": "#e6550d",   # orange: the largest tier stands apart
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
    # VEP run / release-column spellings of the three base pLMs.
    "ESM2_150M": "ESM2-150M",
    "ESM2 150M": "ESM2-150M",
    "ESMC-300M": "ESM-C 300M",
    "ESM2_650M": "ESM2-650M",
    "ESM2 650M": "ESM2-650M",
    "PEINT (ESM2 650M)": "PEINT (ESM2-650M)",
}

# The (base pLM, PEINT) color pair per frozen backbone, in the VEP display order. Kept
# here so paper.vep and figures.figure5_vep cannot drift from the rest of the figures.
BASE_LM_PAIRS = {
    "ESM2-150M": ("ESM2-150M", "PEINT (ESM2)"),
    "ESM-C 300M": ("ESM-C 300M", "PEINT (ESM-C)"),
    "ESM2-650M": ("ESM2-650M", "PEINT (ESM2-650M)"),
}


def model_colors(palette=None):
    """Return {model_name -> color} for every canonical name and alias.

    Indexed entries use the seaborn ``deep`` palette unless ``palette`` overrides it;
    ``_HEX`` entries are hex strings and ignore the palette. Both forms are accepted
    everywhere matplotlib takes a color.
    """
    pal = palette if palette is not None else sns.color_palette("deep")
    colors = {name: pal[i] for name, i in _INDEX.items()}
    colors.update(_HEX)
    for alias, canon in _ALIAS.items():
        colors[alias] = colors[canon]
    return colors


def color_for(model, palette=None):
    """Color for a single model name (canonical or alias)."""
    return model_colors(palette)[model]


def base_lm_pair_colors(palette=None):
    """{backbone label -> (base pLM color, PEINT color)} for the VEP figures."""
    colors = model_colors(palette)
    return {lm: (colors[base], colors[peint]) for lm, (base, peint) in BASE_LM_PAIRS.items()}
