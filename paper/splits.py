"""Tree-based splits of a family's leaves into two halves.

Shared by the conservation figure and (once migrated) the structure / 3Di
benchmarks. Ported from the model repo's ``benchmarks/utils.py``; the tree
primitives themselves come from the installed ``peint``.
"""

import os
from typing import Dict, List

from protevo.datasets._datasets import find_optimal_edge_split, split_tree_on_edge
from protevo.io import read_tree

SPLIT_A = "A"
SPLIT_B = "B"

# Model keys for the empirical data. "Real" is split A; "Real (other split)" is the same
# empirical data restricted to split B, used as the baseline any model is measured against.
# Defined here so the JSD, sequence and structure modules cannot drift apart on the spelling.
REAL = "Real"
REAL_OTHER_SPLIT = "Real (other split)"


def generate_tree_split(tree_dir: str, family: str) -> Dict[str, List[str]]:
    """Split a family's tree on its most balanced edge, returning leaf names per side.

    Split A is by convention the side holding ``seq1``, the experimental reference
    structure, so the "other split" used as a real-data baseline is always the
    half further from that reference.
    """
    tree = read_tree(os.path.join(tree_dir, family + ".txt"))
    tree_a, tree_b = split_tree_on_edge(tree, find_optimal_edge_split(tree))

    if "seq1" in tree_a.nodes():
        return {SPLIT_A: tree_a.nodes(), SPLIT_B: tree_b.nodes()}
    return {SPLIT_A: tree_b.nodes(), SPLIT_B: tree_a.nodes()}


def filter_msa_based_on_split(
    msa: Dict[str, str],
    tree_split: Dict[str, List[str]],
    split: str,
) -> Dict[str, str]:
    """Restrict an MSA to the sequences on one side of the split."""
    if split not in tree_split:
        raise KeyError(f"Unknown split {split!r}; expected one of {sorted(tree_split)}.")
    members = set(tree_split[split])
    return {name: seq for name, seq in msa.items() if name.strip() in members}
