"""Tree-based splits of a family's leaves into two halves.

Shared by the conservation figure and (once migrated) the structure / 3Di
benchmarks. Ported from the model repo's ``benchmarks/utils.py``; the tree
primitives themselves come from the installed ``peint``.
"""

import os
import re
from typing import Dict, List

from ete3 import Tree as Ete3Tree
from peint.datasets._datasets import find_optimal_edge_split, split_tree_on_edge
from peint.io import read_tree

SPLIT_A = "A"
SPLIT_B = "B"

# Model keys for the empirical data. "Real" is split A; "Real (other split)" is the same
# empirical data restricted to split B, used as the baseline any model is measured against.
# Defined here so the JSD, sequence and structure modules cannot drift apart on the spelling.
REAL = "Real"
REAL_OTHER_SPLIT = "Real (other split)"

# cherryml node-list files start with "<num_nodes> nodes"; newick files do not.
_CHERRYML_HEADER = re.compile(r"^\d+\s+nodes\s*$")


def _split_with_seq1_as_a(nodes_a: List[str], nodes_b: List[str]) -> Dict[str, List[str]]:
    """Return the split with the seq1-containing side as A (the reference side)."""
    if "seq1" in nodes_a:
        return {SPLIT_A: nodes_a, SPLIT_B: nodes_b}
    return {SPLIT_A: nodes_b, SPLIT_B: nodes_a}


def _tree_split_cherryml(tree_path: str) -> Dict[str, List[str]]:
    tree = read_tree(tree_path)
    tree_a, tree_b = split_tree_on_edge(tree, find_optimal_edge_split(tree))
    return _split_with_seq1_as_a(tree_a.nodes(), tree_b.nodes())


def _tree_split_newick(tree_path: str) -> Dict[str, List[str]]:
    """Balanced-edge split for a newick tree, mirroring ``find_optimal_edge_split``.

    The most balanced edge is the (parent, child) whose child-subtree holds a leaf
    count closest to half; removing it partitions every node into the child subtree
    and the rest. Node lists (leaves + internals) match the cherryml path's semantics.
    """
    tree = Ete3Tree(tree_path, format=1)
    n_leaves = len(tree)

    best_node, best_error = None, float(n_leaves)
    for node in tree.traverse():
        if node.is_root():
            continue
        error = abs(len(node) - n_leaves / 2)  # ete3 len(node) == leaves under node
        if error < best_error:
            best_error, best_node = error, node
    if best_node is None:
        raise ValueError(f"Could not find a split edge for tree {tree_path}")

    inside_ids = {id(n) for n in best_node.traverse()}
    inside = [n.name for n in best_node.traverse()]
    outside = [n.name for n in tree.traverse() if id(n) not in inside_ids]
    return _split_with_seq1_as_a(inside, outside)


def generate_tree_split(tree_dir: str, family: str) -> Dict[str, List[str]]:
    """Split a family's tree on its most balanced edge, returning leaf names per side.

    Accepts either a cherryml node-list tree or a newick tree (the simulators read
    newick, so shipped trees are newick; the cherryml path is kept for legacy inputs).

    Split A is by convention the side holding ``seq1``, the experimental reference
    structure, so the "other split" used as a real-data baseline is always the
    half further from that reference.
    """
    tree_path = os.path.join(tree_dir, family + ".txt")
    with open(tree_path) as f:
        first_line = f.readline().strip()

    if _CHERRYML_HEADER.match(first_line):
        return _tree_split_cherryml(tree_path)
    return _tree_split_newick(tree_path)


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
