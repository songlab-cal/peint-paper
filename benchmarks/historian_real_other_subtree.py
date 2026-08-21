"""Historian reconstruction on the OTHER (complement) subtree of Real.

real_msa_historian reconstructs one half of each family's optimal-edge split (the eval
subtree). This runs the SAME historian pipeline on the COMPLEMENT half, giving a
real-vs-real baseline: how much do the two subtrees of the same real family differ?

Robustly picks the complement: derive the eval subtree exactly as _prep_seqs_for_historian
does (keep the half NOT containing the root-sequence leaf), then keep the OTHER half. Real
has leaves only -> no dummy nodes (historian infers ancestors), matching the rev1 real
historian: MAFFT guide -> historian recon with -allspan -band 40 -refine. Every step is
cached, so it shards over epurdom and resumes after preemption.
"""

import argparse
import json
import os

from protevo import caching as pc
from protevo.io import read_tree, write_msa
from protevo.utils import read_msa
from protevo.datasets._datasets import find_optimal_edge_split, split_tree_on_edge
from paper.alignment import run_mafft
from paper.historian import run_historian

SIM = ("/scratch/users/akoehl/old/protein-evolution/local_data/simulation/"
       "final_simulation_512_leaves/ratio_0-1_nucleus_1-0")
TREE_DIR = f"{SIM}/trees"            # cherryml node-list trees (read_tree)
MSA_DIR = f"{SIM}/empirical_msas"    # real leaves
ROOT_DIR = f"{SIM}/root_sequences"
OUT = "/scratch/users/akoehl/protein-evolution/local_data/results_revision1/simulations/real_other_subtree"
HISTORIAN = "/scratch/users/akoehl/peint-paper/historian/bin/historian"


def prep(families, seq_out, tree_out):
    """Filter each real family to its COMPLEMENT subtree (leaves) + write that subtree's tree."""
    os.makedirs(seq_out, exist_ok=True)
    os.makedirs(tree_out, exist_ok=True)
    for fam in families:
        sp, tp = f"{seq_out}/{fam}.txt", f"{tree_out}/{fam}.txt"
        if os.path.exists(sp) and os.path.exists(tp):
            continue
        tree = read_tree(f"{TREE_DIR}/{fam}.txt")
        msa = read_msa(f"{MSA_DIR}/{fam}.txt")
        rs_name = list(read_msa(f"{ROOT_DIR}/{fam}.txt"))[0]
        edge = find_optimal_edge_split(tree)
        ta, tb = split_tree_on_edge(tree, edge)
        eval_keep = tb if rs_name in ta.leaves() else ta   # what real_msa_historian kept
        other = ta if eval_keep is tb else tb              # the complement half
        nodes = set(other.nodes())
        write_msa({k: v for k, v in msa.items() if k in nodes}, sp)
        ete = other.to_ete3()
        # Historian requires a strictly binary tree. Splitting on an edge leaves a unary
        # stub node on the complement side ("node ... has 1 child"). prune keeps every leaf
        # and collapses redundant single-child internals (preserving branch lengths); then
        # drop a unary root if one remains. (Scan of all 545: only unary stubs, no polytomies.)
        ete.prune([l.name for l in ete.get_leaves()], preserve_branch_length=True)
        while len(ete.children) == 1:
            ete = ete.children[0]
            ete.up = None
        nwk = ete.write(format=1)
        if nwk.endswith(";"):
            nwk = nwk[:-1] + f"{ete.name};"
        with open(tp, "w") as f:
            f.write(nwk)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families_json", required=True)
    ap.add_argument("--num_processes", type=int, default=32)
    ap.add_argument("--band", type=int, default=40)
    ap.add_argument("--max_families", type=int, default=0)
    args = ap.parse_args()

    families = json.load(open(args.families_json))["families"]
    if args.max_families > 0:
        families = families[: args.max_families]
    families = [f for f in families if os.path.exists(f"{MSA_DIR}/{f}.txt")]
    print(f"real other-subtree historian: {len(families)} families (np={args.num_processes})")

    pc.set_cache_dir(f"{OUT}/_cache")
    pc.set_dir_levels(3)
    seq_dir, tree_dir = f"{OUT}/_prep/seqs", f"{OUT}/_prep/trees"

    prep(families, seq_dir, tree_dir)

    guide = run_mafft(data_dir=seq_dir, families=families,
                      num_processes=args.num_processes)["output_msa_dir"]
    recon = run_historian(
        sequences_dir=guide, tree_dir=tree_dir, families=families,
        num_processes=args.num_processes, historian_path=HISTORIAN,
        output_sequences_dir=f"{OUT}/real_other_subtree_historian",
        extra_command_line_args=["-allspan", "-band", str(args.band), "-refine"],
    )["output_sequences_dir"]
    print("RECON:", recon)
    print("DONE")


if __name__ == "__main__":
    main()
