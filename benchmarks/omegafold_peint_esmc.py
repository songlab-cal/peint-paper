"""One-off OmegaFold folding of PEINT-ESM-C simulated leaves (revision 2).

Folds each family's PEINT progressive + single-shot leaves with OmegaFold, writing PDBs to the
SAME path ``generate_all_results`` uses:

    <out>/omegafold/<family>/<MODEL_KEY>/structures/

so a later ``generate_all_results ... --include_plddt`` (no ``--use_af2``) cache-hits these and
only runs the pLDDT scoring + aggregation -- no re-folding, and no Real/classical folding
(reused from revision 1 for the comparison).

The leaf subset is chosen by calling the REAL ``_subsample_selection`` + ``generate_tree_split``
(subsample 30, matching rev1's ``--subsample_msa_size 30``), so the exact same 30 leaves are
folded -- seeded by md5(family+root), PEINT uses ``subsampled_a`` (seq1 + 29 from split-A). We
fold those leaves' RAW simulated sequences (gap-stripped by OmegaFold), so no alignment is
needed and this can run before the classical pipeline finishes. GPU + omegafold on PATH
(protevo-env). Per (family, model) is cherryml-cached -> resumes after preemption, shards
cleanly over GPUs (disjoint families -> shared out dir, no collision).
"""

import argparse
import json
import os
from argparse import Namespace

from protevo import caching as pc
from protevo.utils import read_msa
from protevo.io import write_msa
from paper.splits import generate_tree_split
from paper.structure_prediction import generate_omegafold_predictions
from benchmarks.generate_all_results import _subsample_selection
import paper_config as cfg

SIM = str(cfg.SIM_ROOT)
TREE_DIR = f"{SIM}/trees_newick"        # same --tree_dir the ESM-C sims/generate_all_results use
ROOT_SEQ_DIR = f"{SIM}/root_sequences"
R2 = str(cfg.RESULTS_R2_DIR)
MODELS = {  # raw sim dir -> generate_all_results model key (dir name); both use subsampled_a
    "peint_progressive_unfiltered": "PEINT (Progressive)",
    "peint_single_shot_unfiltered": "PEINT (Single Shot)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families_json", required=True)
    ap.add_argument("--subsample_msa_size", type=int, default=30)
    ap.add_argument("--max_families", type=int, default=0)
    ap.add_argument("--progressive_only", action="store_true",
                    help="Fold only PEINT (Progressive); skip single-shot (folds poorly, ~2x faster).")
    args = ap.parse_args()

    families = json.load(open(args.families_json))["families"]
    if args.max_families > 0:
        families = families[: args.max_families]
    pc.set_cache_dir(f"{R2}/simulations/omegafold_peint/_cache")
    pc.set_dir_levels(3)

    sub_args = Namespace(subsample_msa_size=args.subsample_msa_size,
                         root_sequences_dir=ROOT_SEQ_DIR)
    models = MODELS
    if args.progressive_only:
        models = {k: v for k, v in MODELS.items() if k == "peint_progressive_unfiltered"}
    prep_base = f"{R2}/simulations/omegafold_peint/subsampled_inputs"
    done = 0
    for fam in families:
        # Exact generate_all_results subsample: PEINT models keep subsampled_a.
        tree_split = generate_tree_split(TREE_DIR, fam)
        subsampled_a, _ = _subsample_selection(sub_args, fam, tree_split)
        keep = list(subsampled_a)
        for sim_dir, model_key in models.items():
            src = f"{R2}/simulations/{sim_dir}/{fam}.txt"
            if not os.path.exists(src):
                continue
            seqs = read_msa(src)
            prep_dir = os.path.join(prep_base, sim_dir)
            os.makedirs(prep_dir, exist_ok=True)
            prep_path = os.path.join(prep_dir, f"{fam}.txt")
            if not os.path.exists(prep_path):
                write_msa({k: seqs[k] for k in keep if k in seqs}, prep_path)
            out_struct = f"{R2}/omegafold/{fam}/{model_key}/structures"
            os.makedirs(out_struct, exist_ok=True)
            generate_omegafold_predictions(
                sequences_dir=prep_dir, family=fam, output_structures_dir=out_struct,
            )
            done += 1
    print(f"DONE — folded {done} (family, model) pairs over {len(families)} families")


if __name__ == "__main__":
    main()
