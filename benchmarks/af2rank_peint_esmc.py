"""AF2Rank for PEINT-ESM-C simulated leaves (revision 2).

Reuses the existing NON-keeplength mafft-add alignments — generate_af2_predictions trims each
sequence to the empirical seq1's non-gap columns internally (``keep_cols``), i.e. it puts
everything in seq1's frame with seq1 gapless (the AF2 keeplength frame). Going non-keeplength ->
keeplength only drops insertions relative to seq1 (which AF2 can't represent anyway), so it's
strictly evidence-preserving and needs NO re-alignment.

Per family: take the same 30-leaf subsample AF2 uses (subsampled_a = seq1 + 29 from split-A, via
the real _subsample_selection), pull those leaves' ALIGNED sequences from the peint mafft-add
alignment, and score them with AF2Rank against the experimental structure. GPU (protevo-env);
cached per (family, model) so it shards over GPUs and resumes. Output ->
<R2>/af2/<family>/<MODEL_KEY>/{structures,scores,sites}, matching rev1's layout for the ECDF.
"""

import argparse
import json
import os
from argparse import Namespace

from protevo import caching as pc
from protevo.utils import read_msa
from protevo.io import write_msa
from paper.splits import generate_tree_split
from paper.structure_prediction import generate_af2_predictions
from benchmarks.generate_all_results import _subsample_selection
import paper_config as cfg

SIM = ("/scratch/users/akoehl/old/protein-evolution/local_data/simulation/"
       "final_simulation_512_leaves/ratio_0-1_nucleus_1-0")
TREE_DIR = f"{SIM}/trees_newick"
ROOT_SEQ_DIR = f"{SIM}/root_sequences"
R2 = "/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc"
PEINT_ALIGN = f"{R2}/mafft_add/peint_progressive_dir"   # non-keeplength aligned PEINT leaves
REAL_ALIGN = f"{R2}/mafft_add/old_sequences"            # non-keeplength aligned real (has seq1)
MODEL_KEY = "PEINT (Progressive)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families_json", required=True)
    ap.add_argument("--subsample_msa_size", type=int, default=30)
    ap.add_argument("--max_families", type=int, default=0)
    args = ap.parse_args()

    families = json.load(open(args.families_json))["families"]
    if args.max_families > 0:
        families = families[: args.max_families]
    gt = str(cfg.require(cfg.GROUND_TRUTH_STRUCTURE_DIR))
    pc.set_cache_dir(f"{R2}/simulations/af2_peint/_cache")
    pc.set_dir_levels(3)
    sub_args = Namespace(subsample_msa_size=args.subsample_msa_size, root_sequences_dir=ROOT_SEQ_DIR)
    prep = f"{R2}/simulations/af2_peint/subsampled_inputs"
    os.makedirs(prep, exist_ok=True)

    done, skipped = 0, []
    for fam in families:
        pfile, rfile = f"{PEINT_ALIGN}/{fam}.txt", f"{REAL_ALIGN}/{fam}.txt"
        gtpdb = f"{gt}/{fam}.pdb"
        if not (os.path.exists(pfile) and os.path.exists(rfile) and os.path.exists(gtpdb)):
            continue
        real = read_msa(rfile)
        if "seq1" not in real:
            continue
        empirical_seq1 = real["seq1"]
        peint = read_msa(pfile)
        subsampled_a, _ = _subsample_selection(sub_args, fam, generate_tree_split(TREE_DIR, fam))
        sub = {k: peint[k] for k in subsampled_a if k in peint}
        if not sub:
            continue
        write_msa(sub, f"{prep}/{fam}.txt")
        out = f"{R2}/af2/{fam}/{MODEL_KEY}"
        for s in ("structures", "scores", "sites"):
            os.makedirs(f"{out}/{s}", exist_ok=True)
        # Isolate per-family failures (e.g. a GT structure whose residue count doesn't
        # match seq1's non-gap length -> JAX broadcast error) so one bad family logs
        # and is skipped instead of killing the whole shard.
        try:
            generate_af2_predictions(
                sequences_dir=prep,
                ground_truth_structure_dir=gt,
                empirical_seq1=empirical_seq1,
                family=fam,
                model_mode="alphafold",
                output_structures_dir=f"{out}/structures",
                output_scores_dir=f"{out}/scores",
                output_sites_dir=f"{out}/sites",
            )
            done += 1
        except Exception as e:
            skipped.append(fam)
            print(f"SKIP {fam}: {type(e).__name__}: {e}", flush=True)
    print(f"DONE — AF2Rank on {done} families ({len(skipped)} skipped: {skipped})")


if __name__ == "__main__":
    main()
