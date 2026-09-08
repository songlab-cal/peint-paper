"""Historian indel-event reconstruction on the ESM-C PEINT *progressive* simulations.

Mirrors the simulated branch of ``figures/figure3_indels.py``, specialized for the
revision-2 ESM-C benchmark:

  * **Progressive only** — single-shot sims have no internal nodes (nothing to
    reconstruct events across); progressive carries all internal + leaf sequences.
  * **Real is skipped** — reuse revision 1's real event counts as the comparison
    baseline (the real data is identical across simulators).
  * **Eval subtrees recycled** — the subtree split is deterministic from the shared
    real tree, so it is identical to revision 1. The sequence-dependent prep still
    runs on the ESM-C sequences: filter each subtree to its nodes, add dummy internal
    nodes (epsilon branches), MAFFT guide alignment, then reconstruct.
  * **historian recon ... -allspan -band 40** — this is the original revision-1 command
    (``-ancseq -output fasta -allspan -band 40 -refine``) MINUS ``-refine``: refinement
    adds large runtime for ~no change here, because the dummy internal nodes with epsilon
    branch lengths already pin the alignment hard. (Refine is off by default.)

CPU/memory heavy -> run on the epurdom partition. Every step is cherryml-cached, so
the job resumes from wherever it left off.
"""

import argparse
import json
import os

from peint import caching as peint_caching
from paper.alignment import run_mafft
from paper.historian import (
    prepare_simulated_vs_real_historian,
    add_dummy_nodes,
    run_historian,
    remove_dummy_nodes_from_historian_output,
    get_all_evolutionary_counts_from_historian_output,
)
import paper_config as cfg

SIM = str(cfg.SIM_ROOT)
R2 = str(cfg.RESULTS_R2_DIR)
HISTORIAN = str(cfg.HISTORIAN_PATH)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--families_json",
        default=str(cfg.HELDOUT_FAMILIES_JSON),
    )
    ap.add_argument("--num_processes", type=int, default=64)
    ap.add_argument("--band", type=int, default=40)
    ap.add_argument("--max_families", type=int, default=0,
                    help="If >0, only process the first N families (for dry runs).")
    ap.add_argument("--refine", action="store_true",
                    help="Add historian's -refine (iterative refinement) to the recon command.")
    args = ap.parse_args()

    prog_dir = f"{R2}/simulations/peint_progressive_unfiltered"
    families = json.load(open(args.families_json))["families"]
    families = [f for f in families if os.path.exists(f"{prog_dir}/{f}.txt")]
    if args.max_families > 0:
        families = families[: args.max_families]
    print(f"Historian on {len(families)} progressive families "
          f"(num_processes={args.num_processes}, band={args.band})")

    base = f"{R2}/simulations/historian_progressive"
    peint_caching.set_cache_dir(f"{base}/_cache")
    peint_caching.set_dir_levels(3)
    N = args.num_processes

    # 1. Pick the eval subtree (deterministic from the real tree) and filter the ESM-C
    #    sims to it. Real outputs are produced too but unused (cheap MSA filtering only).
    prep = prepare_simulated_vs_real_historian(
        simulated_tree_dir=f"{SIM}/trees",
        simulated_sequences_dir=prog_dir,
        root_sequences_dir=f"{SIM}/root_sequences",
        real_tree_dir=f"{SIM}/trees",
        real_sequences_dir=f"{SIM}/empirical_msas",
        families=families,
        num_processes=N,
    )

    # 2. Add dummy internal nodes (epsilon branches) to the simulated subtree.
    dummy = add_dummy_nodes(
        sequences_dir=prep["output_simulated_sequences_dir"],
        tree_dir=prep["output_simulated_tree_dir"],
        families=families,
        num_processes=N,
    )

    # 3. MAFFT guide alignment for historian.
    guide = run_mafft(
        data_dir=dummy["output_sequences_dir"],
        families=families,
        num_processes=N,
    )

    # 4. Historian reconstruction (the heavy step): band 40, no refine.
    recon = run_historian(
        sequences_dir=guide["output_msa_dir"],
        tree_dir=dummy["output_tree_dir"],
        families=families,
        num_processes=N,
        historian_path=HISTORIAN,
        # Original rev-1 command minus -refine (default). --refine adds it back.
        extra_command_line_args=["-allspan", "-band", str(args.band)]
        + (["-refine"] if args.refine else []),
    )

    # 5. Strip the dummy nodes back out.
    nondummy = remove_dummy_nodes_from_historian_output(
        sequences_dir=recon["output_sequences_dir"],
        families=families,
        num_processes=N,
    )

    # 6. Per-family evolutionary event counts (subst / ins / del with lengths).
    counts = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=nondummy["output_sequences_dir"],
        tree_dir=prep["output_simulated_tree_dir"],
        families=families,
        num_processes=N,
    )

    print("EVENTS_DIR:", counts["output_events_dir"])
    print("DONE")


if __name__ == "__main__":
    main()
