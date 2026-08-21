"""Central benchmark driver: simulate sequences under every model, then score them.

Produces the results the paper's figures are built from. The pipeline is:

1. Simulate evolution down each family's tree with PEINT (progressive and single-shot).
2. Align everything into one reference frame: MAFFT on the empirical sequences, then
   ``mafft --add`` to place the PEINT sequences in that same frame.
3. Simulate the classical baselines with AliSim on that alignment (WAG, LG, LG4X,
   LG+C60, LG+S256; ``inference`` and ``prior_anchored`` modes).
4. Optionally predict structures, then score: TM-score, conservation (JSD), 3Di
   states, pLDDT.

**Structure prediction comes in two flavours and the paper uses both.** ``--use_af2``
selects AF2Rank (threads each sequence onto the experimental structure, and yields extra
scores: pLDDT, PAE, RMSD, composite); omitting it selects OmegaFold (folds each sequence
de novo, yielding pLDDT only). The choice propagates: results land under a ``results_af2``
or ``results`` subdirectory, and the pLDDT panels come from different functions. AF2Rank
additionally emits a "sites" file recording which alignment columns survived; nothing
currently consumes it, but it is kept as a record of that filtering.

Divergence on conserved sites is Jensen-Shannon, computed by ``paper.jsd`` — the single
JSD implementation shared with the conservation figure. The original called a function
named ``compute_kld`` here, but always with ``use_jsd=True``, so no KL divergence was ever
computed despite the "kld" naming throughout; KL is dropped (it is unbounded when a
conserved residue's replacement falls outside a model's support).

GPU: PEINT simulation, structure prediction and 3Di annotation all need CUDA.

Run from the repo root::

    python -m benchmarks.generate_all_results --families_path ... --include_conservation
"""

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)

import hashlib
import json
import logging
import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd

from cherryml import caching as cherryml_caching
from protevo import caching as protevo_caching
from protevo.simulation import simulate_alisim_evolution, simulate_peint_evolution_down_tree
from protevo.simulation._alisim import (
    MODEL_DEFINITIONS as ALISIM_MODEL_DEFINITIONS,
    PRIOR_MODE_SUPPORTED_MODELS,
    ALISIM_MODES,
)
from protevo.utils import read_msa, write_msa

from paper.alignment import clean_msa, run_mafft, run_mafft_add
from paper.jsd import REAL, REAL_OTHER_SPLIT, family_jsd
from paper.sequence import generate_conservation_plots, generate_3di_certainty_plots
from paper.splits import generate_tree_split
from paper.structure import (
    compute_tm_scores_against_ground_truth,
    generate_af2_score_plots,
    plot_af2_scores_by_model,
    plot_omegafold_plddt,
    plot_omegafold_plddt_by_model,
    plot_tm_score_boxplot,
    plot_tm_score_histogram,
    plot_tm_scores_by_model,
)
from paper.structure_prediction import generate_af2_predictions, generate_omegafold_predictions
from paper.threedi import generate_3di_annotations
from figures.figure3_conservation import plot_jsd_boxplot
import paper_config as cfg

logger = logging.getLogger(__name__)

BASE_SEQUENCE_TYPES = [REAL, REAL_OTHER_SPLIT, "PEINT (Progressive)", "PEINT (Single Shot)"]


def _seq_type_key(model: str, mode: str) -> str:
    """Bare model name for inference mode (preserves existing cache keys + downstream
    pattern-matching); parenthesized form for new prior modes."""
    return model if mode == "inference" else f"{model} ({mode})"


def _frame_suffix(use_af2: bool) -> str:
    """Output-dir suffix distinguishing the two alignment frames.

    AF2Rank runs ``mafft --add --keeplength`` (everything stays in the seq1 reference
    frame); OmegaFold does not. The two frames' alignments — and everything derived from
    them (the AliSim classical sims, 3Di states, subsampled MSAs) — differ, so their
    output dirs must not collide. Because the cache decides hits from output-dir contents
    alone, a shared dir would let the second run silently reuse the first frame's
    alignment. The default (OmegaFold) frame keeps the original unsuffixed paths for cache
    back-compat; the keeplength (AF2Rank) frame gets a distinct suffix. Frame-invariant
    outputs (``mafft``, the PEINT sims) are deliberately left unsuffixed and shared.
    """
    return "_keeplength" if use_af2 else ""


def process_alignment_pipeline(
    empirical_sequences_dir,
    peint_progressive_dir,
    peint_single_shot_dir,
    tree_dir,
    root_seqs_dir,
    families,
    keep_length,
    num_processes,
    alisim_models,
    alisim_modes,
    output_dir=None,
):
    """Put every model's sequences into one shared alignment frame.

    MAFFT aligns the empirical sequences; ``mafft --add`` then places the PEINT sequences
    into that existing alignment, so sites correspond one-to-one across models. AliSim
    baselines are simulated on that same alignment.

    ``keep_length`` must be True on the AF2Rank path: AF2 cannot represent insertions
    relative to the reference structure, so the alignment must not grow new columns.
    """
    # Frame-dependent outputs (mafft_add + AliSim classical sims) are suffixed so the two
    # frames never share a dir; mafft (empirical align) and the PEINT sims are frame-invariant.
    frame_suffix = _frame_suffix(keep_length)

    cleaned_emp_dir = clean_msa(
        data_dir=empirical_sequences_dir,
        families=families,
        chars_to_exclude=("-", "B", "Z", "J", "U", "O"),
        num_processes=num_processes,
    )

    mafft_dir = os.path.join(output_dir, "mafft") if output_dir else None
    aligned_emp_path = run_mafft(
        data_dir=cleaned_emp_dir["output_msa_dir"],
        families=families,
        num_processes=num_processes,
        output_msa_dir=mafft_dir,
    )["output_msa_dir"]

    if peint_single_shot_dir is None:
        peint_dirs = [peint_progressive_dir]
        peint_names = ["peint_progressive"]
    else:
        peint_dirs = [peint_progressive_dir, peint_single_shot_dir]
        peint_names = ["peint_progressive", "peint_single_shot"]

    mafft_add_dirs = run_mafft_add(
        existing_alignment_dir=aligned_emp_path,
        new_sequences_dirs=peint_dirs,
        new_sequences_names=peint_names,
        families=families,
        num_processes=num_processes,
        extra_command_line_args=["--keeplength"] if keep_length else None,
        output_all_sequences_msa_dir=f"{output_dir}/mafft_add{frame_suffix}/all_sequences",
        output_new_sequences_msa_dir=f"{output_dir}/mafft_add{frame_suffix}/new_sequences",
        output_old_sequences_msa_dir=f"{output_dir}/mafft_add{frame_suffix}/old_sequences",
    )
    aligned_emp_path = mafft_add_dirs["output_old_sequences_msa_dir"]
    base_path = os.path.dirname(aligned_emp_path)

    if peint_single_shot_dir is None:
        aligned_peint_progressive_path = mafft_add_dirs["output_new_sequences_msa_dir"]
        aligned_peint_single_shot_path = None
    else:
        # run_mafft_add renamed the outputs per sequence set.
        aligned_peint_progressive_path = os.path.join(base_path, "peint_progressive_dir")
        aligned_peint_single_shot_path = os.path.join(base_path, "peint_single_shot_dir")

    aligned_alisim_paths = {}
    for model in alisim_models:
        for mode in alisim_modes:
            if mode != "inference" and model not in PRIOR_MODE_SUPPORTED_MODELS:
                logger.info(f"Skipping ({model}, {mode}): {model} does not support prior mode")
                continue

            # Slugify for the directory name ("LG+S256" -> "lg_s256"). Inference mode keeps
            # the bare-model directory so existing cache entries still resolve.
            model_slug = model.lower().replace("+", "_")
            path_slug = model_slug if mode == "inference" else f"{model_slug}_{mode}"

            model_path = os.path.join(output_dir, f"simulations/{path_slug}{frame_suffix}") if output_dir else None
            aligned_alisim_paths[_seq_type_key(model, mode)] = simulate_alisim_evolution(
                tree_dir=tree_dir,
                msa_dir=aligned_emp_path,
                evolutionary_model=model,
                mode=mode,
                families=families,
                num_processes=num_processes,
                root_seqs_dir=root_seqs_dir,
                insertion_rate=0,
                deletion_rate=0,
                output_msa_dir=model_path,
            )["output_msa_dir"]

    return {
        REAL: aligned_emp_path,
        "PEINT (Progressive)": aligned_peint_progressive_path,
        "PEINT (Single Shot)": aligned_peint_single_shot_path,
        **aligned_alisim_paths,
    }


def process_3di_pipeline(aligned_data_dirs, families, model_checkpoint_path, num_processes, output_dir, frame_suffix=""):
    """Annotate every model's aligned sequences with 3Di structural states.

    ``frame_suffix`` keeps the AF2 (keeplength) frame's 3Di annotations in their own dir,
    since they are computed from the frame-specific aligned sequences.
    """
    return {
        model: generate_3di_annotations(
            input_dir=data_dir,
            families=families,
            model_checkpoint_path=model_checkpoint_path,
            num_processes=num_processes,
            input_is_aligned=True,
            include_internals=False,
            output_3di_dir=f"{output_dir}/3di{frame_suffix}/{model}/sequences",
            output_probabilities_dir=f"{output_dir}/3di{frame_suffix}/{model}/probabilities",
        )
        for model, data_dir in aligned_data_dirs.items()
    }


def _subsample_selection(args, family, tree_split):
    """Deterministically pick which leaves to keep when an MSA is too big to fold.

    Seeded by family name + root sequence so the same subset is chosen no matter how many
    families have been processed before this one.
    """
    leaves_a = sorted(x for x in tree_split["A"] if x.startswith("seq"))
    leaves_b = sorted(x for x in tree_split["B"] if x.startswith("seq"))

    with open(os.path.join(args.root_sequences_dir, family + ".txt")) as f:
        root_sequence = f.readlines()[1].strip()

    seed = int(hashlib.md5((family + root_sequence).encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
    rng = np.random.RandomState(seed)

    # seq1 is the experimental reference structure, so it must always be retained.
    subsampled_a = rng.choice(
        [x for x in leaves_a if x != "seq1"], args.subsample_msa_size - 1, replace=False
    )
    subsampled_a = np.append("seq1", subsampled_a)
    subsampled_b = rng.choice(leaves_b, args.subsample_msa_size, replace=False)
    return subsampled_a, subsampled_b


def predict_structures(args, families, aligned_dirs, sequence_types):
    """Run the selected structure predictor over every family and model.

    Returns three per-family/per-model path maps. ``sites`` and ``scores`` are AF2Rank-only
    (OmegaFold produces neither), so they stay None on the OmegaFold path.
    """
    structures, sites, af2_scores = ({f: {} for f in families} for _ in range(3))

    for family in families:
        empirical_msa = read_msa(os.path.join(aligned_dirs[REAL], family + ".txt"))
        # AF2 needs the empirical seq1 to know which columns the template covers.
        empirical_seq = empirical_msa["seq1"]

        subsample = args.subsample_msa_size != -1 and len(empirical_msa) > args.subsample_msa_size
        if subsample:
            subsampled_a, subsampled_b = _subsample_selection(
                args, family, generate_tree_split(args.tree_dir, family)
            )

        for model in sequence_types:
            seq_dir = aligned_dirs[REAL] if model == REAL_OTHER_SPLIT else aligned_dirs[model]
            msa = read_msa(os.path.join(seq_dir, family + ".txt"))

            if subsample and len(msa) > args.subsample_msa_size:
                keep = subsampled_b if model == REAL_OTHER_SPLIT else subsampled_a
                seq_dir = f"{args.out_path}/subsampled_msas{_frame_suffix(args.use_af2)}/{model}"
                os.makedirs(seq_dir, exist_ok=True)
                subsampled_path = os.path.join(seq_dir, family + ".txt")
                if not os.path.exists(subsampled_path):
                    write_msa({k: v for k, v in msa.items() if k in keep}, subsampled_path)

            if args.use_af2:
                out_path = f"{args.out_path}/af2/{family}/{model}"
                predicted = generate_af2_predictions(
                    sequences_dir=seq_dir,
                    empirical_seq1=empirical_seq,
                    ground_truth_structure_dir=str(cfg.require(cfg.GROUND_TRUTH_STRUCTURE_DIR)),
                    model_mode="alphafold",
                    family=family,
                    output_structures_dir=f"{out_path}/structures",
                    output_scores_dir=f"{out_path}/scores",
                    output_sites_dir=f"{out_path}/sites",
                )
            else:
                out_path = f"{args.out_path}/omegafold/{family}/{model}"
                predicted = generate_omegafold_predictions(
                    sequences_dir=seq_dir,
                    family=family,
                    output_structures_dir=f"{out_path}/structures",
                )

            structures[family][model] = predicted["output_structures_dir"]
            sites[family][model] = predicted.get("output_sites_dir")
            af2_scores[family][model] = predicted.get("output_scores_dir")

    return structures, sites, af2_scores


def main(args):
    protevo_caching.set_cache_dir("_cache_protevo")
    protevo_caching.set_log_level(9)
    protevo_caching.set_dir_levels(3)

    cherryml_caching.set_cache_dir("_cache_benchmarking")
    cherryml_caching.set_log_level(9)
    cherryml_caching.set_dir_levels(3)

    np.random.seed(0)

    with open(cfg.require(args.families_path)) as f:
        fams = json.load(f)
    if "in_family" in fams and "held_out_family" in fams:
        in_fams = fams["in_family"]
        families = in_fams + fams["held_out_family"]
    else:
        families = fams["families"]
        in_fams = families  # no split provided: treat everything as in-family

    print(f"Families to evaluate: {len(families)}")
    training_fams_map = {fam: fam in in_fams for fam in families}

    suffix = "" if args.use_likelihood_filtering else "_unfiltered"
    peint_progressive_sequences_dir = simulate_peint_evolution_down_tree(
        tree_dir=args.tree_dir,
        root_sequences_dir=args.root_sequences_dir,
        families=families,
        model_path=args.peint_checkpoint_path,
        msa_dir=None,
        max_batch_size=args.peint_max_batch_size,
        family_bucket_size=args.peint_family_bucket_size,
        use_likelihood_filtering=args.use_likelihood_filtering,
        output_sequences_dir=f"{args.out_path}/simulations/peint_progressive{suffix}",
    )["output_sequences_dir"]

    peint_single_shot_sequences_dir = simulate_peint_evolution_down_tree(
        tree_dir=args.tree_dir,
        root_sequences_dir=args.root_sequences_dir,
        families=families,
        model_path=args.peint_checkpoint_path,
        msa_dir=None,
        single_shot=True,
        max_batch_size=args.peint_max_batch_size,
        family_bucket_size=args.peint_family_bucket_size,
        use_likelihood_filtering=args.use_likelihood_filtering,
        output_sequences_dir=f"{args.out_path}/simulations/peint_single_shot{suffix}",
    )["output_sequences_dir"]

    sequence_types = BASE_SEQUENCE_TYPES + [
        _seq_type_key(m, mode)
        for m in args.alisim_models
        for mode in args.alisim_modes
        if mode == "inference" or m in PRIOR_MODE_SUPPORTED_MODELS
    ]

    print("Running alignment pipeline...")
    aligned_dirs = process_alignment_pipeline(
        empirical_sequences_dir=args.real_sequences_dir,
        peint_progressive_dir=peint_progressive_sequences_dir,
        peint_single_shot_dir=peint_single_shot_sequences_dir,
        families=families,
        root_seqs_dir=args.root_sequences_dir,
        tree_dir=args.tree_dir,
        keep_length=args.use_af2,
        num_processes=args.num_processes,
        alisim_models=args.alisim_models,
        alisim_modes=args.alisim_modes,
        output_dir=args.out_path,
    )

    if args.include_tmscore or args.include_plddt:
        predicted_structure_paths, filtered_sites_paths, af2_scores_paths = predict_structures(
            args, families, aligned_dirs, sequence_types
        )

    if args.include_3di:
        foldseek_dirs = process_3di_pipeline(
            aligned_data_dirs=aligned_dirs,
            families=families,
            model_checkpoint_path=args.prostt5_checkpoint_path,
            num_processes=args.num_processes,
            output_dir=args.out_path,
            frame_suffix=_frame_suffix(args.use_af2),
        )

    all_jsd, all_jsd_3di = {}, {}
    all_tm_scores = {}
    all_af2_scores, all_omegafold_plddts = {}, {}

    results_output_path = os.path.join(args.out_path, "results_af2" if args.use_af2 else "results")
    os.makedirs(results_output_path, exist_ok=True)

    print(f"Starting benchmarking for {len(families)} families:")
    for family in families:
        tree_split = generate_tree_split(tree_dir=args.tree_dir, family=family)
        viz_path = os.path.join(results_output_path, family)
        os.makedirs(viz_path, exist_ok=True)

        if args.include_tmscore:
            tm_scores_by_model = compute_tm_scores_against_ground_truth(
                family=family,
                predicted_structure_dirs=predicted_structure_paths[family],
                ground_truth_structure_dir=str(cfg.require(cfg.GROUND_TRUTH_STRUCTURE_DIR)),
                sequence_types=sequence_types,
                tree_split=tree_split,
            )
            tm_scores_by_model.to_csv(os.path.join(viz_path, "tmscores.csv"))
            plot_tm_score_histogram(scores=tm_scores_by_model, family=family, output_path=viz_path)
            plot_tm_score_boxplot(scores=tm_scores_by_model, family=family, output_path=viz_path)
            all_tm_scores[family] = tm_scores_by_model.groupby("Model")["TM-score"].mean().to_dict()

        if args.include_conservation:
            generate_conservation_plots(
                msa_dirs=aligned_dirs,
                family=family,
                tree_split=tree_split,
                conservation_threshold=args.conservation_threshold,
                in_family=training_fams_map[family],
                out_path=viz_path,
                foldseek_states=False,
            )
            mean_jsd, per_site = family_jsd(
                msa_dirs=aligned_dirs,
                family=family,
                tree_split=tree_split,
                conservation_threshold=args.conservation_threshold,
            )
            per_site.to_csv(os.path.join(viz_path, "jsd_vs_real_aa.csv"))
            all_jsd[family] = mean_jsd

        if args.include_3di:
            generate_conservation_plots(
                msa_dirs=foldseek_dirs,
                family=family,
                tree_split=tree_split,
                conservation_threshold=args.conservation_threshold,
                in_family=training_fams_map[family],
                out_path=viz_path,
                foldseek_states=True,
            )
            generate_3di_certainty_plots(
                probabilities_dirs=foldseek_dirs,
                family=family,
                in_family=training_fams_map[family],
                out_path=viz_path,
            )
            mean_jsd_3di, per_site_3di = family_jsd(
                msa_dirs=foldseek_dirs,
                family=family,
                tree_split=tree_split,
                conservation_threshold=args.conservation_threshold,
                foldseek_states=True,
            )
            per_site_3di.to_csv(os.path.join(viz_path, "jsd_vs_real_foldseek.csv"))
            all_jsd_3di[family] = mean_jsd_3di

        # AF2Rank reports pLDDT alongside PAE/RMSD/composite; OmegaFold gives pLDDT only.
        if args.include_plddt:
            if args.use_af2:
                all_af2_scores[family] = generate_af2_score_plots(
                    scores_paths=af2_scores_paths[family],
                    scores_to_plot=AF2_SCORES_TO_PLOT,
                    output_path=viz_path,
                )
            else:
                all_omegafold_plddts[family] = plot_omegafold_plddt(
                    structures_paths=predicted_structure_paths[family],
                    output_path=viz_path,
                )

    if args.include_tmscore:
        print("Generating TM-score plots aggregated across all families")
        plot_tm_scores_by_model(
            tm_scores=all_tm_scores,
            training_fams_map=training_fams_map,
            output_path=os.path.join(results_output_path, "tmscore"),
        )


    if args.include_conservation:
        print("Generating conservation JSD plots aggregated across all families")
        if args.include_3di:
            write_jsd_boxplots(all_jsd_3di, training_fams_map, os.path.join(results_output_path, "jsd/3di"))
        write_jsd_boxplots(all_jsd, training_fams_map, os.path.join(results_output_path, "jsd/aa"))

    if args.include_plddt:
        print("Generating pLDDT score plots aggregated across all families")
        if args.use_af2:
            plot_af2_scores_by_model(
                all_scores=all_af2_scores,
                scores_to_plot=AF2_SCORES_TO_PLOT,
                training_fams_map=training_fams_map,
                output_path=os.path.join(results_output_path, "af2_scores"),
            )
        else:
            plot_omegafold_plddt_by_model(
                all_scores=all_omegafold_plddts,
                training_fams_map=training_fams_map,
                output_path=os.path.join(results_output_path, "omegafold_plddt"),
            )


AF2_SCORES_TO_PLOT = ["plddt", "rmsd_io", "pae", "composite"]


def write_jsd_boxplots(all_jsd, training_fams_map, output_path):
    """Aggregate per-family mean JSD into all / in-family / held-out boxplots.

    Reuses the conservation figure's boxplot so the driver and the figure cannot drift.
    """
    jsd_df = pd.DataFrame.from_dict(all_jsd, orient="index")
    if jsd_df.empty:
        print(f"No JSD scores to plot for {output_path}; skipping.")
        return

    os.makedirs(output_path, exist_ok=True)
    jsd_df.to_csv(os.path.join(output_path, "jsd_by_family.csv"))

    in_family = jsd_df.loc[[f for f in jsd_df.index if training_fams_map[f]]]
    held_out = jsd_df.loc[[f for f in jsd_df.index if not training_fams_map[f]]]

    plot_jsd_boxplot(jsd_df, output_path, filename_stem="all")
    if not in_family.empty:
        plot_jsd_boxplot(in_family, output_path, filename_stem="in_family")
    if not held_out.empty:
        plot_jsd_boxplot(held_out, output_path, filename_stem="held_out")


if __name__ == "__main__":
    parser = ArgumentParser(description=__doc__)

    # Benchmark flags
    parser.add_argument("--include_conservation", action="store_true", help="Include conservation benchmark")
    parser.add_argument("--include_tmscore", action="store_true", help="Include TM-score benchmark")
    parser.add_argument("--include_3di", action="store_true", help="Include 3Di benchmark")
    parser.add_argument("--include_plddt", action="store_true", help="Include pLDDT benchmark")

    # Required inputs
    parser.add_argument("--families_path", type=str, help="Path to file listing families to evaluate")
    parser.add_argument("--real_sequences_dir", type=str, help="Path to directory of real sequences")
    parser.add_argument("--tree_dir", type=str, default=str(cfg.TREE_DIR), help="Path to tree split directory")
    parser.add_argument(
        "--peint_checkpoint_path",
        type=str,
        default=str(cfg.PEINT_CHECKPOINT),
        help="PEINT checkpoint (defaults to the model shipped with peint)",
    )
    parser.add_argument(
        "--peint_max_batch_size", type=int, default=64,
        help="Max sequences per GPU forward during PEINT simulation. Lower for larger "
             "backbones (e.g. ESM-C) to avoid OOM.",
    )
    parser.add_argument(
        "--peint_family_bucket_size", type=int, default=32,
        help="Families per length-sorted simulation bucket: families are sorted by root "
             "sequence length and simulated in buckets so similar lengths decode together "
             "(better GPU use, no short-family-waits-on-long-family).",
    )
    parser.add_argument(
        "--root_sequences_dir",
        type=str,
        default=str(cfg.ROOT_SEQ_DIR),
        help="Directory containing root sequences of the given trees",
    )

    # Misc.
    parser.add_argument("--out_path", type=str, default=str(cfg.RESULTS_DIR), help="Output directory")
    parser.add_argument(
        "--use_likelihood_filtering", action="store_true",
        help="Use likelihood filtering for generated PEINT sequences",
    )
    parser.add_argument(
        "--subsample_msa_size", type=int, default=-1,
        help="Sequences to subsample from each MSA for structure prediction. -1 uses all",
    )
    parser.add_argument(
        "--use_af2", action="store_true",
        help="Use AF2Rank structure prediction (omit to use OmegaFold). The paper reports both",
    )
    parser.add_argument(
        "--prostt5_checkpoint_path", type=str, default=str(cfg.PROSTT5_CACHE_DIR),
        help="ProstT5 HuggingFace cache directory",
    )
    parser.add_argument("--conservation_threshold", type=float, default=0.7, help="Conservation site threshold")
    parser.add_argument("--num_processes", type=int, default=1, help="Number of parallel processes to use")
    parser.add_argument(
        "--alisim_models", nargs="+", default=["WAG", "LG"],
        choices=list(ALISIM_MODEL_DEFINITIONS.keys()),
        help="AliSim baseline models. Each appears as a sequence type in all downstream benchmarks.",
    )
    parser.add_argument(
        "--alisim_modes", nargs="+", default=["inference"], choices=list(ALISIM_MODES),
        help="AliSim simulation modes. inference: model parameters fit to the empirical MSA via "
             "-s (all models). prior_anchored: empirical root simulated down the tree under the "
             "prior (no -s, no per-site posterior leak); the root gap pattern is overlaid on every "
             "leaf so output stays in the empirical alignment frame (LG+C60 / LG+S256 only). The "
             "cartesian product of models x modes is enumerated; unsupported pairs are skipped.",
    )

    main(parser.parse_args())
