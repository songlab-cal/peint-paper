"""3Di conservation JSD across all models, with ESM-C PEINT (rev2) folded in.

The 3Di analog of the amino-acid "JSD all models" figure. For each family it computes the
mean Jensen-Shannon divergence between each model's per-site 3Di distribution and the real
3Di distribution on conserved sites, then draws one box per model.

Two kinds of inputs, mirroring how ``figures/figure3_conservation.py`` folds ESM-C into the
amino-acid pipeline (``collect_extra_revision_jsd`` / ``EXTRA_REVISION_MODELS``):

  * Standard models (rev1) already have 3Di annotations on disk (one dir per model, each with
    ``<family>_aligned.txt``), produced by ``benchmarks/generate_all_results.py``'s
    ``process_3di_pipeline``. We reuse those dirs as-is and score every model against the rev1
    real 3Di.
  * ESM-C (rev2) has NO 3Di yet, so we generate it here from rev2's aligned sequences
    (``mafft_add/peint_progressive_dir``) and rev2's aligned real (``mafft_add/old_sequences``),
    then score ESM-C against ITS OWN rev2 real 3Di. JSD-vs-real is a frame-consistent per-family
    scalar and the real data is identical across revisions, so the two are comparable and fold
    into one boxplot — exactly the argument the amino-acid figure already relies on.

Everything downstream is the shared single-source machinery: 3Di generation is
``paper.threedi.generate_3di_annotations`` (the same call the general driver uses), JSD is
``paper.jsd.family_jsd(..., foldseek_states=True)``, and the boxplot is
``figures.figure3_conservation.plot_jsd_boxplot``. No function is patched; this script only
wires provided directories into those calls.

GPU: the ESM-C 3Di generation step needs CUDA (ProstT5). The JSD/plotting steps are CPU-only,
so once the rev2 3Di dirs exist a rerun with ``--skip-3di-generation`` is CPU-only.

Run from the repo root, e.g.::

    python -m benchmarks.threedi_jsd_all_models --num-processes 1
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import pandas as pd
from tqdm import tqdm

from cherryml import caching as cherryml_caching
from peint import caching as peint_caching

from paper.jsd import REAL, REAL_OTHER_SPLIT, family_jsd
from paper.splits import generate_tree_split
# Imported lazily inside the function that generates 3Di states: it pulls in ProstT5 via
# `transformers`, which the replot path (--from-csv) has no need of. Keeping it at module
# level made redrawing a shipped CSV require the whole folding stack.
from figures.figure3_conservation import plot_jsd_boxplot
import paper_config as cfg

# rev1 models that already have 3Di annotations on disk. Names must match BOTH the on-disk
# 3Di subdir (rev1-3di-root/<name>/sequences) AND the boxplot's model keys in
# figures.figure3_conservation.BOXPLOT_MODELS, so plot_jsd_boxplot picks them up and relabels.
STANDARD_MODELS: List[str] = [
    "WAG",
    "LG",
    "LG4X",
    "LG+C60",
    "LG+S256",
    "PEINT (Progressive)",
]
# The cross-revision model generated here; its boxplot key (matches BOXPLOT_MODELS).
ESMC_MODEL = "PEINT (ESM-C)"

_R1 = str(cfg.RESULTS_R1_DIR)
_R2 = str(cfg.RESULTS_R2_DIR)


def _threedi_dir(sequences_dir: str) -> Dict[str, str]:
    """Wrap a 3Di sequences dir the way ``family_jsd(foldseek_states=True)`` reads it:
    it looks up ``msa_dir['output_3di_dir']`` and reads ``<family>_aligned.txt`` from there."""
    return {"output_3di_dir": sequences_dir}


def collect_foldseek_jsd(
    families: List[str],
    msa_dirs: Dict[str, Dict[str, str]],
    conservation_threshold: float,
    desc: str,
) -> pd.DataFrame:
    """Per-family mean 3Di JSD-vs-real for every model in ``msa_dirs``.

    Same shape/skip semantics as ``figure3_conservation.collect_family_jsd`` but on 3Di states
    (``foldseek_states=True``). Rows are families, columns are models (plus ``Real (other
    split)`` derived inside ``family_jsd``). Families whose inputs are missing/unusable are
    skipped and logged, matching the amino-acid collector.
    """
    tree_dir = str(cfg.require(cfg.TREE_DIR))
    rows, skipped = {}, {}
    for family in tqdm(families, desc=desc):
        try:
            tree_split = generate_tree_split(tree_dir, family)
            mean_jsd, _ = family_jsd(
                msa_dirs, family, tree_split, conservation_threshold, foldseek_states=True
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            skipped[family] = f"{type(exc).__name__}: {exc}"
            continue
        rows[family] = mean_jsd
    if not rows:
        raise RuntimeError(
            f"[{desc}] no family produced a 3Di JSD score ({len(skipped)} skipped). "
            f"First failure: {next(iter(skipped.items()), None)}"
        )
    if skipped:
        print(f"[{desc}] skipped {len(skipped)}/{len(families)} families; first few:")
        for family, reason in list(skipped.items())[:5]:
            print(f"  {family}: {reason}")
    return pd.DataFrame.from_dict(rows, orient="index")


def generate_esmc_3di(args, families: List[str]) -> Dict[str, str]:
    """Generate 3Di for rev2 real + rev2 ESM-C from their aligned sequences.

    Returns {REAL: <real 3Di sequences dir>, ESMC_MODEL: <esmc 3Di sequences dir>}. The generator
    is cached (``paper.threedi.generate_3di_annotations``), so reruns are no-ops once populated.
    """
    out = {}
    for name, input_dir in (
        (REAL, args.esmc_real_aligned_dir),
        (ESMC_MODEL, args.esmc_aligned_dir),
    ):
        # Slugless subdir per model under esmc-3di-root, mirroring process_3di_pipeline's
        # "3di/<model>/{sequences,probabilities}" layout so it reads back identically.
        from paper.threedi import generate_3di_annotations  # heavy: ProstT5/transformers
        result = generate_3di_annotations(
            input_dir=cfg.require(input_dir),
            families=families,
            model_checkpoint_path=args.prostt5_checkpoint_path,
            num_processes=args.num_processes,
            input_is_aligned=True,
            include_internals=False,
            output_3di_dir=os.path.join(args.esmc_3di_root, name, "sequences"),
            output_probabilities_dir=os.path.join(args.esmc_3di_root, name, "probabilities"),
        )
        out[name] = result["output_3di_dir"]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families_path",
        default=str(cfg.HELDOUT_FAMILIES_JSON),
        help="JSON with a 'families' list (the simulation eval set).",
    )
    # Standard (rev1) 3Di annotations, already on disk, one <name>/sequences dir per model.
    parser.add_argument("--rev1-3di-root", default=os.path.join(_R1, "3di"),
                        help="Root holding rev1 3Di dirs: <root>/<model>/sequences/<fam>_aligned.txt")
    # rev2 (ESM-C) aligned sequences to annotate, and where the new 3Di lands.
    parser.add_argument("--esmc-aligned-dir", default=os.path.join(_R2, "mafft_add", "peint_progressive_dir"),
                        help="Aligned ESM-C PEINT sequences (rev2 frame).")
    parser.add_argument("--esmc-real-aligned-dir", default=os.path.join(_R2, "mafft_add", "old_sequences"),
                        help="Aligned rev2 real sequences (ESM-C's own real reference).")
    parser.add_argument("--esmc-3di-root", default=os.path.join(_R2, "3di"),
                        help="Where the generated ESM-C + rev2-real 3Di dirs are written.")
    parser.add_argument("--prostt5-checkpoint-path", default=str(cfg.PROSTT5_CACHE_DIR),
                        help="ProstT5 HuggingFace cache directory.")
    parser.add_argument("--conservation-threshold", type=float, default=0.7,
                        help="Residue frequency above which a site counts as conserved "
                             "(matches the amino-acid ESM-C summary and generate_all_results).")
    parser.add_argument("--num-processes", type=int, default=1,
                        help="Processes for 3Di generation (each uses cuda:0; keep at 1 unless "
                             "you have the GPU memory for more).")
    parser.add_argument("--skip-3di-generation", action="store_true",
                        help="Assume rev2 3Di already exists under --esmc-3di-root (CPU-only run).")
    parser.add_argument("--from-csv", "--replot", dest="from_csv", action="store_true",
                        help="Redraw from the shipped figure_data table instead of scoring "
                             "3Di states again. Same plotting code (plot_jsd_boxplot).")
    parser.add_argument("--output-dir", default=str(cfg.FIGURES_DIR / "esmc_summary"),
                        help="Where the 3Di JSD CSV + boxplot are written.")
    parser.add_argument("--max-families", type=int, default=None,
                        help="Cap the number of families (smoke runs). Note: 3Di generation is "
                             "cached per family, so a capped run only annotates that subset.")
    args = parser.parse_args()

    if args.from_csv:
        # Replot path: load the table this script itself writes (shipped as
        # figure_data/esmc_summary/conservation_jsd_3di.csv) and hand it to the SAME
        # plot_jsd_boxplot the recompute path uses -- one figure implementation.
        stem = "conservation_jsd_3di.csv"
        for cand in (Path(cfg.FIGURE_DATA_DIR) / "esmc_summary" / stem,
                     Path(args.output_dir) / stem):
            if cand.exists():
                jsd_df = pd.read_csv(cand, index_col=0)
                print(f"replotting 3Di JSD from {cand} ({len(jsd_df)} families)")
                os.makedirs(args.output_dir, exist_ok=True)
                plot_jsd_boxplot(jsd_df, args.output_dir,
                                 filename_stem="conservation_jsd_3di_boxplot")
                return
        raise SystemExit(
            f"No saved {stem} in {cfg.FIGURE_DATA_DIR}/esmc_summary or {args.output_dir}."
        )

    peint_caching.set_cache_dir("_cache_peint")
    peint_caching.set_log_level(9)
    peint_caching.set_dir_levels(3)
    cherryml_caching.set_cache_dir("_cache_benchmarking")
    cherryml_caching.set_log_level(9)
    cherryml_caching.set_dir_levels(3)

    with open(cfg.require(args.families_path)) as f:
        families = json.load(f)["families"]
    if args.max_families is not None:
        families = families[: args.max_families]
    print(f"{len(families)} families")

    # 1) Generate 3Di for ESM-C + its rev2 real (GPU), unless already present.
    if args.skip_3di_generation:
        esmc_dirs = {
            REAL: os.path.join(args.esmc_3di_root, REAL, "sequences"),
            ESMC_MODEL: os.path.join(args.esmc_3di_root, ESMC_MODEL, "sequences"),
        }
    else:
        print("Generating 3Di for rev2 real + ESM-C PEINT ...")
        esmc_dirs = generate_esmc_3di(args, families)

    # 2) Standard models: score against the rev1 real 3Di (all in the rev1 frame).
    standard_msa_dirs = {
        REAL: _threedi_dir(os.path.join(args.rev1_3di_root, REAL, "sequences")),
        **{
            model: _threedi_dir(os.path.join(args.rev1_3di_root, model, "sequences"))
            for model in STANDARD_MODELS
        },
    }
    print("Computing 3Di JSD for standard (rev1) models ...")
    std_df = collect_foldseek_jsd(
        families, standard_msa_dirs, args.conservation_threshold, desc="3Di JSD rev1"
    )

    # 3) ESM-C: score against its OWN rev2 real 3Di (rev2 frame).
    esmc_msa_dirs = {
        REAL: _threedi_dir(esmc_dirs[REAL]),
        ESMC_MODEL: _threedi_dir(esmc_dirs[ESMC_MODEL]),
    }
    print("Computing 3Di JSD for ESM-C (rev2) ...")
    esmc_df = collect_foldseek_jsd(
        families, esmc_msa_dirs, args.conservation_threshold, desc=f"3Di JSD {ESMC_MODEL}"
    )

    # 4) Fold ESM-C in as one more boxplot column, then write CSV + boxplot.
    jsd_df = std_df.join(esmc_df[[ESMC_MODEL]])

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "conservation_jsd_3di.csv")
    jsd_df.to_csv(csv_path)
    plot_jsd_boxplot(jsd_df, args.output_dir, filename_stem="conservation_jsd_3di_boxplot")

    print(f"\nWrote 3Di JSD over {len(jsd_df)} families to {args.output_dir}")
    print(f"  {csv_path}")
    print("  conservation_jsd_3di_boxplot.{pdf,png}")
    print("\n3Di JSD median vs real:")
    print(jsd_df.median(numeric_only=True).round(3).to_dict())


if __name__ == "__main__":
    main()
