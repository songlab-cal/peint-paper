"""Structure-based benchmarks: TM-scores and predictor score plots.

Ported from the model repo's ``benchmarks/structure.py``. This module consumes what
``paper.structure_prediction`` produces — OmegaFold structures, or AF2Rank scored
structures — and turns them into the figures and CSVs the paper reports.

Contact-map analysis is deliberately absent: it is not part of the paper, so the
predicted-vs-experimental contact functions were removed along with the contact
generation that fed them.

Both predictors are kept, and their score plots are separate entry points:
``plot_omegafold_plddt`` / ``plot_omegafold_plddt_by_model`` for OmegaFold pLDDT, and
``generate_af2_score_plots`` / ``plot_af2_scores_by_model`` for the richer AF2Rank
metrics. The TM-score functions serve both.

Deviations from the original, all reported in the port: the TM-align binary comes from
``paper_config`` and is resolved on first use rather than at import time (so this module
imports fine without it); ``sequence_types`` is a required argument, since the
module-level default was a second, stale source of truth for the model list; the caching
layer's marker files are filtered out of the OmegaFold structure listing (they were being
parsed as PDBs and yielding NaN pLDDT rows); and the two silent skips in
``generate_af2_score_plots`` now raise.
"""

import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser

from paper.splits import REAL, REAL_OTHER_SPLIT, SPLIT_A, SPLIT_B
from paper.structure_prediction import parse_tm_align

import paper_config as cfg


# Markers the caching layer drops into every output dir; they are not structures.
CACHING_FILES = ["_unhashed_output_dir.log", "_function_binding.log", "result.txt", "result.success"]


def create_boxplot(df: pd.DataFrame, x: str, y: str, title: str, output_path: str) -> None:
    """Boxplot of ``y`` grouped by ``x``, saved to ``output_path``."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=x, y=y, data=df)
    plt.title(title)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def compute_tm_scores_against_ground_truth(
    family: str,
    predicted_structure_dirs: Dict[str, str],
    ground_truth_structure_dir: str,
    tree_split: Dict[str, List[str]],
    sequence_types: Sequence[str],
) -> pd.DataFrame:
    """TM-align every model's predicted structures against the family's experimental structure.

    Real sequences are scored on both sides of the tree split — split A as ``Real``, split B
    as ``Real (other split)`` — while every other model is scored on split A only.

    Args:
        family: Protein family name.
        predicted_structure_dirs: model name -> directory of predicted structures.
        ground_truth_structure_dir: Directory holding ``<family>.pdb``.
        tree_split: Split A / split B leaf names, from ``paper.splits.generate_tree_split``.
        sequence_types: Model names to evaluate.

    Returns:
        TM-score results, one row per (model, sequence).
    """
    if REAL not in predicted_structure_dirs:
        raise KeyError(
            f"predicted_structure_dirs must contain a {REAL!r} entry; "
            f"got {sorted(predicted_structure_dirs)}."
        )

    tmalign = str(cfg.require(cfg.TMALIGN_PATH))

    # "Real (other split)" is derived from "Real" via split B, so it is not iterated over.
    models = [m for m in sequence_types if m != REAL_OTHER_SPLIT]
    all_scores_ground_truth = {model: {} for model in sequence_types}
    sequences = [
        x.strip().split(".")[0]
        for x in os.listdir(predicted_structure_dirs[REAL])
        if x not in CACHING_FILES
    ]

    for seqid in sequences:
        real_pdb_path = f"{ground_truth_structure_dir}/{family}.pdb"
        for model in models:
            file_ext = (
                ".txt"
                if os.path.exists(os.path.join(predicted_structure_dirs[model], seqid + ".txt"))
                else ".pdb"
            )
            model_pdb_path = os.path.join(predicted_structure_dirs[model], seqid + file_ext)

            result = subprocess.run([tmalign, real_pdb_path, model_pdb_path], capture_output=True)
            scores = parse_tm_align(result.stdout)

            if model == REAL:
                if seqid in tree_split[SPLIT_A]:
                    target = all_scores_ground_truth[REAL]
                else:
                    if seqid not in tree_split[SPLIT_B]:
                        raise ValueError(
                            f"Sequence {seqid} for {family} isn't found in split A or split B"
                        )
                    target = all_scores_ground_truth[REAL_OTHER_SPLIT]
            else:
                if seqid not in tree_split[SPLIT_A]:
                    continue
                target = all_scores_ground_truth[model]

            target.setdefault(seqid, {}).update(scores)

    data = []
    for model, seq_scores in all_scores_ground_truth.items():
        for seqid, scores_dict in seq_scores.items():
            data.append({"Model": model, "SeqID": seqid, **scores_dict})

    return pd.DataFrame(data)


def plot_omegafold_plddt(
    structures_paths: Dict[str, str],
    output_path: str,
) -> Dict[str, Dict[str, float]]:
    """Sequence-level OmegaFold pLDDT per model: writes ``plddt.csv`` and ``plddt.png``.

    Args:
        structures_paths: model name -> directory of OmegaFold PDBs.
        output_path: Directory for the CSV and boxplot.

    Returns:
        model name -> {seqid -> mean pLDDT}.
    """
    plddts = {model: {} for model in structures_paths}
    parser = PDBParser(QUIET=True)

    for model, structures_path in structures_paths.items():
        for fpath in os.listdir(structures_path):
            if fpath in CACHING_FILES:
                continue
            seqid = fpath.split(".")[0]
            structure = parser.get_structure("protein", os.path.join(structures_path, fpath))

            residue_b_factors = []
            for prediction in structure:
                for chain in prediction:
                    for residue in chain:
                        # OmegaFold stores pLDDT as a per-residue B-factor.
                        residue_b_factors.append(residue.child_list[0].get_bfactor())

            if not residue_b_factors:
                raise ValueError(
                    f"No residues parsed from {os.path.join(structures_path, fpath)}; "
                    f"expected an OmegaFold PDB."
                )

            plddts[model][seqid] = np.mean(residue_b_factors)

    data = [
        [model, seqid, plddt]
        for model, seqid_dict in plddts.items()
        for seqid, plddt in seqid_dict.items()
    ]
    df = pd.DataFrame(data, columns=["Model", "SeqID", "pLDDT"])
    df.to_csv(os.path.join(output_path, "plddt.csv"))

    save_path = os.path.join(output_path, "plddt.png")
    create_boxplot(df=df, x="Model", y="pLDDT", title="Boxplot of pLDDT", output_path=save_path)

    return plddts


def plot_tm_score_histogram(scores: pd.DataFrame, family: str, output_path: str) -> None:
    """Per-model TM-score histogram for one family."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    score_columns = ["TM-score"]
    for score_kw in score_columns:
        save_path = os.path.join(output_path, f"{score_kw}_hist.png")
        plt.figure(figsize=(8, 6))
        sns.histplot(data=scores, x=score_kw, hue="Model", alpha=0.5)
        plt.title(f"{family} Histogram of {score_kw}")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_tm_score_boxplot(scores: pd.DataFrame, family: str, output_path: str) -> None:
    """Per-model TM-score boxplot for one family."""
    score_columns = ["TM-score"]
    for score_kw in score_columns:
        save_path = os.path.join(output_path, f"{score_kw}_boxplot.png")
        create_boxplot(
            df=scores,
            x="Model",
            y=score_kw,
            title=f"{family} Boxplot of {score_kw}",
            output_path=save_path,
        )


def plot_tm_scores_by_model(
    tm_scores: Dict[str, Dict[str, float]],
    training_fams_map: Dict[str, bool],
    output_path: str,
) -> None:
    """Per-model TM-score boxplots aggregated over all / in-family / held-out families.

    Args:
        tm_scores: family -> {model -> mean TM-score}, from
            ``compute_tm_scores_against_ground_truth``.
        training_fams_map: family -> whether it was in the training set.
        output_path: Directory to save under.
    """
    in_scores = defaultdict(list)
    out_scores = defaultdict(list)
    all_scores = defaultdict(list)

    for family, score_dict in tm_scores.items():
        for model, score in score_dict.items():
            all_scores[model].append(score)
            if training_fams_map[family]:
                in_scores[model].append(score)
            else:
                out_scores[model].append(score)

    def plot(scores_dict, title, save_path):
        rows = [
            {"Model": model, "TM-score": score}
            for model, scores in scores_dict.items()
            for score in scores
        ]
        df = pd.DataFrame(rows)

        if df.empty:
            print(f"Data for {title} was empty — no plot generated!")
            return

        df.to_csv(Path(save_path).with_suffix(".csv"))
        create_boxplot(df=df, x="Model", y="TM-score", title=title, output_path=save_path)

    os.makedirs(output_path, exist_ok=True)

    plot(in_scores, "In-Family TM-score by Model", f"{output_path}/in_family.png")
    plot(out_scores, "Held-Out TM-score by Model", f"{output_path}/held_out.png")
    plot(all_scores, "All Families TM-score by Model", f"{output_path}/all.png")


def generate_af2_score_plots(
    scores_paths: Dict[str, str],
    scores_to_plot: Sequence[str],
    output_path: str,
) -> Dict[str, Dict[str, list]]:
    """Per-model AF2Rank score boxplots for one family, one figure per score type.

    Args:
        scores_paths: model name -> directory holding AF2Rank's ``result.txt``.
        scores_to_plot: Score columns to plot (the paper uses plddt, rmsd_io, pae, composite).
        output_path: Directory to save under.

    Returns:
        model name -> {score type -> values}.
    """
    os.makedirs(output_path, exist_ok=True)

    scores = {model: {} for model in scores_paths}
    for model, path in scores_paths.items():
        result_path = os.path.join(path, "result.txt")
        if not os.path.exists(result_path):
            raise FileNotFoundError(
                f"No AF2Rank scores for model {model!r} at {result_path}. "
                f"Run the AF2 predictions for this family first."
            )
        df = pd.read_csv(result_path)
        missing = [s for s in scores_to_plot if s not in df.columns]
        if missing:
            raise KeyError(
                f"AF2Rank scores for model {model!r} at {result_path} are missing "
                f"columns {missing}; found {list(df.columns)}."
            )
        for score_type in scores_to_plot:
            scores[model][score_type] = df[score_type].dropna().tolist()

    for score_type in scores_to_plot:
        data = [
            {"Model": model, score_type: val}
            for model, model_scores in scores.items()
            for val in model_scores[score_type]
        ]
        df = pd.DataFrame(data)

        save_path = os.path.join(output_path, f"{score_type}.png")
        create_boxplot(
            df=df, x="Model", y=score_type, title=f"Boxplot of {score_type}", output_path=save_path
        )

    return scores


def plot_af2_scores_by_model(
    all_scores: Dict[str, Dict[str, Dict[str, list]]],
    scores_to_plot: Sequence[str],
    training_fams_map: Dict[str, bool],
    output_path: str,
) -> None:
    """Per-model AF2Rank boxplots aggregated over all / in-family / held-out families.

    Args:
        all_scores: family -> model -> {score type -> values}, from
            ``generate_af2_score_plots``.
        scores_to_plot: Score types to include.
        training_fams_map: family -> whether it was in the training set.
        output_path: Directory to save under.
    """
    os.makedirs(output_path, exist_ok=True)

    def plot(scores_dict, score_type, title, save_path):
        rows = [
            {"Model": model, score_type: val}
            for model, score_list in scores_dict.items()
            for val in score_list
        ]
        df = pd.DataFrame(rows)

        if df.empty:
            print(f"Data for {title} was empty — no plot generated!")
            return

        create_boxplot(df=df, x="Model", y=score_type, title=title, output_path=save_path)

    for score_type in scores_to_plot:
        in_scores = defaultdict(list)
        out_scores = defaultdict(list)
        all_fam_scores = defaultdict(list)

        for family, score_dict in all_scores.items():
            for model, scores in score_dict.items():
                all_fam_scores[model].extend(scores[score_type])
                if training_fams_map[family]:
                    in_scores[model].extend(scores[score_type])
                else:
                    out_scores[model].extend(scores[score_type])

        plot(in_scores, score_type, f"In-Family {score_type} by Model",
             f"{output_path}/in_family_{score_type}.png")
        plot(out_scores, score_type, f"Held-Out {score_type} by Model",
             f"{output_path}/held_out_{score_type}.png")
        plot(all_fam_scores, score_type, f"All Families {score_type} by Model",
             f"{output_path}/all_{score_type}.png")


def plot_omegafold_plddt_by_model(
    all_scores: Dict[str, Dict[str, Dict[str, float]]],
    training_fams_map: Dict[str, bool],
    output_path: str,
) -> None:
    """Per-model OmegaFold pLDDT boxplots aggregated over all / in-family / held-out families.

    Also writes every per-sequence score to ``all_plddt.csv``.

    Args:
        all_scores: family -> model -> {seqid -> pLDDT}, from ``plot_omegafold_plddt``.
        training_fams_map: family -> whether it was in the training set.
        output_path: Directory to save under.
    """
    os.makedirs(output_path, exist_ok=True)

    def plot(scores_dict, title, save_path):
        rows = [
            {"Model": model, "pLDDT": val}
            for model, values in scores_dict.items()
            for val in values
        ]
        df = pd.DataFrame(rows)

        if df.empty:
            print(f"Data for {title} was empty — no plot generated!")
            return

        create_boxplot(df=df, x="Model", y="pLDDT", title=title, output_path=save_path)

    in_scores = defaultdict(list)
    out_scores = defaultdict(list)
    all_fam_scores = defaultdict(list)

    for family, score_dict in all_scores.items():
        for model, scores in score_dict.items():
            all_fam_scores[model].extend(scores.values())
            if training_fams_map[family]:
                in_scores[model].extend(scores.values())
            else:
                out_scores[model].extend(scores.values())

    plot(in_scores, "In-Family pLDDT by Model", f"{output_path}/in_family_pLDDT.png")
    plot(out_scores, "Held-Out pLDDT by Model", f"{output_path}/held_out_pLDDT.png")
    plot(all_fam_scores, "All Families pLDDT by Model", f"{output_path}/all_pLDDT.png")

    data = [
        [family, model, seqid, plddt]
        for family, model_dict in all_scores.items()
        for model, seqid_dict in model_dict.items()
        for seqid, plddt in seqid_dict.items()
    ]
    df = pd.DataFrame(data, columns=["family", "model", "seqid", "plddt"])
    df.to_csv(f"{output_path}/all_plddt.csv")
