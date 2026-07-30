"""Structure-based benchmarks: TM-scores, contact maps, and predictor score plots.

Ported from the model repo's ``benchmarks/structure.py``. This module consumes what
``paper.structure_prediction`` produces — OmegaFold structures/contacts, or AF2Rank
scored structures plus its per-family site subset — and turns them into the figures
and CSVs the paper reports.

Both predictors are kept, and their score plots are separate entry points:
``plot_omegafold_plddt`` / ``plot_omegafold_plddt_by_model`` for OmegaFold pLDDT, and
``generate_af2_score_plots`` / ``plot_af2_scores_by_model`` for the richer AF2Rank
metrics. The TM-score and contact-map functions serve both.

Deviations from the original, all reported in the port: the TM-align binary comes from
``paper_config`` and is resolved on first use rather than at import time (so this module
imports fine without it); ``sequence_types`` is a required argument, since the
module-level default was a second, stale source of truth for the model list; the caching
layer's marker files are filtered out of the OmegaFold structure listing (they were being
parsed as PDBs and yielding NaN pLDDT rows); and the two silent skips in
``generate_af2_score_plots`` now raise.
"""

import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio.PDB import PDBParser
from sklearn.metrics import f1_score, precision_score, recall_score

from protevo.utils import read_msa

from paper.splits import REAL, REAL_OTHER_SPLIT, SPLIT_A, SPLIT_B
from paper.structure_prediction import generate_contact_map, parse_tm_align

import paper_config as cfg


DISTANCE_FOR_NONTRIVIAL_CONTACT = 6

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


def msa_to_unaligned_map(msa: Dict[str, str]) -> Dict[str, Dict[int, int]]:
    """Per-sequence map from aligned column index to unaligned residue index (gaps omitted)."""
    unaligned_maps = {}
    for seqid, seq in msa.items():
        non_gap_positions = [i for i, char in enumerate(seq) if char != "-"]
        unaligned_maps[seqid] = {pos: i for i, pos in enumerate(non_gap_positions)}
    return unaligned_maps


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


def generate_predicted_contacts(
    contacts_dir: str,
    msa_dir: str,
    filtered_sites_path,
    family: str,
    tree_split: Dict[str, List[str]],
    split: str,
    include_internals: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate per-sequence predicted contact maps into the alignment's reference frame.

    Counts, for each pair of MSA columns, how many sequences place those residues in contact,
    separately for all contacts and for nontrivial ones (>= 6 residues apart in the unaligned
    sequence). Use ``count_contacts`` for the experimental structure instead.

    Args:
        contacts_dir: Per-sequence predicted contact maps (``<seqid>.txt``).
        msa_dir: Directory holding the family's aligned MSA.
        filtered_sites_path: Directory holding AF2Rank's ``result.txt`` list of kept columns,
            or ``None`` to use every column. AF2 ignores insertions relative to the reference,
            so its predictions only cover a subset of sites; OmegaFold needs no filtering.
        family: Protein family name.
        tree_split: Split A / split B leaf names.
        split: Which side of the split to aggregate over (split A except for real split B).
        include_internals: Whether to keep internal-node sequences.

    Returns:
        ``(contacts, nontrivial_contacts, count_nongap)``, each ``(L, L)``. ``count_nongap``
        is the number of sequences with residues at both columns, the denominator for the
        downstream contact-preservation threshold.
    """
    msa = read_msa(f"{msa_dir}/{family}.txt")
    msa = {k: v for k, v in msa.items() if k in tree_split[split] or k == "seq1"}
    if not include_internals:
        msa = {k: v for k, v in msa.items() if not k.startswith("internal")}

    L = len(list(msa.values())[0])
    if filtered_sites_path is not None:
        with open(os.path.join(filtered_sites_path, "result.txt"), "r") as f:
            filtered_sites = json.load(f)
    else:
        filtered_sites = list(range(L))

    msa = {seq_id: "".join(seq[i] for i in filtered_sites) for seq_id, seq in msa.items()}
    unaligned_map = msa_to_unaligned_map(msa)

    predicted_contacts = np.zeros(shape=(L, L), dtype=np.int8)
    predicted_nontrivial_contacts = np.zeros(shape=(L, L), dtype=np.int8)
    count_nongap = np.zeros(shape=(L, L), dtype=np.int8)

    for seqid in msa:
        # Structure prediction excludes ambiguous characters, so those sequences have no map.
        if seqid not in tree_split[split] or "X" in msa[seqid]:
            continue
        seq = msa[seqid]
        idx_mapping = unaligned_map[seqid]
        contact_map = np.loadtxt(f"{contacts_dir}/{seqid}.txt")

        for i in range(len(seq) - 1):
            if seq[i] == "-":
                continue
            for j in range(i + 1, len(seq)):
                if seq[j] == "-":
                    continue

                # i, j index the filtered sequence; filtered_sites maps back to MSA columns.
                filtered_i = filtered_sites[i]
                filtered_j = filtered_sites[j]

                count_nongap[filtered_i][filtered_j] += 1
                count_nongap[filtered_j][filtered_i] += 1

                unaligned_i = idx_mapping[i]
                unaligned_j = idx_mapping[j]

                nontrivial_contact = (
                    abs(unaligned_i - unaligned_j) >= DISTANCE_FOR_NONTRIVIAL_CONTACT
                )

                if contact_map[unaligned_i, unaligned_j] == 1:
                    predicted_contacts[filtered_i, filtered_j] += 1
                    predicted_contacts[filtered_j, filtered_i] += 1

                    if nontrivial_contact:
                        predicted_nontrivial_contacts[filtered_i, filtered_j] += 1
                        predicted_nontrivial_contacts[filtered_j, filtered_i] += 1

    return predicted_contacts, predicted_nontrivial_contacts, count_nongap


def count_contacts(pdb_path: str, msa_dir: str, family: str) -> Tuple[np.ndarray, np.ndarray]:
    """Experimental contact map for ``seq1``, projected into the alignment's reference frame.

    The counterpart of ``generate_predicted_contacts`` for the experimental structure.

    Returns:
        ``(aligned_contacts, nontrivial_aligned_contacts)``, each ``(L, L)``.
    """
    msa = read_msa(f"{msa_dir}/{family}.txt")
    unaligned_map = msa_to_unaligned_map(msa)
    L = len(list(msa.values())[0])

    exp_contact_map, _ = generate_contact_map(pdb_path)
    exp_contact_map_aligned = np.zeros(shape=(L, L))
    exp_contact_map_nontrivial_aligned = np.zeros(shape=(L, L))

    seq, idx_mapping = msa["seq1"], unaligned_map["seq1"]
    for i in range(len(seq) - 1):
        if seq[i] == "-":
            continue
        for j in range(i + 1, len(seq)):
            if seq[j] == "-":
                continue

            unaligned_i = idx_mapping[i]
            unaligned_j = idx_mapping[j]

            nontrivial_contact = abs(unaligned_i - unaligned_j) >= DISTANCE_FOR_NONTRIVIAL_CONTACT

            if exp_contact_map[unaligned_i, unaligned_j] == 1:
                exp_contact_map_aligned[i, j] += 1
                exp_contact_map_aligned[j, i] += 1

                if nontrivial_contact:
                    exp_contact_map_nontrivial_aligned[i, j] += 1
                    exp_contact_map_nontrivial_aligned[j, i] += 1

    return exp_contact_map_aligned, exp_contact_map_nontrivial_aligned


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


def plot_contact_map(
    contacts: Dict[str, np.ndarray],
    experimental_contacts: np.ndarray,
    models: List[str],
    positions: List[Tuple[int, int]],
    threshold: float,
    output_path: str,
    title_suffix: str,
) -> None:
    """One panel per model: reference contacts below the diagonal, predictions above.

    Predictions are colored by whether they match the reference. Saved to
    ``<output_path>/contact_maps/<title_suffix>_<threshold>.png``.

    Args:
        contacts: model name -> ``(N, 2)`` array of predicted contact site pairs.
        experimental_contacts: ``(N, 2)`` array of reference contact site pairs.
        models: Model names to plot.
        positions: ``(row, col)`` subplot slot per model.
        threshold: Threshold used to binarize, for the title and filename.
        output_path: Directory to save under.
        title_suffix: ``all`` or ``nontrivial``.
    """
    n_rows = (len(models) + 1) // 2
    fig, axs = plt.subplots(n_rows, 2, figsize=(10, 5 * n_rows), squeeze=False)
    alpha = 0.5

    for model, (subplot_x, subplot_y) in zip(models, positions):
        model_contacts = contacts[model]

        model_contact_mask = model_contacts[:, 0] >= model_contacts[:, 1]
        exp_contact_mask = experimental_contacts[:, 0] <= experimental_contacts[:, 1]

        model_contacts_masked = model_contacts[model_contact_mask]
        exp_contacts_masked = experimental_contacts[exp_contact_mask]

        # `in` against an ndarray is numpy's any-elementwise membership, not a row lookup.
        # The published panels were generated with these semantics; do not "fix" it.
        correct_contacts = np.array(
            [tuple(pt) for pt in model_contacts_masked if tuple(pt) in experimental_contacts]
        )
        incorrect_contacts = np.array(
            [tuple(pt) for pt in model_contacts_masked if tuple(pt) not in experimental_contacts]
        )

        axs[subplot_x, subplot_y].scatter(
            exp_contacts_masked[:, 0], exp_contacts_masked[:, 1],
            s=10, alpha=alpha, color="gray", label="Reference",
        )

        if correct_contacts.size > 0:
            axs[subplot_x, subplot_y].scatter(
                correct_contacts[:, 0], correct_contacts[:, 1],
                s=5, alpha=alpha, color="blue", label="Correct",
            )

        if incorrect_contacts.size > 0:
            axs[subplot_x, subplot_y].scatter(
                incorrect_contacts[:, 0], incorrect_contacts[:, 1],
                s=5, alpha=alpha, color="red", label="Incorrect",
            )

        axs[subplot_x, subplot_y].set_title(model)
        axs[subplot_x, subplot_y].grid(True)
        axs[subplot_x, subplot_y].invert_yaxis()
        axs[subplot_x, subplot_y].legend(loc="upper right")
        axs[subplot_x, subplot_y].set_aspect("equal", adjustable="box")

    fig.supxlabel("Site")
    fig.supylabel("Site")
    fig.suptitle(f"Predicted Contacts ({title_suffix}), threshold {threshold:.2f}")
    plt.tight_layout()

    if not os.path.exists(f"{output_path}/contact_maps"):
        os.makedirs(f"{output_path}/contact_maps", exist_ok=True)

    plt.savefig(f"{output_path}/contact_maps/{title_suffix}_{threshold:.2f}.png", dpi=300)
    plt.close()


def plot_contact_maps(
    predicted_contacts: Dict[str, np.ndarray],
    predicted_nontrivial_contacts: Dict[str, np.ndarray],
    experimental_contacts: np.ndarray,
    experimental_nontrivial_contacts: np.ndarray,
    count_nongap: Dict[str, np.ndarray],
    threshold: float,
    output_path: str,
    recall_only: bool = False,
) -> None:
    """Binarize the aligned contact counts at ``threshold`` and plot them against the reference.

    A site pair counts as predicted when more than ``threshold * count_nongap[i][j]`` of the
    sequences that have residues there place them in contact. Takes the outputs of
    ``generate_predicted_contacts`` and ``count_contacts``, and emits both the ``all`` and
    ``nontrivial`` panels via ``plot_contact_map``.

    Args:
        predicted_contacts: model name -> ``(L, L)`` contact counts.
        predicted_nontrivial_contacts: model name -> ``(L, L)`` nontrivial contact counts.
        experimental_contacts: ``(L, L)`` binary reference contacts.
        experimental_nontrivial_contacts: ``(L, L)`` binary reference nontrivial contacts.
        count_nongap: model name -> ``(L, L)`` non-gap sequence pair counts.
        threshold: Binarization threshold, as a fraction of ``count_nongap``.
        output_path: Directory to save under.
        recall_only: Restrict predictions to those recovering experimental contacts.
    """
    os.makedirs(output_path, exist_ok=True)
    experimental_contacts = np.argwhere(experimental_contacts == 1)
    experimental_nontrivial_contacts = np.argwhere(experimental_nontrivial_contacts == 1)

    binarized_contacts = {}
    binarized_nontrivial_contacts = {}
    for model in predicted_contacts:
        binarized_contacts[model] = (
            predicted_contacts[model] > threshold * count_nongap[model]
        ).astype(int)
        binarized_contacts[model] = np.argwhere(binarized_contacts[model] == 1)

        binarized_nontrivial_contacts[model] = (
            predicted_nontrivial_contacts[model] > threshold * count_nongap[model]
        ).astype(int)
        binarized_nontrivial_contacts[model] = np.argwhere(
            binarized_nontrivial_contacts[model] == 1
        )

        if recall_only:
            recall_nontrivial_contacts = np.isin(
                binarized_nontrivial_contacts[model], experimental_nontrivial_contacts
            )
            recall_contacts = np.isin(binarized_contacts[model], experimental_contacts)

            binarized_nontrivial_contacts[model] = binarized_nontrivial_contacts[model][
                recall_nontrivial_contacts
            ]
            binarized_contacts[model] = binarized_contacts[model][recall_contacts]

    models = list(predicted_contacts.keys())
    n_rows = (len(models) + 1) // 2
    positions = [(r, c) for r in range(n_rows) for c in range(2)]

    plot_contact_map(
        binarized_contacts, experimental_contacts, models, positions, threshold, output_path, "all"
    )
    plot_contact_map(
        binarized_nontrivial_contacts, experimental_nontrivial_contacts,
        models, positions, threshold, output_path, "nontrivial",
    )


def plot_contacts_precision_recall(
    predicted_contacts: Dict[str, np.ndarray],
    predicted_nontrivial_contacts: Dict[str, np.ndarray],
    experimental_contacts: np.ndarray,
    experimental_nontrivial_contacts: np.ndarray,
    count_nongap: Dict[str, np.ndarray],
    thresholds: Sequence[float],
    family: str,
    output_path: str,
    recall_only: bool = False,
) -> pd.DataFrame:
    """Precision / recall / F1 vs binarization threshold for one family's contact predictions.

    Args: as ``plot_contact_maps``, but sweeping ``thresholds``.

    Returns:
        Per-threshold metrics for each model and contact type; also written to
        ``<output_path>/contact_maps/contact_metrics.csv``.
    """
    contact_types = {
        "all": (predicted_contacts, experimental_contacts),
        "nontrivial": (predicted_nontrivial_contacts, experimental_nontrivial_contacts),
    }

    results = {
        contact_type: {
            metric: {model: [] for model in predicted_contacts}
            for metric in ["Precision", "Recall", "F1-Score"]
        }
        for contact_type in contact_types
    }

    for contact_type, (predicted, experimental) in contact_types.items():
        real_flat = experimental.ravel()

        for threshold in thresholds:
            binarized_contacts = {}

            for model in predicted:
                binarized_contacts[model] = (
                    predicted[model] > threshold * count_nongap[model]
                ).astype(int)
                binarized_contacts[model] = np.argwhere(binarized_contacts[model] == 1)

            for model in predicted:
                pred_flat = np.zeros_like(real_flat)
                pred_flat[
                    binarized_contacts[model][:, 0] * experimental.shape[1]
                    + binarized_contacts[model][:, 1]
                ] = 1

                results[contact_type]["Recall"][model].append(recall_score(real_flat, pred_flat))

                if recall_only:
                    continue

                results[contact_type]["Precision"][model].append(
                    precision_score(real_flat, pred_flat)
                )
                results[contact_type]["F1-Score"][model].append(f1_score(real_flat, pred_flat))

    os.makedirs(f"{output_path}/contact_maps/metrics", exist_ok=True)
    for contact_type in contact_types:
        for metric in ["Precision", "Recall", "F1-Score"]:
            if recall_only and metric != "Recall":
                continue
            plt.figure()
            for model in predicted_contacts:
                plt.plot(thresholds, results[contact_type][metric][model], label=model)
            plt.xlabel("Threshold")
            plt.ylabel(metric)
            plt.title(f"{metric} vs Threshold ({family}) - {contact_type}")
            plt.legend()
            plt.grid(True)
            plt.savefig(
                f"{output_path}/contact_maps/metrics/{metric.lower()}_{contact_type}.png", dpi=300
            )
            plt.close()

    df_data = {"Threshold": thresholds}
    for contact_type in contact_types:
        for metric in ["Precision", "Recall", "F1-Score"]:
            if recall_only and metric != "Recall":
                continue
            for model in predicted_contacts:
                df_data[f"{model}_{metric}_{contact_type}"] = results[contact_type][metric][model]

    df = pd.DataFrame(df_data)
    df.to_csv(f"{output_path}/contact_maps/contact_metrics.csv", index=False)
    return df


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


def plot_contact_precision_recall_by_model(
    pr_scores: Dict[str, pd.DataFrame],
    training_fams_map: Dict[str, bool],
    output_path: str,
) -> None:
    """Per-model precision / recall / F1 curves, averaged over families.

    Args:
        pr_scores: family -> the DataFrame from ``plot_contacts_precision_recall``.
        training_fams_map: family -> whether it was in the training set.
        output_path: Directory to save under.
    """
    os.makedirs(output_path, exist_ok=True)

    def collect_scores(filter_fn):
        collected = defaultdict(lambda: defaultdict(list))
        thresholds = None

        for family, df in pr_scores.items():
            if not filter_fn(training_fams_map[family]):
                continue
            if thresholds is None:
                thresholds = df["Threshold"].values

            for col in df.columns:
                if col == "Threshold":
                    continue
                model, metric, contact_type = col.rsplit("_", 2)
                collected[(metric, contact_type)][model].append(df[col].values)

        return collected, thresholds

    for label, fam_filter in [
        ("all", lambda _: True),
        ("in_family", lambda x: x),
        ("held_out", lambda x: not x),
    ]:
        collected_scores, thresholds = collect_scores(fam_filter)

        for (metric, contact_type), model_scores in collected_scores.items():
            rows = []
            for model, scores_list in model_scores.items():
                mean_scores = np.mean(np.array(scores_list), axis=0)
                for t, score in zip(thresholds, mean_scores):
                    rows.append({"Model": model, "Threshold": t, metric: score})

            df = pd.DataFrame(rows)

            plt.figure(figsize=(8, 6))
            sns.lineplot(data=df, x="Threshold", y=metric, hue="Model")
            plt.title(f"{metric} vs Threshold ({contact_type}) - {label}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{output_path}/{metric.lower()}_{contact_type}_{label}.png", dpi=300)
            plt.close()


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
