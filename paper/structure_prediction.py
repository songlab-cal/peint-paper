"""Structure prediction for the paper benchmarks: OmegaFold and AF2Rank.

Ported from the model repo's ``peint/datasets/_structure_prediction.py``. Structure
prediction is a *benchmarking* dependency, not part of the ``peint`` model library.

The paper uses **both** predictors, and they have very different requirements:

``generate_omegafold_predictions``
    Folds sequences with the ``omegafold`` CLI. Needs CUDA and ``omegafold`` on PATH.
``generate_af2_predictions``
    Scores sequences against a reference structure with AF2Rank. Needs CUDA, JAX +
    ColabDesign, AlphaFold weights (``cfg.AF2_WEIGHTS_DIR``), and TM-align.

The original imported ColabDesign at module scope and raised at import time if the
TM-align binary was missing, so the OmegaFold-only path could not be used without the
whole JAX stack installed. Both are deferred here: ColabDesign is imported inside the
AF2Rank code, and TM-align is checked when it is first invoked.
"""

import json
import os
import re
import subprocess
import tempfile
from typing import Optional

import numpy as np
import pandas as pd
import torch

from peint import caching as peint_caching
from peint.utils import read_msa, write_msa

import paper_config as cfg

OMEGAFOLD_PATH = "omegafold"  # installed into the environment, resolved on PATH

def parse_tm_align(output: bytes) -> dict:
    """Pull TM-score and RMSD out of TM-align stdout (chain 1 is the reference)."""
    output = output.decode("utf-8")
    rmsd_match = re.search(r"RMSD=\s+([\d.]+)", output)
    tm_score_match = re.search(
        r"TM-score=\s+([\d.]+)\s+\(if normalized by length of Chain_1", output
    )
    if not (rmsd_match and tm_score_match):
        raise ValueError(f"Could not parse TM-align output: {output[:500]}")

    return {"TM-score": float(tm_score_match.group(1)), "RMSD": float(rmsd_match.group(1))}


def tmscore(x, y) -> dict:
    """TM-align two CA coordinate arrays by writing them out as minimal PDBs."""
    tmalign = cfg.require(cfg.TMALIGN_PATH)

    with tempfile.TemporaryDirectory() as tmpdir:
        pdb1 = os.path.join(tmpdir, "struct1.pdb")
        pdb2 = os.path.join(tmpdir, "struct2.pdb")
        for filename, coords in [(pdb1, x), (pdb2, y)]:
            with open(filename, "w") as out:
                for k, c in enumerate(coords):
                    out.write(
                        "ATOM  %5d  %-2s  %3s %s%4d    %8.3f%8.3f%8.3f  %4.2f  %4.2f\n"
                        % (k + 1, "CA", "ALA", "A", k + 1, c[0], c[1], c[2], 1, 0)
                    )

        result = subprocess.run([str(tmalign), pdb1, pdb2], capture_output=True)
        return parse_tm_align(result.stdout)


class af2rank:
    """AF2Rank: score a sequence by threading it onto a template structure with AlphaFold."""

    def __init__(self, pdb, chain=None, model_name="model_1_ptm", model_names=None, data_dir="."):
        self.args = {
            "pdb": pdb,
            "chain": chain,
            "use_multimer": ("multimer" in model_name),
            "model_name": model_name,
            "model_names": model_names,
            "data_dir": data_dir,
        }
        self.reset()

    def reset(self):
        from colabdesign import mk_af_model

        self.model = mk_af_model(
            protocol="fixbb",
            use_templates=True,
            use_multimer=self.args["use_multimer"],
            debug=False,
            model_names=self.args["model_names"],
            data_dir=self.args["data_dir"],
        )

        self.model.prep_inputs(self.args["pdb"], chain=self.args["chain"])
        self.model.set_seq(mode="wildtype")
        self.wt_batch = _copy_dict(self.model._inputs["batch"])
        self.wt = self.model._wt_aatype

    def set_pdb(self, pdb, chain=None):
        if chain is None:
            chain = self.args["chain"]
        self.model.prep_inputs(pdb, chain=chain)
        self.model.set_seq(mode="wildtype")
        self.wt = self.model._wt_aatype

    def set_seq(self, seq):
        self.model.set_seq(seq=seq)
        self.wt = self.model._params["seq"][0].argmax(-1)

    def _get_score(self):
        score = _copy_dict(self.model.aux["log"])

        score["pae"] = 31.0 * score["pae"]
        score["rmsd_io"] = score.pop("rmsd", None)

        i_xyz = self.model._inputs["batch"]["all_atom_positions"][:, 1]
        o_xyz = np.array(self.model.aux["atom_positions"][:, 1])

        if hasattr(self, "wt_batch"):
            n_xyz = self.wt_batch["all_atom_positions"][:, 1]
            score["tm_i"] = tmscore(n_xyz, i_xyz)["TM-score"]
            score["tm_o"] = tmscore(n_xyz, o_xyz)["TM-score"]

        score["tm_io"] = tmscore(i_xyz, o_xyz)["TM-score"]
        score["composite"] = score["ptm"] * score["plddt"] * score["tm_io"]
        return score

    def predict(
        self,
        pdb=None,
        seq=None,
        chain=None,
        input_template=True,
        model_name=None,
        rm_seq=True,
        rm_sc=True,
        rm_ic=False,
        recycles=1,
        iterations=1,
        output_pdb=None,
        extras=None,
        verbose=True,
    ):
        if model_name is not None:
            self.args["model_name"] = model_name
            wants_multimer = "multimer" in model_name
            if wants_multimer != self.args["use_multimer"]:
                self.args["use_multimer"] = wants_multimer
                self.reset()

        if pdb is not None:
            self.set_pdb(pdb, chain)
        if seq is not None:
            self.set_seq(seq)

        self.model._inputs["batch"]["aatype"] = self.wt

        self.model.set_opt(template=dict(rm_ic=rm_ic), num_recycles=recycles)
        self.model._inputs["rm_template"][:] = not input_template
        self.model._inputs["rm_template_sc"][:] = rm_sc
        self.model._inputs["rm_template_seq"][:] = rm_seq

        # "Manual" recycles: feed the previous prediction back in as the template.
        ini_atoms = self.model._inputs["batch"]["all_atom_positions"].copy()
        for i in range(iterations):
            self.model.predict(models=self.args["model_name"], verbose=False)
            if i < iterations - 1:
                self.model._inputs["batch"]["all_atom_positions"] = self.model.aux["atom_positions"]
            else:
                self.model._inputs["batch"]["all_atom_positions"] = ini_atoms

        score = self._get_score()
        if extras is not None:
            score.update(extras)

        if output_pdb is not None:
            self.model.save_pdb(output_pdb)

        if verbose:
            keys = ["tm_i", "tm_o", "tm_io", "composite", "ptm", "i_ptm", "plddt", "fitness", "id"]
            print(*[
                f"{k} {score[k]:.4f}" if isinstance(score[k], float) else f"{k} {score[k]}"
                for k in keys if k in score
            ])

        return score


def _copy_dict(d):
    from colabdesign.shared.utils import copy_dict

    return copy_dict(d)


def create_valid_output_for_caching(result_dir):
    """Write the SUCCESS marker the caching layer looks for."""
    out_path = os.path.join(result_dir, "result.txt")
    with open(out_path, "w") as f:
        f.write("SUCCESS")
        os.chmod(out_path, mode=444)


@peint_caching.cached_computation(
    output_dirs=["output_structures_dir"],
    exclude_args_if_default=["input_filename", "keep_prefix"],
    write_extra_log_files=True,
)
def generate_omegafold_predictions(
    sequences_dir,
    family,
    input_filename: Optional[str] = None,
    keep_prefix: Optional[str] = "seq",
    output_structures_dir=None,
):
    """Fold a family's sequences with OmegaFold, writing one PDB per sequence.

    ``keep_prefix`` restricts folding to records whose id starts with it — ``"seq"`` keeps the
    empirical sequences and drops internal nodes. Pass ``None`` to fold every record, which the
    Figure 2 star-topology simulation needs (its ids are ``<family>-sample<i>-<t>``).
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "OmegaFold structure prediction requires CUDA; no GPU is visible to this process."
        )

    input_path = os.path.join(sequences_dir, input_filename or f"{family}.txt")
    msa = read_msa(input_path)
    if keep_prefix is not None:
        msa = {k: v for k, v in msa.items() if k.startswith(keep_prefix)}
    msa = {k: v.replace("-", "") for k, v in msa.items()}
    if not msa:
        raise ValueError(
            f"No sequences left to fold for {family} from {input_path} "
            f"(keep_prefix={keep_prefix!r})."
        )

    # OmegaFold reads a plain FASTA; write the gap-stripped, filtered set to a temp file.
    with tempfile.NamedTemporaryFile(delete=False, mode="w", suffix=".fasta") as temp_file:
        temp_input_path = temp_file.name
    try:
        write_msa(msa, temp_input_path)
        subprocess.run([OMEGAFOLD_PATH, temp_input_path, output_structures_dir], check=True)
    finally:
        os.remove(temp_input_path)

    for file in os.listdir(output_structures_dir):
        os.chmod(os.path.join(output_structures_dir, file), mode=444)
    create_valid_output_for_caching(output_structures_dir)


@peint_caching.cached_computation(
    output_dirs=[
        "output_structures_dir",
        "output_scores_dir",
        "output_sites_dir",
    ],
    exclude_args_if_default=[
        "mask_sequence", "mask_sidechains", "mask_interchain",
        "recycles", "iterations", "exclude_internals",
    ],
    write_extra_log_files=True,
)
def generate_af2_predictions(
    sequences_dir,
    ground_truth_structure_dir,
    empirical_seq1,
    family,
    model_mode,
    exclude_internals: bool = True,
    model_num: int = 2,
    mask_sequence: bool = True,
    mask_sidechains: bool = True,
    mask_interchain: bool = False,
    recycles: int = 3,
    iterations: int = 1,
    output_structures_dir: Optional[str] = None,
    output_scores_dir: Optional[str] = None,
    output_sites_dir: Optional[str] = None,
):
    """Score a family's sequences with AF2Rank against its experimental structure."""
    if model_mode == "alphafold":
        model_name = f"model_{model_num}_ptm"
    elif model_mode == "alphafold-multimer":
        model_name = f"model_{model_num}_multimer_v3"
    else:
        raise ValueError(
            f"Unknown model_mode {model_mode!r}; expected 'alphafold' or 'alphafold-multimer'."
        )

    settings = {
        "rm_seq": mask_sequence,
        "rm_sc": mask_sidechains,
        "rm_ic": mask_interchain,
        "recycles": recycles,
        "iterations": iterations,
        "model_name": model_name,
    }

    from colabdesign import clear_mem

    ground_truth_pdb = os.path.join(ground_truth_structure_dir, family + ".pdb")
    chain = family.split("_")[-1]
    clear_mem()
    af = af2rank(
        ground_truth_pdb,
        chain=chain,
        model_name=settings["model_name"],
        data_dir=str(cfg.require(cfg.AF2_WEIGHTS_DIR)),
    )

    seqs = read_msa(os.path.join(sequences_dir, family + ".txt"))

    # AF2 cannot represent insertions relative to the template, so drop every column where the
    # empirical seq1 is a gap. PEINT's own seq1 is not guaranteed to align to the reference,
    # which is why the empirical seq1 is passed in explicitly.
    keep_cols = [i for i, char in enumerate(empirical_seq1) if char != "-"]
    filtered_msa = {
        seq_id: "".join(seq[i] for i in keep_cols) for seq_id, seq in seqs.items()
    }
    if exclude_internals:
        filtered_msa = {k: v for k, v in filtered_msa.items() if "internal" not in k}

    scores, labels = [], []
    for label, x in filtered_msa.items():
        output_pdb = os.path.join(output_structures_dir, f"{label}.txt")
        scores.append(af.predict(seq=x, **settings, output_pdb=output_pdb))
        labels.append(label)
        os.chmod(output_pdb, mode=444)

    create_valid_output_for_caching(output_structures_dir)

    # `seq_id` is the MSA record id; `seqid` (from AF2Rank) is % identity to the template.
    scores_df = pd.DataFrame(
        [{"seq_id": label, **score} for label, score in zip(labels, scores)]
    )
    scores_df.to_csv(os.path.join(output_scores_dir, "result.txt"), index=False)

    # JSON: the downstream TM-score analysis reads this back with json.load.
    output_sites_path = os.path.join(output_sites_dir, "result.txt")
    with open(output_sites_path, "w") as f:
        json.dump(keep_cols, f)
    os.chmod(output_sites_path, mode=444)
