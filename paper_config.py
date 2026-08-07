"""Central data locations for the PEINT paper figures/benchmarks.

Large inputs and precomputed result dirs live OUTSIDE this repo (they are not
copied in). Point the two roots below at wherever those live, via environment
variables, or accept the defaults (the current cluster layout). Small rate
matrices are kept in-repo under ``data/rate_matrices``.

Import as ``import paper_config as cfg`` from the repo root (scripts are run with
``python -m benchmarks.<name>`` from the repo root, so the root is on sys.path).
"""

import importlib.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent


def _peint_repo_root() -> Path:
    """Root of the installed ``peint`` checkout, which ships the model checkpoints.

    Resolved from the installed ``protevo`` package rather than assuming a sibling
    directory, so it works wherever ``pip install -e`` pointed. Uses ``find_spec`` to
    avoid importing (and thus loading torch) at config-import time.
    """
    spec = importlib.util.find_spec("protevo")
    if spec is None or not spec.origin:
        raise ImportError(
            "Cannot locate the installed 'protevo' package. Install peint first: "
            "pip install -e /path/to/peint"
        )
    return Path(spec.origin).resolve().parent.parent

# Root holding the current results + a3m inputs (e.g. results_revision1, input_data/a3m).
DATA_ROOT = Path(os.environ.get(
    "PEINT_PAPER_DATA_ROOT",
    "/scratch/users/akoehl/protein-evolution",
))

# Root holding the 512-leaf simulation inputs (empirical_msas / trees / root_sequences).
# These live under a separate legacy tree in the current layout.
SIM_ROOT = Path(os.environ.get(
    "PEINT_PAPER_SIM_ROOT",
    "/scratch/users/akoehl/old/protein-evolution/local_data/simulation"
    "/final_simulation_512_leaves/ratio_0-1_nucleus_1-0",
))

# --- Derived locations (extend as figures are migrated) ---
# The a3m alignments (and the ground-truth PDBs below) come from the trRosetta training
# set: https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz (~30 GB, fetch manually).
INPUT_A3M_DIR = Path(os.environ.get(
    "PEINT_PAPER_INPUT_A3M_DIR", str(DATA_ROOT / "input_data" / "a3m")
))
RESULTS_DIR = Path(os.environ.get(
    "PEINT_PAPER_RESULTS_DIR",
    str(DATA_ROOT / "local_data" / "results_revision1"),
))

EMPIRICAL_MSA_DIR = SIM_ROOT / "empirical_msas"
TREE_DIR = SIM_ROOT / "trees"
ROOT_SEQ_DIR = SIM_ROOT / "root_sequences"

# Small, in-repo assets.
RATE_MATRIX_DIR = REPO_ROOT / "data" / "rate_matrices"

# --- Figure 3 (PCP mutation counts / Historian indels) specifics ---
SIM_FAMILIES_FILE = Path(os.environ.get(
    "PEINT_PAPER_SIM_FAMILIES_FILE",
    str(DATA_ROOT / "local_data" / "sim_families_out_lg_s256_ok.txt"),
))
SIMULATIONS_DIR = RESULTS_DIR / "simulations"
LEAF_DISTANCES_DIR = RESULTS_DIR / "leaf_distances"

# Where generated figures are written.
FIGURES_DIR = Path(os.environ.get("PEINT_PAPER_FIGURES_DIR", str(REPO_ROOT / "figures" / "output")))

# --- Figure 2 (simulation / time estimation) specifics ---
# Both Figure 2 panels use the checkpoint shipped with the peint repo
# (model_checkpoints/peint.ckpt, documented there for generation and time estimation).
# The paper's time-estimation panel originally used a separate time-embedding checkpoint
# from later experiments; the released model supersedes it.
PEINT_CHECKPOINT = Path(os.environ.get(
    "PEINT_PAPER_CHECKPOINT",
    str(_peint_repo_root() / "model_checkpoints" / "peint.ckpt"),
))

# Transitions (x, y, t triples) behind the time-estimation panel.
# PROVISIONAL: this default is the 192-leaf subset that was used to keep runtimes down,
# not the full 1024-leaf set. Point PEINT_PAPER_TRANSITIONS_DIR at the full set before
# generating the final figure.
TRANSITIONS_DIR = Path(os.environ.get(
    "PEINT_PAPER_TRANSITIONS_DIR",
    "/scratch/users/akoehl/old/protein-evolution/local_data/15k_gapless_scale_test_192l",
))
NONTRAIN_FAMILIES_FILE = Path(os.environ.get(
    "PEINT_PAPER_NONTRAIN_FAMILIES_FILE",
    str(DATA_ROOT / "local_data" / "14k5_nontrain_fam.json"),
))

# Per-site rates (4-category) for the LG arm of the simulation panel, one <family>.txt per
# family. Originally produced by cherryml's treewise train/test split
# (``train_site_rates_4cat_dir``); shipped as data rather than recomputed here.
SITE_RATES_DIR = Path(os.environ.get(
    "PEINT_PAPER_SITE_RATES_DIR",
    str(REPO_ROOT / "local_data" / "output_site_rates_dir"),
))

# --- Structure prediction (Figure 2 pLDDT + the TM-score benchmarks) ---
# AlphaFold weights for AF2Rank and the TM-align binary for its TM-score terms are both
# fetched by scripts/fetch_af2rank_assets.sh into the repo. Override the env vars to point
# at copies you already have.
AF2_WEIGHTS_DIR = Path(os.environ.get(
    "PEINT_PAPER_AF2_WEIGHTS_DIR", str(REPO_ROOT / "data" / "af2_params")
))
TMALIGN_PATH = Path(os.environ.get(
    "PEINT_PAPER_TMALIGN_PATH", str(REPO_ROOT / "bin" / "TMalign")
))

# Experimental structures the TM-score / contact benchmarks compare against
# (one <family>.pdb per family). These come from the trRosetta training set, which also
# supplies INPUT_A3M_DIR above:
#     https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz   (~30 GB)
# Download and extract it yourself, then point the two env vars at its pdb/ and a3m/
# subdirectories. Deliberately not fetched by any script here given the size.
GROUND_TRUTH_STRUCTURE_DIR = Path(os.environ.get(
    "PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR", "/scratch/users/matthew_liu/input_data/pdb"
))

# --- 3Di annotation (ProstT5) ---
# Both fetched by scripts/fetch_3di_weights.sh. PROSTT5_CACHE_DIR is a HuggingFace cache
# directory for the encoder; PROSTT5_CNN_CHECKPOINT is the CNN head that maps embeddings
# to 3Di states.
PROSTT5_CACHE_DIR = Path(os.environ.get(
    "PEINT_PAPER_PROSTT5_CACHE_DIR", str(REPO_ROOT / "data" / "prostt5")
))
PROSTT5_CNN_CHECKPOINT = Path(os.environ.get(
    "PEINT_PAPER_PROSTT5_CNN_CHECKPOINT",
    str(REPO_ROOT / "data" / "prostt5" / "cnn_chkpnt" / "model.pt"),
))

# --- Figure 3 (conservation) specifics ---
# Real + PEINT sequences realigned into the reference frame via `mafft --add`.
MAFFT_ADD_DIR = RESULTS_DIR / "mafft_add"
CONSERVATION_DIR = RESULTS_DIR / "conservation"

# indels-only inputs. The "real" FastTree trees + subsampled pfam MSAs were cherryml
# cache artifacts on the original machine and are NOT materialized here; the heldout
# family list is likewise absent. These default to the closest present dirs but should
# be confirmed / pointed at the real data (override via the env vars below).
SIMULATED_MSA_DIR = SIM_ROOT / "simulated_msas"
REAL_TREE_DIR = Path(os.environ.get("PEINT_PAPER_REAL_TREE_DIR", str(TREE_DIR)))
REAL_MSA_DIR = Path(os.environ.get("PEINT_PAPER_REAL_MSA_DIR", str(EMPIRICAL_MSA_DIR)))
HELDOUT_FAMILIES_FILE = Path(os.environ.get(
    "PEINT_PAPER_HELDOUT_FAMILIES_FILE",
    str(DATA_ROOT / "local_data" / "final_sim_held_out_family.txt"),
))

# --- Generalization analysis (reviewer response) ---
# Tests whether simulation quality holds on protein classes ABSENT from training. The
# eval families are already held out at the PDB level; here we further split them by
# whether their Pfam family / CATH superfamily was ever seen in the 14,498 training
# families. Fetched annotation sources (SIFTS Pfam + CATH, Pfam clans, Pfam-A HMMs) and
# the built family->labels cache are populated idempotently by ``paper.generalization``.
#
# The train / held-out family lists. The held-out list is the simulation eval set; it is
# the JSON form of HELDOUT_FAMILIES_FILE (same 553 families), kept separate because the
# JSON also carries the canonical ordering used elsewhere.
TRAIN_FAMILIES_FILE = Path(os.environ.get(
    "PEINT_PAPER_TRAIN_FAMILIES_FILE", str(DATA_ROOT / "local_data" / "14k5_fam.json")
))
EVAL_FAMILIES_FILE = Path(os.environ.get(
    "PEINT_PAPER_EVAL_FAMILIES_FILE", str(DATA_ROOT / "local_data" / "14k5_nontrain_fam.json")
))
ANNOTATION_DIR = Path(os.environ.get(
    "PEINT_PAPER_ANNOTATION_DIR", str(DATA_ROOT / "local_data" / "generalization" / "annotations")
))
GENERALIZATION_DIR = Path(os.environ.get(
    "PEINT_PAPER_GENERALIZATION_DIR", str(FIGURES_DIR / "generalization")
))


# --- lm-design energy (ESM-MCMC proposer experiment) ---
# The literal lm-design accept/reject energy needs the linear distogram-projection weights and
# background n-gram stats, both fetched by scripts/fetch_lmdesign_assets.sh into data/lm_design/.
# The vendored model code lives in-repo under paper/vendor/lm_design/ (MIT, see its NOTICE.md).
LMDESIGN_DIR = Path(os.environ.get("PEINT_PAPER_LMDESIGN_DIR", str(REPO_ROOT / "data" / "lm_design")))
LMDESIGN_WEIGHTS = LMDESIGN_DIR / "linear_projection_model.pt"
LMDESIGN_NGRAM_DIR = LMDESIGN_DIR / "ngram_stats"


def require(path) -> Path:
    """Return ``path`` if it exists, else raise a clear error (fail-fast, no fallback)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected data at {p}. Set PEINT_PAPER_DATA_ROOT / PEINT_PAPER_SIM_ROOT "
            f"to point at your data, or check the path."
        )
    return p
