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

    Resolved from the installed ``peint`` package rather than assuming a sibling
    directory, so it works wherever ``pip install -e`` pointed. Uses ``find_spec`` to
    avoid importing (and thus loading torch) at config-import time.
    """
    spec = importlib.util.find_spec("peint")
    if spec is None or not spec.origin:
        raise ImportError(
            "Cannot locate the installed 'peint' package. Install peint first: "
            "pip install -e /path/to/peint"
        )
    return Path(spec.origin).resolve().parent.parent


# The peint checkout: it ships the model checkpoints and, for the Figure 2 likelihood
# panel, the transition data and the _cache_peint evaluation cache.
_peint_repo_env = os.environ.get("PEINT_PAPER_PEINT_REPO")
PEINT_REPO = Path(_peint_repo_env) if _peint_repo_env else _peint_repo_root()

# Root holding the current results + a3m inputs (e.g. results_revision1, input_data/a3m).
DATA_ROOT = Path(os.environ.get(
    "PEINT_PAPER_DATA_ROOT",
    "/scratch/users/akoehl/protein-evolution",
))

# Single root for the collated inputs, matching the layout of the Zenodo tarballs (each
# unpacks relative to this directory). On the development machine the r1/r2/sim entries
# are symlinks out to the authoritative trees; for anyone else they are real directories.
LOCAL_DATA = Path(os.environ.get("PEINT_PAPER_LOCAL_DATA", str(REPO_ROOT / "local_data")))


# Deposit mode. Normally a path resolves to the authoritative tree when that exists and to
# LOCAL_DATA otherwise, which is what lets one config serve both the machine that produced
# the data and someone who just unpacked it. On a machine that has BOTH -- this one, and
# anyone who keeps a copy of the source trees around -- "otherwise" never fires, so there is
# no way to check that the deposit is self-sufficient. Setting PEINT_PAPER_LOCAL_DATA_ONLY=1
# makes LOCAL_DATA win outright: every input then comes from one directory, deterministically.
LOCAL_DATA_ONLY = os.environ.get("PEINT_PAPER_LOCAL_DATA_ONLY", "").lower() not in ("", "0", "false", "no")


def _first_existing(*candidates, env=None):
    """First candidate that exists, else the last one, with an env var overriding all.

    One config has to serve two layouts: this machine, where these inputs sit in scattered
    absolute trees, and an unpacked data deposit, where everything lives under LOCAL_DATA.
    The authoritative tree is listed first so the producing machine keeps resolving to the
    exact same string -- peint cache keys hash absolute argument paths, and a "harmless"
    reordering here would cold-start every cached computation.

    Under LOCAL_DATA_ONLY the first candidate *under LOCAL_DATA* wins whether or not it
    exists, so a missing input fails loudly at the deposit path instead of silently falling
    back to a local tree the deposit does not contain.

    Falling through to the last candidate rather than raising means a missing input is
    reported against the deposit path, which is the layout whoever hits that error is using.
    """
    if env and os.environ.get(env):
        return Path(os.environ[env])
    if LOCAL_DATA_ONLY:
        # Prefer the first deposit-relative candidate that EXISTS, then fall back to the first
        # one regardless. A role can legitimately have more than one deposit-relative location
        # -- e.g. the family split lists ship both in the `aux` archive (LOCAL_DATA/splits/)
        # and inside the figure_data tier (LOCAL_DATA/figure_data/splits/) -- and returning the
        # first unconditionally made the figure_data tier unusable on its own. Falling back to
        # candidates[0] preserves the original intent: a genuinely missing input is reported
        # against the deposit path, not silently resolved to a local tree.
        under = [c for c in candidates if str(c).startswith(str(LOCAL_DATA))]
        for c in under:
            if Path(c).exists():
                return Path(c)
        if under:
            return Path(under[0])
    for c in candidates:
        if Path(c).exists():
            return Path(c)
    return Path(candidates[-1])

# Per-panel tables that fully determine each figure — the lightweight reproduction tier.
FIGURE_DATA_DIR = Path(os.environ.get(
    "PEINT_PAPER_FIGURE_DATA_DIR", str(LOCAL_DATA / "figure_data")
))

# Everything the benchmarks WRITE goes here, so the input trees above stay read-only.
DERIVED_DIR = Path(os.environ.get("PEINT_PAPER_DERIVED_DIR", str(LOCAL_DATA / "derived")))

# Computation caches. These were previously set from RELATIVE paths, which forced the
# PCP benchmark to be run from one specific working directory; routing them through
# config means every script can run from the repo root.
PROTEVO_CACHE_DIR = Path(os.environ.get(
    "PEINT_PAPER_PROTEVO_CACHE_DIR", str(DERIVED_DIR / "_cache_protevo")
))
CHERRYML_CACHE_DIR = Path(os.environ.get(
    "PEINT_PAPER_CHERRYML_CACHE_DIR", str(DERIVED_DIR / "_cache_benchmarking")
))

# Root holding the 512-leaf simulation inputs (empirical_msas / trees / root_sequences).
# These live under a separate legacy tree in the current layout.
SIM_ROOT = _first_existing(
    "/scratch/users/akoehl/old/protein-evolution/local_data/simulation"
    "/final_simulation_512_leaves/ratio_0-1_nucleus_1-0",
    LOCAL_DATA / "sim",
    env="PEINT_PAPER_SIM_ROOT",
)

# --- Derived locations (extend as figures are migrated) ---
# The a3m alignments (and the ground-truth PDBs below) come from the trRosetta training
# set: https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz (~30 GB, fetch manually).
INPUT_A3M_DIR = _first_existing(
    DATA_ROOT / "input_data" / "a3m",
    LOCAL_DATA / "a3m",
    env="PEINT_PAPER_INPUT_A3M_DIR",
)
# The two result generations, kept separate: revision 1 holds the classical baselines,
# PEINT-ESM2 and real; revision 2 holds the ESM-C rerun (its own mafft frame).
RESULTS_R1_DIR = _first_existing(
    DATA_ROOT / "local_data" / "results_revision1",
    LOCAL_DATA / "r1",
    env="PEINT_PAPER_RESULTS_R1_DIR",
) if not os.environ.get("PEINT_PAPER_RESULTS_DIR") else Path(
    os.environ["PEINT_PAPER_RESULTS_DIR"])
RESULTS_R2_DIR = _first_existing(
    DATA_ROOT / "local_data" / "results_revision2_esmc",
    LOCAL_DATA / "r2",
    env="PEINT_PAPER_RESULTS_R2_DIR",
) if not os.environ.get("PEINT_PAPER_ESMC_RESULTS_DIR") else Path(
    os.environ["PEINT_PAPER_ESMC_RESULTS_DIR"])
# Back-compat alias; RESULTS_R1_DIR is the name to use in new code.
RESULTS_DIR = RESULTS_R1_DIR

EMPIRICAL_MSA_DIR = SIM_ROOT / "empirical_msas"
TREE_DIR = SIM_ROOT / "trees"
ROOT_SEQ_DIR = SIM_ROOT / "root_sequences"

# Small, in-repo assets.
RATE_MATRIX_DIR = REPO_ROOT / "data" / "rate_matrices"

# --- Figure 3 (PCP mutation counts / Historian indels) specifics ---
SIM_FAMILIES_FILE = _first_existing(
    DATA_ROOT / "local_data" / "sim_families_out_lg_s256_ok.txt",
    LOCAL_DATA / "splits" / "sim_families_out_lg_s256_ok.txt",
    env="PEINT_PAPER_SIM_FAMILIES_FILE",
)
SIMULATIONS_DIR = RESULTS_DIR / "simulations"
LEAF_DISTANCES_DIR = RESULTS_DIR / "leaf_distances"

# Where generated figures are written.
FIGURES_DIR = Path(os.environ.get("PEINT_PAPER_FIGURES_DIR", str(REPO_ROOT / "figures" / "output")))

# --- Figure 2 (simulation / time estimation) specifics ---
# Both Figure 2 panels use the checkpoint shipped with the peint repo
# (model_checkpoints/peint.ckpt, documented there for generation and time estimation).
# The paper's time-estimation panel originally used a separate time-embedding checkpoint
# from later experiments; the released model supersedes it.
def _checkpoint(name, *extra, env=None):
    """A model checkpoint: the peint repo first, then an unpacked deposit.

    Checkpoints are gitignored in the peint repo and absent from this one, so the only ways
    anyone else gets them are the shared account or the data deposit -- both of which land
    them under LOCAL_DATA/peint/model_checkpoints/, matching the manifest's peint_checkpoints
    role. Repo first so this machine resolves unchanged.
    """
    return _first_existing(
        PEINT_REPO / "model_checkpoints" / name,
        LOCAL_DATA / "peint" / "model_checkpoints" / name,
        *extra, env=env,
    )


PEINT_CHECKPOINT = _checkpoint("peint.ckpt", env="PEINT_PAPER_CHECKPOINT")
VEP_CHECKPOINT = _checkpoint("vep.ckpt", env="PEINT_PAPER_VEP_CHECKPOINT")

# PEINT trained on the Biohub ESM-C backbone, used by the revision-2 simulation runs and the
# Figure 2 likelihood panel. Now kept alongside the other checkpoints in the peint repo as
# peint_esmc.ckpt; the collaborator's scratch path it was trained at is the fallback, so runs
# predating the copy still resolve.
ESMC_SIM_CHECKPOINT = str(_checkpoint(
    "peint_esmc.ckpt",
    "/scratch/users/yufan.cao/protevo_ablations/esmc/"
    "20260729-5e5d20h960d-esmc-14498fams-esmc/epoch=4-step=60000.ckpt",
    env="PEINT_PAPER_ESMC_CHECKPOINT",
))

# The ESM-C VEP head, likewise copied in as vep_esmc.ckpt. This is the checkpoint behind the
# peint_esmc300m run directory under local_data/vep.
ESMC_VEP_CHECKPOINT = str(_checkpoint(
    "vep_esmc.ckpt",
    PEINT_REPO / "model_checkpoints" / "esmc-biohub" / "1e1d-ep_13-step_4130.ckpt",
    env="PEINT_PAPER_ESMC_VEP_CHECKPOINT",
))

# Transitions (x, y, t triples) behind the time-estimation panel.
# PROVISIONAL: this default is the 192-leaf subset that was used to keep runtimes down,
# not the full 1024-leaf set. Point PEINT_PAPER_TRANSITIONS_DIR at the full set before
# generating the final figure.
TRANSITIONS_DIR = _first_existing(
    LOCAL_DATA / "transitions_192l",
    env="PEINT_PAPER_TRANSITIONS_DIR",
)
NONTRAIN_FAMILIES_FILE = _first_existing(
    DATA_ROOT / "local_data" / "14k5_nontrain_fam.json",
    LOCAL_DATA / "splits" / "14k5_nontrain_fam.json",
    FIGURE_DATA_DIR / "splits" / "14k5_nontrain_fam.json",
    env="PEINT_PAPER_NONTRAIN_FAMILIES_FILE",
)

# Per-site rates (4-category) for the LG arm of the simulation panel, one <family>.txt per
# family. Originally produced by cherryml's treewise train/test split
# (``train_site_rates_4cat_dir``); shipped as data rather than recomputed here.
SITE_RATES_DIR = Path(os.environ.get(
    "PEINT_PAPER_SITE_RATES_DIR",
    str(LOCAL_DATA / "peint" / "local_data" / "output_site_rates_dir"),
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

# Historian, built by scripts/build_historian.sh from the submodule. Only the ancestral
# reconstruction *producers* need it; none of the released figure panels invoke it.
HISTORIAN_PATH = Path(os.environ.get(
    "PEINT_PAPER_HISTORIAN_PATH", str(REPO_ROOT / "historian" / "bin" / "historian")
))

# Experimental structures the TM-score / contact benchmarks compare against
# (one <family>.pdb per family). These come from the trRosetta training set, which also
# supplies INPUT_A3M_DIR above:
#     https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz   (~30 GB)
# Download and extract it yourself, then point the two env vars at its pdb/ and a3m/
# subdirectories. Deliberately not fetched by any script here given the size.
GROUND_TRUTH_STRUCTURE_DIR = _first_existing(
    DATA_ROOT / "input_data" / "pdb",
    LOCAL_DATA / "pdb",
    env="PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR",
)

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
HELDOUT_FAMILIES_FILE = _first_existing(
    DATA_ROOT / "local_data" / "final_sim_held_out_family.txt",
    LOCAL_DATA / "splits" / "final_sim_held_out_family.txt",
    FIGURE_DATA_DIR / "splits" / "final_sim_held_out_family.txt",
    env="PEINT_PAPER_HELDOUT_FAMILIES_FILE",
)
# Same 553 families as HELDOUT_FAMILIES_FILE, in the {"families": [...]} JSON form the
# revision-2 benchmarks read.
HELDOUT_FAMILIES_JSON = _first_existing(
    DATA_ROOT / "local_data" / "final_sim_held_out_family.json",
    LOCAL_DATA / "splits" / "final_sim_held_out_family.json",
    FIGURE_DATA_DIR / "splits" / "final_sim_held_out_family.json",
    env="PEINT_PAPER_HELDOUT_FAMILIES_JSON",
)

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
TRAIN_FAMILIES_FILE = _first_existing(
    DATA_ROOT / "local_data" / "14k5_fam.json",
    LOCAL_DATA / "splits" / "14k5_fam.json",
    FIGURE_DATA_DIR / "splits" / "14k5_fam.json",
    env="PEINT_PAPER_TRAIN_FAMILIES_FILE",
)
EVAL_FAMILIES_FILE = _first_existing(
    DATA_ROOT / "local_data" / "14k5_nontrain_fam.json",
    LOCAL_DATA / "splits" / "14k5_nontrain_fam.json",
    FIGURE_DATA_DIR / "splits" / "14k5_nontrain_fam.json",
    env="PEINT_PAPER_EVAL_FAMILIES_FILE",
)
ANNOTATION_DIR = _first_existing(
    DATA_ROOT / "local_data" / "generalization" / "annotations",
    LOCAL_DATA / "annotations",
    env="PEINT_PAPER_ANNOTATION_DIR",
)
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
