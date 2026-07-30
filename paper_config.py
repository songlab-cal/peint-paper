"""Central data locations for the PEINT paper figures/benchmarks.

Large inputs and precomputed result dirs live OUTSIDE this repo (they are not
copied in). Point the two roots below at wherever those live, via environment
variables, or accept the defaults (the current cluster layout). Small rate
matrices are kept in-repo under ``data/rate_matrices``.

Import as ``import paper_config as cfg`` from the repo root (scripts are run with
``python -m benchmarks.<name>`` from the repo root, so the root is on sys.path).
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

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
INPUT_A3M_DIR = DATA_ROOT / "input_data" / "a3m"
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


def require(path) -> Path:
    """Return ``path`` if it exists, else raise a clear error (fail-fast, no fallback)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected data at {p}. Set PEINT_PAPER_DATA_ROOT / PEINT_PAPER_SIM_ROOT "
            f"to point at your data, or check the path."
        )
    return p
