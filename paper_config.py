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
RESULTS_DIR = DATA_ROOT / "local_data" / "results_revision1"

EMPIRICAL_MSA_DIR = SIM_ROOT / "empirical_msas"
TREE_DIR = SIM_ROOT / "trees"
ROOT_SEQ_DIR = SIM_ROOT / "root_sequences"

# Small, in-repo assets.
RATE_MATRIX_DIR = REPO_ROOT / "data" / "rate_matrices"


def require(path) -> Path:
    """Return ``path`` if it exists, else raise a clear error (fail-fast, no fallback)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Expected data at {p}. Set PEINT_PAPER_DATA_ROOT / PEINT_PAPER_SIM_ROOT "
            f"to point at your data, or check the path."
        )
    return p
