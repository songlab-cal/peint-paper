"""Run the ESM2-MCMC evolutionary simulator over a set of families (reviewer response).

Situates PEINT against extant protein language models: evolve the SAME phylogenetic trees,
from the SAME root sequences, as the PEINT simulation, but drive substitutions with ESM2 via
MCMC (see :mod:`paper.esm_mcmc`). The algorithm is adapted from two repositories the reviewer
pointed at:

* Bitbol-Lab **Phylogeny-ESM2** — ``MSAGenerator/MSAGeneratorESM.py``
  (https://github.com/Bitbol-Lab/Phylogeny-ESM2): evolve down a tree, branch length ->
  ``round(branch_length * L * neff)`` MCMC attempts, Metropolis accept via the masked-position
  softmax ratio.
* ESM **lm-design** — ``examples/lm-design/lm_design.py``
  (https://github.com/facebookresearch/esm/tree/main/examples/lm-design): MCMC over sequence
  space with ESM as the accept/reject energy (we sample at T=1 rather than annealing).

Outputs (all under ``figures/output/esm_mcmc/``, kept separate from the main paper figures):

* ``sequences/<family>.txt`` — the simulated MSA (ancestral ``root`` + all tree nodes), same
  format as the PEINT simulator's output, so the same evals can be applied to it later.
* ``acceptance_stats.csv`` — one row per family, appended as each finishes (so partial results
  are on disk mid-run): acceptance rate, realised vs. expected divergence, timing.
* ``run.log`` — live progress (flushed), mirrored to stdout: per-family heartbeats with the
  running acceptance rate and frontier size, so ``tail -f run.log`` gives signal without waiting.

Run from the repo root::

    python -m benchmarks.esm_mcmc_simulation --limit 8            # 8-family smoke test
    python -m benchmarks.esm_mcmc_simulation --limit 8 --neff 6   # rescale divergence up
    python -m benchmarks.esm_mcmc_simulation --families 1a2t_1_A 1acf_1_A
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path

import paper_config as cfg
from protevo.io import read_tree
from protevo.utils import read_msa, write_msa

# Base output dir; the per-run dirs live under a mode subdir (esm_mcmc/<target_mode>/...) so an
# "attempts" baseline and a "hamming" calibration run never clobber each other. Set in main().
OUT_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc"
SEQ_DIR = OUT_DIR / "sequences"
STATS_CSV = OUT_DIR / "acceptance_stats.csv"
LOG_FILE = OUT_DIR / "run.log"

# How many empirical leaves to pairwise-align for the divergence benchmark (evenly spaced; a
# reference statistic, so a subsample is plenty and keeps it fast).
EMP_ID_CAP = 120

STATS_FIELDS = [
    "family", "length", "n_nodes", "n_leaves", "n_branches", "n_cap_hits",
    "total_attempts", "total_accepted", "acceptance_rate",
    "mean_leaf_identity_to_root", "empirical_leaf_identity_to_root",
    "mean_root_to_leaf_distance", "seconds",
]


def _make_logger() -> logging.Logger:
    """Logger that writes to both stdout and run.log, flushed line-by-line."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("esm_mcmc")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    for handler in (logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE)):
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    return logger


def _has_inputs(fam: str) -> bool:
    """A family is runnable if it has both a tree and a root sequence (no structure needed)."""
    return os.path.exists(os.path.join(str(cfg.TREE_DIR), f"{fam}.txt")) and os.path.exists(
        os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")
    )


_STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def empirical_identity_to_root(fam: str, root_label: str, cap: int = EMP_ID_CAP) -> float:
    """Empirical leaf divergence benchmark: mean fraction of ROOT residues preserved in the real
    leaves, measured the same way as the simulator's ``mean_leaf_identity_to_root``.

    The empirical MSAs (``EMPIRICAL_MSA_DIR``) are *unaligned* raw sequences of varying length
    (real evolution has indels), so we pairwise-align each empirical leaf to the root and count
    root positions where the aligned leaf residue matches, divided by the root length. This is
    the right yardstick for the simulator's divergence: raw branch length (subs/site) saturates,
    but empirical identity-to-root and simulated identity-to-root are on the same, comparable
    scale. Returns NaN if the empirical MSA or the root row is unavailable.
    """
    import numpy as np
    import biotite.sequence as bseq
    import biotite.sequence.align as balign

    path = os.path.join(str(cfg.EMPIRICAL_MSA_DIR), f"{fam}.txt")
    if not os.path.exists(path):
        return float("nan")
    msa = read_msa(path)
    if root_label not in msa:
        return float("nan")

    def clean(s: str) -> str:
        return "".join(c for c in s.upper() if c in _STD_AA)

    root_str = clean(msa[root_label])
    if not root_str:
        return float("nan")
    root = bseq.ProteinSequence(root_str)
    matrix = balign.SubstitutionMatrix.std_protein_matrix()
    names = [n for n in msa if n != root_label]
    if len(names) > cap:
        idx = np.unique(np.linspace(0, len(names) - 1, cap).astype(int))
        names = [names[i] for i in idx]
    ids = []
    for n in names:
        leaf = clean(msa[n])
        if not leaf:
            continue
        aln = balign.align_optimal(
            bseq.ProteinSequence(leaf), root, matrix, gap_penalty=(-10, -1)
        )[0]
        matches = sum(1 for a, b in aln.trace if a >= 0 and b >= 0 and leaf[a] == root_str[b])
        ids.append(matches / len(root_str))
    return float(np.mean(ids)) if ids else float("nan")


def _select_families(args) -> list:
    if args.families:
        return list(args.families)
    from paper.generalization import eval_families
    fams = [f for f in eval_families() if _has_inputs(f)]
    return fams[: args.limit] if args.limit else fams


def _append_stats_row(stats, empirical_identity: float) -> None:
    """Append one FamilyStats row (+ empirical benchmark) to the CSV (header on first use)."""
    new_file = not STATS_CSV.exists()
    with open(STATS_CSV, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=STATS_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow({
            "family": stats.family, "length": stats.length, "n_nodes": stats.n_nodes,
            "n_leaves": stats.n_leaves, "n_branches": stats.n_branches,
            "n_cap_hits": stats.n_cap_hits,
            "total_attempts": stats.total_attempts, "total_accepted": stats.total_accepted,
            "acceptance_rate": round(stats.acceptance_rate, 4),
            "mean_leaf_identity_to_root": round(stats.mean_leaf_identity_to_root, 4),
            "empirical_leaf_identity_to_root": round(empirical_identity, 4),
            "mean_root_to_leaf_distance": round(stats.mean_root_to_leaf_distance, 4),
            "seconds": round(stats.seconds, 1),
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=8, help="only the first N runnable eval families")
    ap.add_argument("--families", nargs="*", default=None, help="explicit family list (overrides --limit)")
    ap.add_argument("--model", default="esm2_t33_650M_UR50D", help="fair-esm model name")
    ap.add_argument("--target-mode", choices=["attempts", "hamming"], default="hamming",
                    help="attempts: round(bl*L*neff) proposals/branch (Bitbol-faithful). "
                         "hamming: propose until the branch differs from its parent at "
                         "round(bl*L) sites (realise the branch length as divergence).")
    ap.add_argument("--neff", type=float, default=1.0,
                    help="attempts mode: mutations/site per unit branch length; attempts=round(bl*L*neff)")
    ap.add_argument("--max-hamming-frac", type=float, default=0.9,
                    help="hamming mode: cap each branch's Hamming target at this fraction of L")
    ap.add_argument("--attempt-cap-mult", type=int, default=40,
                    help="hamming mode: per-branch attempt budget = this * Hamming target")
    ap.add_argument("--max-batch", type=int, default=256, help="max chains per masked forward pass")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heartbeat-every", type=int, default=2000, help="steps between progress logs")
    args = ap.parse_args()

    # Route this run's outputs under a per-mode subdir so attempts/hamming runs don't clobber.
    global OUT_DIR, SEQ_DIR, STATS_CSV, LOG_FILE
    OUT_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / args.target_mode
    SEQ_DIR = OUT_DIR / "sequences"
    STATS_CSV = OUT_DIR / "acceptance_stats.csv"
    LOG_FILE = OUT_DIR / "run.log"

    log = _make_logger()
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    # Import the engine after logging is set up (loading esm/torch is slow and noisy).
    from paper import esm_mcmc

    families = _select_families(args)
    cfg_str = (f"neff={args.neff}" if args.target_mode == "attempts"
               else f"max_hamming_frac={args.max_hamming_frac} attempt_cap_mult={args.attempt_cap_mult}")
    log.info(f"ESM2-MCMC simulation | model={args.model} target_mode={args.target_mode} "
             f"{cfg_str} max_batch={args.max_batch}")
    log.info(f"families ({len(families)}): {', '.join(families)}")
    log.info(f"output -> {OUT_DIR}")
    log.info("attribution: algorithm adapted from Bitbol-Lab Phylogeny-ESM2 (MSAGeneratorESM) "
             "and ESM lm-design (facebookresearch/esm).")

    esm_mcmc.load_model(args.model)
    log.info(f"model loaded on {esm_mcmc._DEVICE}")

    def progress_cb(p: dict) -> None:
        log.info(
            f"  [{p['family']}] step {p['step']:>5} | frontier {p['frontier']:>3} | "
            f"nodes {p['nodes_done']}/{p['n_nodes']} | "
            f"attempts {p['attempts']:>7} accepted {p['accepted']:>6} "
            f"(acc {p['acceptance']:.3f}) | cap-hits {p['cap_hits']:>3} | {p['elapsed']:.0f}s"
        )

    done = 0
    for i, fam in enumerate(families, 1):
        tree = read_tree(os.path.join(str(cfg.TREE_DIR), f"{fam}.txt"))
        root_label, root_seq = next(iter(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")).items()))
        log.info(f"[{i}/{len(families)}] {fam}: L={len(root_seq)} root_label={root_label} "
                 f"| starting simulation")

        seqs, stats = esm_mcmc.simulate_family(
            tree=tree, root_label=root_label, root_seq=root_seq, family=fam,
            model_name=args.model, target_mode=args.target_mode, neff=args.neff,
            max_hamming_frac=args.max_hamming_frac, attempt_cap_mult=args.attempt_cap_mult,
            max_batch=args.max_batch, seed=args.seed, progress_cb=progress_cb,
            heartbeat_every=args.heartbeat_every,
        )

        write_msa(seqs, str(SEQ_DIR / f"{fam}.txt"))
        # Saturation-fair divergence benchmark: compare the simulator's realised identity-to-root
        # to the EMPIRICAL leaves' identity-to-root, not to the raw branch length (which saturates).
        try:
            emp_id = empirical_identity_to_root(fam, root_label)
        except Exception as e:  # pragma: no cover - benchmark is best-effort, never blocks output
            log.info(f"  [{fam}] empirical benchmark failed: {e}")
            emp_id = float("nan")
        _append_stats_row(stats, emp_id)
        done += 1
        cap_note = (f" | cap-hits {stats.n_cap_hits}/{stats.n_branches} branches"
                    if stats.n_cap_hits else "")
        log.info(
            f"[{i}/{len(families)}] {fam}: DONE in {stats.seconds:.0f}s | "
            f"acceptance {stats.acceptance_rate:.3f} "
            f"({stats.total_accepted}/{stats.total_attempts}){cap_note} | "
            f"identity-to-root: ESM-MCMC {stats.mean_leaf_identity_to_root:.3f} vs "
            f"empirical {emp_id:.3f} (branch-length dist {stats.mean_root_to_leaf_distance:.2f} "
            f"subs/site, saturates) | wrote {len(seqs)} records"
        )

    log.info(f"All done: {done}/{len(families)} families -> {SEQ_DIR}")
    log.info(f"Per-family acceptance/divergence stats: {STATS_CSV}")


if __name__ == "__main__":
    main()
