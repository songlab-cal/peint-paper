"""Figure 2 — simulation: PEINT vs WAG vs LG evolving one sequence in a star topology.

A single starting sequence is evolved to a grid of evolutionary times, repeated
``--num-repeats`` times per time point, under three models. Two panels:

``mutations`` evolutionary events per site vs. time. PEINT substitutions and indels are
             counted by Historian (it can produce indels); WAG and LG are substitution-only,
             so their counts are plain Hamming distances from the starting sequence.
``plddt``    OmegaFold pLDDT vs. time, against the starting sequence's pLDDT.

GPU: the PEINT arm (``model.generate``) and all OmegaFold folding require CUDA. The
classical WAG/LG arms are pure CPU, so ``--skip-peint --skip-structures`` produces the
classical half of the mutations panel without a GPU.

Also needs ``historian`` on PATH (for the PEINT arm) and ``omegafold`` on PATH.

Run from the repo root::

    python -m figures.figure2_simulation --skip-peint --skip-structures
"""

import argparse
import hashlib
import os
import random
from typing import Dict, List, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import biotite.structure.io as bsio
import torch
from cherryml import caching as cherryml_caching
from cherryml.benchmarking import pfam_15k
from cherryml.markov_chain import get_lg_path, get_wag_path

from protevo import caching as protevo_caching
from protevo.io import read_msa, read_site_rates, write_msa
from protevo.simulation import load_model
from protevo.simulation.classical import evolve_classical

from paper.historian import analyze_star_topology
from paper.structure_prediction import generate_omegafold_predictions
import paper_config as cfg

# Families deliberately held out of training and moved into the test set.
HELD_OUT_CAS = ["5e2r_1_A", "1ekj_1_C"]
NUM_TRAIN_FAMILIES = 14500

ROOT_RECORD = "initial_sequence"


@protevo_caching.cached_computation(
    output_dirs=["output_sequences_dir"],
    exclude_args=["device"],
    exclude_args_if_default=["num_samples_per_time", "temperature", "p", "max_time", "delta_time"],
)
def simulate_evolution_peint(
    family: str,
    starting_sequence: str,
    model_checkpoint_path: str,
    device: str = "cuda",
    num_samples_per_time: int = 5,
    max_time: float = 1.0,
    delta_time: float = 0.05,
    temperature: float = 1.0,
    p: float = 1.0,
    output_sequences_dir: Optional[str] = None,
):
    """Sample PEINT descendants of one sequence across a grid of evolutionary times."""
    model, vocab = load_model(
        model_checkpoint_path=model_checkpoint_path,
        use_cached_model=True,
        device=device,
    )

    times = _time_grid(max_time, delta_time)
    ts = torch.tensor(times, dtype=torch.float32, device=device).unsqueeze(1)
    x_toks = torch.tensor(
        [vocab.cls_idx] + vocab.encode(starting_sequence) + [vocab.cls_idx],
    ).repeat(ts.size()).to(device)

    all_outputs = {ROOT_RECORD: starting_sequence}
    for i in range(num_samples_per_time):
        outputs = model.generate(
            x=x_toks,
            t=ts,
            max_decode_steps=2 * len(starting_sequence),
            device=device,
            temperature=temperature,
            p=p,
        )
        all_outputs.update(
            {_record_id(family, i, t): output for t, output in zip(times, outputs)}
        )

    os.makedirs(output_sequences_dir, exist_ok=True)
    write_msa(all_outputs, os.path.join(output_sequences_dir, "result.txt"))


@protevo_caching.cached_computation(
    output_dirs=["output_sequences_dir"],
    exclude_args_if_default=["num_samples_per_time", "max_time", "delta_time"],
)
def simulate_evolution_classical(
    family: str,
    starting_sequence: str,
    rate_matrix_path: str,
    num_samples_per_time: int = 5,
    max_time: float = 1.0,
    delta_time: float = 0.05,
    site_rates_path: Optional[str] = None,
    random_seed: Optional[int] = None,
    output_sequences_dir: Optional[str] = None,
):
    """Same star topology as PEINT, but evolved under a classical rate matrix.

    Classical models have no notion of gaps, so the starting sequence is stripped of them
    (and any site rates are subset to the retained columns).
    """
    os.makedirs(output_sequences_dir, exist_ok=True)

    nongap_pos = [i for i, aa in enumerate(starting_sequence) if aa != "-"]
    nongap_sequence = "".join(starting_sequence[i] for i in nongap_pos)

    site_rates = None
    if site_rates_path is not None:
        site_rates = read_site_rates(site_rates_path)
        if len(site_rates) < len(starting_sequence):
            raise ValueError(
                f"{family}: {len(site_rates)} site rates for a sequence of length "
                f"{len(starting_sequence)} in {site_rates_path}."
            )
        site_rates = [site_rates[i] for i in nongap_pos]

    all_outputs = {ROOT_RECORD: nongap_sequence}
    for i in range(num_samples_per_time):
        for t in _time_grid(max_time, delta_time):
            all_outputs[_record_id(family, i, t)] = evolve_classical(
                x=nongap_sequence,
                t=t,
                rate_matrix_path=rate_matrix_path,
                random_seed=None if random_seed is None else random_seed + i,
                site_rates=site_rates,
            )

    write_msa(all_outputs, os.path.join(output_sequences_dir, "result.txt"))


def _time_grid(max_time: float, delta_time: float) -> np.ndarray:
    return np.linspace(delta_time, max_time, int(max_time / delta_time))


def _record_id(family: str, sample: int, t: float) -> str:
    return f"{family}-sample{sample}-{str(t).replace('.', '_')}"


def _parse_record_id(record_id: str) -> tuple:
    """``<family>-sample<i>-<t with _ for .>`` -> (sample, branch_length)."""
    parts = record_id.split("-")
    return parts[1], round(float(parts[-1].replace("_", ".")), 2)


def substitution_rates(sequences: Dict[str, str]) -> pd.DataFrame:
    """Hamming distance per site from the root, summarised per branch length.

    Used for the classical arms, which only ever substitute, so every difference from the
    starting sequence is a substitution.
    """
    root = sequences[ROOT_RECORD]
    rows = []
    for record_id, seq in sequences.items():
        if record_id == ROOT_RECORD:
            continue
        sample, branch_length = _parse_record_id(record_id)
        mutations = sum(1 for a, b in zip(root, seq) if a != b)
        rows.append([branch_length, sample, mutations / len(root)])

    df = pd.DataFrame(rows, columns=["branch_length", "sample", "relative_mutations"])
    return df.groupby("branch_length")["relative_mutations"].agg(["mean", "std"])


def peint_event_rates(sequences: Dict[str, str], historian_path: str) -> pd.DataFrame:
    """Substitution and indel counts per site for the PEINT arm, via Historian."""
    events = analyze_star_topology(
        sequences=sequences,
        root_sequence_name=ROOT_RECORD,
        historian_path=historian_path,
    )
    events["sample"] = events.sequence_id.apply(lambda x: x.split("-")[1])
    events["event_category"] = events["event_type"].map(
        {"substitution": "substitution", "insertion": "indel", "deletion": "indel"}
    )

    # Count distinct (position, category) events per branch/sample, then normalise by the
    # root length so the units match the classical arms.
    event_counts = (
        events.groupby(["branch_length", "sample", "position", "event_category"])
        .size()
        .groupby(["branch_length", "sample", "event_category"])
        .size()
        .unstack("event_category", fill_value=0)
    )
    relative_rates = event_counts.div(len(sequences[ROOT_RECORD]))
    return relative_rates.groupby("branch_length").agg(["mean", "std"])


def collect_plddt(structures_dir: str) -> pd.DataFrame:
    """Mean pLDDT per predicted structure, summarised per branch length."""
    rows = []
    for filename in os.listdir(structures_dir):
        if not filename.endswith(".pdb") or filename.startswith(ROOT_RECORD):
            continue
        structure = bsio.load_structure(
            os.path.join(structures_dir, filename), extra_fields=["b_factor"]
        )
        sample, branch_length = _parse_record_id(filename[: -len(".pdb")])
        rows.append([branch_length, sample, structure.b_factor.mean()])

    df = pd.DataFrame(rows, columns=["branch_length", "sample", "plddt"])
    return df.groupby("branch_length")["plddt"].agg(["mean", "std"])


def root_plddt(structures_dir: str) -> float:
    structure = bsio.load_structure(
        os.path.join(structures_dir, f"{ROOT_RECORD}.pdb"), extra_fields=["b_factor"]
    )
    return float(structure.b_factor.mean())


def pick_family(a3m_dir: str) -> str:
    """Reproduce the paper's deterministic family choice from the held-out test split."""
    all_families = pfam_15k.get_families(pfam_15k_msa_dir=a3m_dir)
    random.Random(42).shuffle(all_families)
    families_test = sorted(all_families[NUM_TRAIN_FAMILIES:] + HELD_OUT_CAS)

    seed = int(
        hashlib.md5(("figure2" + "simulation").encode("utf-8")).hexdigest(), 16
    ) % (2**32 - 1)
    return np.random.RandomState(seed).choice(families_test, 1)[0]


def starting_sequence_for(a3m_dir: str, family: str) -> str:
    """First sequence of the family's a3m — the only one guaranteed to share a length
    between the PEINT and classical arms."""
    with open(os.path.join(a3m_dir, family + ".a3m")) as f:
        f.readline()  # header
        return f.readline().strip()


def _apply_paper_style() -> None:
    sns.set_theme(style="white")
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams["ytick.minor.left"] = True
    plt.rcParams["grid.linewidth"] = 0.5
    plt.rcParams.update(
        {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7}
    )
    mpl.rcParams["pdf.fonttype"] = 42


def _errorbar(ax, index, stats, color, label):
    ax.errorbar(
        index,
        stats["mean"],
        yerr=stats["std"],
        fmt="o",
        capsize=1,
        capthick=0.1,
        elinewidth=0.25,
        markersize=0.75,
        linewidth=0.25,
        color=color,
        label=label,
    )


def plot_mutations(
    peint_rates: Optional[pd.DataFrame],
    classical_rates: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    if peint_rates is not None:
        _errorbar(ax, peint_rates.index, peint_rates["substitution"], "green", "PEINT Substitutions")
    for model, color in (("WAG", "blue"), ("LG", "orange")):
        if model in classical_rates:
            _errorbar(ax, classical_rates[model].index, classical_rates[model], color, model)
    if peint_rates is not None:
        _errorbar(ax, peint_rates.index, peint_rates["indel"], "lightblue", "PEINT Indels")

    ax.set_xlabel("Evolutionary Time", labelpad=0.1)
    ax.set_ylabel("Evolutionary Events / Site", labelpad=0.1)
    ax.legend(frameon=False, handlelength=1, handletextpad=0.3, columnspacing=0.5, fontsize=7)
    sns.despine(ax=ax)
    ax.tick_params(width=0.5, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(output_dir, f"figure2_simulation_mutations.{ext}"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def plot_plddt(
    plddt_stats: Dict[str, pd.DataFrame],
    initial_plddt: float,
    output_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(3, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    colors = {"PEINT": "green", "WAG": "blue", "LG": "orange"}
    for model, stats in plddt_stats.items():
        _errorbar(ax, stats.index, stats, colors[model], f"{model} pLDDT")
    ax.axhline(initial_plddt, color="red", linestyle="--", label="Initial pLDDT", linewidth=0.4)

    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Evolutionary Time", labelpad=0.1)
    ax.set_ylabel("pLDDT", labelpad=0.1)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, handlelength=1, handletextpad=0.3, columnspacing=0.5, fontsize=7)
    sns.despine(ax=ax)
    ax.tick_params(width=0.5, length=2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(output_dir, f"figure2_simulation_plddt.{ext}"),
            bbox_inches="tight",
            dpi=300,
        )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", default=None, help="Family to simulate (default: the paper's).")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="PEINT checkpoint. Defaults to cfg.PEINT_CHECKPOINT (the model shipped with peint); "
             "point this at your own checkpoint to reproduce the figure with a different model.",
    )
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=1.5)
    parser.add_argument("--delta-time", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--historian-path", default="historian")
    parser.add_argument("--skip-peint", action="store_true", help="Skip the PEINT arm (needs a GPU).")
    parser.add_argument(
        "--skip-structures", action="store_true", help="Skip OmegaFold + the pLDDT panel (needs a GPU)."
    )
    parser.add_argument("--output-dir", default=str(cfg.FIGURES_DIR))
    args = parser.parse_args()

    cherryml_caching.set_cache_dir("_cache_cherryml")
    cherryml_caching.set_read_only(False)
    protevo_caching.set_cache_dir("_cache_protevo")
    protevo_caching.set_read_only(False)

    a3m_dir = str(cfg.require(cfg.INPUT_A3M_DIR))
    family = args.family or pick_family(a3m_dir)
    starting_sequence = starting_sequence_for(a3m_dir, family)
    print(f"Family: {family} (starting sequence length {len(starting_sequence)})")

    sim_kwargs = dict(
        family=family,
        starting_sequence=starting_sequence,
        num_samples_per_time=args.num_repeats,
        max_time=args.max_time,
        delta_time=args.delta_time,
    )

    sequence_dirs = {}
    sequence_dirs["WAG"] = simulate_evolution_classical(
        rate_matrix_path=get_wag_path(),
        site_rates_path=None,
        random_seed=args.random_seed,
        **sim_kwargs,
    )["output_sequences_dir"]

    sequence_dirs["LG"] = simulate_evolution_classical(
        rate_matrix_path=get_lg_path(),
        site_rates_path=str(cfg.require(cfg.SITE_RATES_DIR / f"{family}.txt")),
        random_seed=args.random_seed,
        **sim_kwargs,
    )["output_sequences_dir"]

    if not args.skip_peint:
        checkpoint = args.checkpoint or str(cfg.require(cfg.PEINT_CHECKPOINT))
        sequence_dirs["PEINT"] = simulate_evolution_peint(
            model_checkpoint_path=checkpoint,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
            temperature=1.0,
            p=1.0,
            **sim_kwargs,
        )["output_sequences_dir"]

    sequences = {
        model: read_msa(os.path.join(d, "result.txt")) for model, d in sequence_dirs.items()
    }

    classical_rates = {
        model: substitution_rates(sequences[model]) for model in ("WAG", "LG") if model in sequences
    }
    peint_rates = (
        peint_event_rates(sequences["PEINT"], args.historian_path)
        if "PEINT" in sequences
        else None
    )

    _apply_paper_style()
    plot_mutations(peint_rates, classical_rates, args.output_dir)
    print(f"Wrote mutations panel to {args.output_dir}")

    if args.skip_structures:
        print("Skipped OmegaFold and the pLDDT panel (--skip-structures).")
        return

    structure_dirs = {
        model: generate_omegafold_predictions(
            sequences_dir=d,
            family=family,
            input_filename="result.txt",
            keep_prefix=None,
        )["output_structures_dir"]
        for model, d in sequence_dirs.items()
    }

    plddt_stats = {model: collect_plddt(d) for model, d in structure_dirs.items()}
    reference = structure_dirs.get("PEINT") or next(iter(structure_dirs.values()))
    plot_plddt(plddt_stats, root_plddt(reference), args.output_dir)
    print(f"Wrote pLDDT panel to {args.output_dir}")


if __name__ == "__main__":
    main()
