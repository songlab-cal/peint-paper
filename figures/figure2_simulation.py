"""Figure 2 — simulation: PEINT vs WAG vs LG evolving one sequence in a star topology.

A single starting sequence is evolved to a grid of evolutionary times, repeated
``--num-repeats`` times per time point, under three models. Two panels:

``mutations`` evolutionary events per site vs. time. PEINT substitutions and indels are
             counted by Historian (it can produce indels); WAG and LG are substitution-only,
             so their counts are plain Hamming distances from the starting sequence.
``plddt``    OmegaFold pLDDT vs. time, against the starting sequence's pLDDT.

By default BOTH released PEINT backbones are simulated (ESM2 and ESM-C) alongside WAG and LG,
so the panel shows PEINT as a family rather than one model. Override with ``--peint``.

GPU: the PEINT arms (``model.generate``) and all OmegaFold folding require CUDA. The classical
WAG/LG arms are pure CPU, so ``--skip-peint --skip-structures`` produces the classical half of
the mutations panel without a GPU.

Also needs Historian (``--historian-path``, defaults to the repo build) to split PEINT events
into substitutions vs indels, and ``omegafold`` on PATH for the pLDDT panel.

Run from the repo root::

    python -m figures.figure2_simulation --skip-structures          # both PEINT arms + WAG/LG
    python -m figures.figure2_simulation --skip-peint --skip-structures
    python -m figures.figure2_simulation --peint 'PEINT (ESM-C)=/path/to.ckpt' --skip-structures
"""

import argparse
import contextlib
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
from paper.model_style import model_colors
from paper.structure_prediction import generate_omegafold_predictions
import paper_config as cfg

# Families deliberately held out of training and moved into the test set.
HELD_OUT_CAS = ["5e2r_1_A", "1ekj_1_C"]
NUM_TRAIN_FAMILIES = 14500

ROOT_RECORD = "initial_sequence"

# The PEINT arms to simulate, as display label -> checkpoint. Both backbones are shown so the
# panel reads as "PEINT on either encoder" rather than one model; --peint overrides.
def default_peint_arms():
    return {
        "PEINT (ESM2)": str(cfg.PEINT_CHECKPOINT),
        "PEINT (ESM-C)": str(cfg.ESMC_SIM_CHECKPOINT),
    }


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
        # bfloat16 autocast, matching protevo.simulation._simulate_on_tree: with Flash
        # Attention the model is a PeintGenerator (cached decoder) and its kernels take only
        # fp16/bf16, so generating in the ambient fp32 raises "FlashAttention only support
        # fp16 and bf16 data type". Dropping to the Vanilla path instead would also throw
        # away the decoder cache, so autocast here rather than passing use_flash=False.
        with _generate_precision(device):
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


def _generate_precision(device):
    """bfloat16 autocast on CUDA, a no-op elsewhere (CPU falls back to the Vanilla fp32 path)."""
    return (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if str(device).startswith("cuda")
        else contextlib.nullcontext()
    )


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


def _errorbar(ax, index, stats, color, label, linestyle="-"):
    ax.errorbar(
        index,
        stats["mean"],
        yerr=stats["std"],
        fmt="o",
        linestyle=linestyle,
        capsize=1,
        capthick=0.1,
        elinewidth=0.25,
        markersize=0.75,
        linewidth=0.25,
        color=color,
        label=label,
    )


def plot_mutations(
    peint_rates: Dict[str, pd.DataFrame],
    classical_rates: Dict[str, pd.DataFrame],
    output_dir: str,
) -> None:
    """Substitutions (solid) and indels (dashed) per site vs time.

    ``peint_rates`` maps a PEINT arm's display label to its rate table, so more than one
    backbone can be shown. Indels are the dashed twin of each arm's own color rather than a
    separate hue, which keeps the legend readable once there are two PEINT arms.
    """
    fig, ax = plt.subplots(figsize=(3.4, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    mc = model_colors()
    for label, rates in (peint_rates or {}).items():
        color = mc.get(label, "green")
        _errorbar(ax, rates.index, rates["substitution"], color, f"{label} subs")
    for model in ("WAG", "LG"):
        if model in classical_rates:
            _errorbar(ax, classical_rates[model].index, classical_rates[model], mc[model], model)
    for label, rates in (peint_rates or {}).items():
        color = mc.get(label, "green")
        _errorbar(ax, rates.index, rates["indel"], color, f"{label} indels", linestyle="--")

    ax.set_xlabel("Evolutionary Time", labelpad=0.1)
    ax.set_ylabel("Evolutionary Events / Site", labelpad=0.1)
    ax.set_ylim(bottom=0)   # counts per site; the default margin dipped below zero
    # Outside the axes: with six series the in-axes legend covered the substitution curves.
    ax.legend(frameon=False, handlelength=1, handletextpad=0.3, columnspacing=0.5,
              fontsize=6.5, loc="upper left", bbox_to_anchor=(1.01, 1.0))
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

    mc = model_colors()
    for model, stats in plddt_stats.items():
        _errorbar(ax, stats.index, stats, mc.get(model, "green"), f"{model} pLDDT")
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
        "--peint",
        action="append",
        metavar="LABEL=PATH",
        default=None,
        help="A PEINT arm to simulate, repeatable, e.g. --peint 'PEINT (ESM-C)=/path/to.ckpt'. "
             "Defaults to both released backbones (see default_peint_arms).",
    )
    parser.add_argument("--num-repeats", type=int, default=10)
    parser.add_argument("--max-time", type=float, default=1.5)
    parser.add_argument("--delta-time", type=float, default=0.05)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--historian-path", default=str(cfg.HISTORIAN_PATH),
                        help="Historian binary; needed only to split PEINT events into "
                             "substitutions vs indels.")
    parser.add_argument("--skip-peint", action="store_true", help="Skip the PEINT arm (needs a GPU).")
    parser.add_argument(
        "--skip-structures", action="store_true", help="Skip OmegaFold + the pLDDT panel (needs a GPU)."
    )
    parser.add_argument("--output-dir", default=str(cfg.FIGURES_DIR))
    args = parser.parse_args()

    # Absolute, from config: these were relative, which tied the script to one working
    # directory and silently missed the cache from anywhere else.
    cherryml_caching.set_cache_dir(str(cfg.CHERRYML_CACHE_DIR))
    cherryml_caching.set_read_only(False)
    protevo_caching.set_cache_dir(str(cfg.PROTEVO_CACHE_DIR))
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

    peint_arms = {}
    if not args.skip_peint:
        if args.peint:
            for spec in args.peint:
                if "=" not in spec:
                    parser.error(f"--peint expects LABEL=PATH, got {spec!r}")
                label, path = spec.split("=", 1)
                peint_arms[label.strip()] = path.strip()
        else:
            peint_arms = default_peint_arms()
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        for label, checkpoint in peint_arms.items():
            print(f"Simulating {label} from {checkpoint}")
            sequence_dirs[label] = simulate_evolution_peint(
                model_checkpoint_path=str(cfg.require(checkpoint)),
                device=device,
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
    peint_rates = {
        label: peint_event_rates(sequences[label], args.historian_path)
        for label in peint_arms
    }

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
    reference = next((structure_dirs[k] for k in peint_arms if k in structure_dirs),
                     next(iter(structure_dirs.values())))
    plot_plddt(plddt_stats, root_plddt(reference), args.output_dir)
    print(f"Wrote pLDDT panel to {args.output_dir}")


if __name__ == "__main__":
    main()
