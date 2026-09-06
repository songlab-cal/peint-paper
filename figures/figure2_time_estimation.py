"""Figure 2 — time estimation: PEINT recovers evolutionary time from a pair of sequences.

For held-out families, each (x, y, t) transition has its time re-estimated by maximum
likelihood under PEINT, and compared against the WAG time it was labelled with. Three panels:

``all``    hexbin of WAG time vs. PEINT-estimated time over every transition, with Pearson R.
``single`` the same for two individual families, to show the per-family spread.
``nll``    the PEINT likelihood as a function of time for one representative transition,
           with the WAG time marked — the curve whose argmax the estimate is taking.

Computing the estimates REQUIRES A GPU: the time-MLE optimisation and the likelihood sweep are
PEINT forward passes. Redrawing them does not. `--from-csv` replots the `all` and `single`
panels from the per-transition table, on CPU, in seconds::

    python -m figures.figure2_time_estimation --from-csv          # no GPU, no checkpoint
    python -m figures.figure2_time_estimation --checkpoint <ckpt> # ~4 h on one A100

The table is looked up in the deposit's `figure_data/` first, then in `--output-dir`. The `nll`
panel is not redrawn: it needs the likelihood curve itself, not the table.

Run from the repo root::

    python -m figures.figure2_time_estimation --from-csv
"""

import argparse
import hashlib
import json
import os
import random
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import pearsonr

from peint import caching as peint_caching
from peint.io import read_transitions

# `load_model` and `estimate_transition_times` are imported inside main()'s compute branch:
# both pull in the training stack (lightning), which a plotting-only environment does not
# have. Importing them at module level would make `--from-csv` fail on exactly the machines
# it exists to serve.

import paper_config as cfg

# Families whose per-family spread is shown in the `single` panel.
HIGHLIGHT_FAMILIES = ["2b3y_1_A", "1j1v_1_A"]
HIGHLIGHT_COLORS = ["purple", "green"]

TIME_AXIS_MAX = 1.5


def nll_curve(model, vocab, x: str, y: str, max_time: float, device, num_points: int = 100):
    """Summed forward + reverse NLL of the pair (x, y) across a grid of times.

    Both directions are scored because the model is not symmetric in x and y, and the
    time estimate should not depend on which sequence was called the ancestor.
    """
    times = torch.linspace(0, max_time, num_points, device=device).unsqueeze(-1)

    def encode(seq, trailing_cls: bool):
        toks = [vocab.cls_idx] + vocab.encode(seq) + ([vocab.cls_idx] if trailing_cls else [])
        return torch.tensor(toks, dtype=torch.long, device=device).unsqueeze(0).repeat(times.size())

    def targets(seq):
        return torch.tensor(
            vocab.encode(seq) + [vocab.eos_idx], dtype=torch.long, device=device
        ).unsqueeze(0).repeat(times.size())

    def direction_nll(source: str, dest: str):
        x_toks = encode(source, trailing_cls=True)
        y_toks = encode(dest, trailing_cls=False)
        _, y_logits = model(
            x_toks, y_toks, times, x_toks.eq(vocab.padding_idx), y_toks.eq(vocab.padding_idx)
        )
        return torch.nn.functional.cross_entropy(
            y_logits.transpose(1, 2),
            targets(dest),
            ignore_index=vocab.padding_idx,
            reduction="none",
        ).mean(dim=1)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return direction_nll(x, y) + direction_nll(y, x), times


def held_out_families(families_file, num_families: int) -> List[str]:
    with open(cfg.require(families_file)) as fin:
        families = json.load(fin)["families"]

    seed = int(hashlib.md5(("figure2" + "time_mle").encode("utf-8")).hexdigest(), 16) % (2**32 - 1)
    random.Random(seed).shuffle(families)
    return families[:num_families]


def collect_time_estimates(
    families: List[str], transitions_dir: str, re_estimated_dir: str
) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Pair each transition's WAG time with PEINT's re-estimate.

    Also returns the first transition where the two agree closely in a mid-range time band,
    which is the one the `nll` panel illustrates.
    """
    rows = []
    representative = None
    skipped = {}
    truncated = {}

    for family in families:
        try:
            original = read_transitions(os.path.join(transitions_dir, family + ".txt"))
            re_estimated = read_transitions(os.path.join(re_estimated_dir, family + ".txt"))
        except FileNotFoundError as exc:
            skipped[family] = str(exc)
            continue

        # Re-estimation can drop transitions (a family whose sequences exceed the model's
        # length limit comes back short, occasionally empty). Pair only as far as both
        # sides go, and report it -- indexing `original`'s length into `re_estimated`
        # raises IndexError and discards the whole run, GPU hours included.
        n = min(len(original), len(re_estimated))
        if n < len(original):
            truncated[family] = (len(original), len(re_estimated))

        # Transitions are stored in both directions; take every other one.
        for i in range(0, n, 2):
            wag_t = original[i][2]
            new_t = re_estimated[i][2]
            rows.append(
                [wag_t, new_t, len(original[i][0]) - len(re_estimated[i][1]), family]
            )

            if representative is None and abs(wag_t - new_t) < 0.04 and 0.2 < wag_t < 0.4:
                representative = {"x": original[i][0], "y": original[i][1], "t": wag_t}

    if not rows:
        raise RuntimeError(
            f"No transitions read for any of {len(families)} families "
            f"({len(skipped)} skipped). First: {next(iter(skipped.items()), None)}"
        )
    if skipped:
        print(f"Skipped {len(skipped)}/{len(families)} families with missing transitions.")
    if truncated:
        worst = sorted(truncated.items(), key=lambda kv: kv[1][1] - kv[1][0])[:5]
        print(
            f"WARNING: {len(truncated)}/{len(families)} families came back short from "
            f"re-estimation and were paired only as far as both sides go: "
            + ", ".join(f"{f} ({b}/{a})" for f, (a, b) in worst)
        )

    return (
        pd.DataFrame(rows, columns=["wag_time", "new_time", "length_difference", "family"]),
        representative,
    )


TABLE_NAME = "figure2_time_estimation.csv"


def load_table(output_dir: str) -> pd.DataFrame:
    """Read the per-transition table, preferring the deposited copy."""
    candidates = []
    figure_data = getattr(cfg, "FIGURE_DATA_DIR", None)
    if figure_data:
        candidates.append(os.path.join(str(figure_data), TABLE_NAME))
    candidates.append(os.path.join(output_dir, TABLE_NAME))
    for path in candidates:
        if os.path.exists(path):
            print(f"replotting from {path}")
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"No {TABLE_NAME} found. Looked in: " + ", ".join(candidates) + ". "
        "Fetch the summary data tier, or generate it with a GPU run (no --from-csv)."
    )


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


def _style_time_axes(ax) -> None:
    ax.set_xticks([0, 0.5, 1.0, 1.5], labels=[0, 0.5, 1.0, 1.5])
    ax.set_xticks([0.25, 0.75, 1.25], labels=[], minor=True)
    ax.set_yticks([0, 0.5, 1.0, 1.5], labels=[0, 0.5, 1.0, 1.5])
    ax.set_yticks([0.25, 0.75, 1.25], labels=[], minor=True)
    ax.set_xlim(0, TIME_AXIS_MAX)
    ax.set_ylim(0, TIME_AXIS_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("WAG time", labelpad=0.1)
    ax.set_ylabel("PEINT Estimated time", labelpad=0.1)
    sns.despine(ax=ax)
    ax.tick_params(width=0.25, length=3, which="major")
    ax.tick_params(width=0.25, length=2, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


def _save(fig, output_dir: str, name: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(output_dir, f"{name}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_all_transitions(data: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    # The hexbin's extent clips to TIME_AXIS_MAX, so quote R over the population the panel
    # actually shows. Beyond it the time-MLE has saturated -- estimates top out around 1.29
    # however large the WAG time -- so those points carry no signal to correlate, and
    # including them in a statistic the reader checks against the visible cloud is
    # misleading. Both numbers are printed so the difference is never silent.
    shown = data[(data.wag_time <= TIME_AXIS_MAX) & (data.new_time <= TIME_AXIS_MAX)]
    r_shown = pearsonr(shown.wag_time, shown.new_time)[0]
    r_all = pearsonr(data.wag_time, data.new_time)[0]
    n_out = len(data) - len(shown)
    print(
        f"Pearson R = {r_shown:.4f} over the {len(shown)} transitions inside the axes "
        f"(t <= {TIME_AXIS_MAX}); {r_all:.4f} over all {len(data)}, "
        f"where the {n_out} beyond the axes ({100 * n_out / len(data):.1f}%) are off-panel "
        f"and past the estimator's ceiling."
    )

    ax.hexbin(
        data=data, x="wag_time", y="new_time", cmap="Greens", gridsize=45, mincnt=10,
        extent=(0, TIME_AXIS_MAX, 0, TIME_AXIS_MAX), linewidths=0, vmin=0, vmax=300,
    )
    ax.plot([0, TIME_AXIS_MAX], [0, TIME_AXIS_MAX], color="black", linestyle="--", linewidth=0.25)
    ax.text(
        0.05, 0.95,
        f"Pearson R: {r_shown:.2f}",
        transform=ax.transAxes, fontsize=8, verticalalignment="top", horizontalalignment="left",
        bbox=dict(facecolor="white", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.1"),
    )
    _style_time_axes(ax)
    _save(fig, output_dir, "figure2_time_estimation_all")


def plot_single_families(data: pd.DataFrame, output_dir: str) -> None:
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    for family, color in zip(HIGHLIGHT_FAMILIES, HIGHLIGHT_COLORS):
        subset = data[data.family == family]
        if subset.empty:
            print(f"Note: {family} has no transitions in this run; omitting from the single panel.")
            continue
        ax.scatter(data=subset, x="wag_time", y="new_time", s=10, color=color, alpha=0.5)

    ax.plot([0, TIME_AXIS_MAX], [0, TIME_AXIS_MAX], color="black", linestyle="--", linewidth=0.5)
    _style_time_axes(ax)
    _save(fig, output_dir, "figure2_time_estimation_single")


def plot_nll_curve(nlls, times, wag_time: float, output_dir: str) -> None:
    likelihoods = np.exp(-1 * nlls.cpu().numpy().squeeze())
    times = times.cpu().numpy().squeeze()

    max_likelihood = float(np.max(likelihoods))
    t_argmax = times[np.argmax(likelihoods)]
    min_likelihood_in_range = float(np.min(likelihoods[times < 1.0]))

    fig, ax = plt.subplots(figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    sns.lineplot(x=times, y=likelihoods, ax=ax, color="green", linewidth=0.5, label="PEINT Likelihood")
    sns.scatterplot(
        x=[t_argmax], y=[max_likelihood], ax=ax, color="green", s=5,
        edgecolor="black", linewidth=0.25,
    )
    ax.axvline(
        x=wag_time, color="red", linestyle="--", linewidth=0.5, label=f"WAG time: {wag_time:.2f}"
    )

    ax.set_xlabel("Transition Time", labelpad=0.1)
    ax.set_ylabel("Likelihood", labelpad=0.1)
    ax.set_xticks(np.arange(0, 1, 0.2), labels=[f"{x:.1f}" for x in np.arange(0, 1, 0.2)])
    ax.set_xticks(np.arange(0.1, 1, 0.2), labels=[], minor=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(
        min_likelihood_in_range - min_likelihood_in_range / 10,
        max_likelihood + max_likelihood / 10,
    )
    sns.despine(ax=ax)
    ax.tick_params(width=0.25, length=3, which="major")
    ax.tick_params(width=0.25, length=2, which="minor")
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.legend(loc="upper right", fontsize=6, frameon=False, handlelength=1.5, handletextpad=0.2)

    _save(fig, output_dir, "figure2_time_estimation_nll")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="PEINT checkpoint. Defaults to cfg.PEINT_CHECKPOINT (the model shipped with peint); "
             "point this at your own checkpoint to reproduce the figure with a different model.",
    )
    parser.add_argument("--num-families", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=80)
    parser.add_argument("--max-nll-time", type=float, default=2.0)
    parser.add_argument("--output-dir", default=str(cfg.FIGURES_DIR))
    parser.add_argument(
        "--from-csv", "--replot", dest="from_csv", action="store_true",
        help="Redraw the `all` and `single` panels from the per-transition table. "
             "No GPU, no checkpoint, no cache.",
    )
    args = parser.parse_args()

    # Return before any GPU check, cache setup or checkpoint load: the replot path must work
    # on a laptop with nothing but the summary data tier.
    if args.from_csv:
        data = load_table(args.output_dir)
        print(f"Replotting {len(data)} transitions across {data.family.nunique()} families.")
        _apply_paper_style()
        os.makedirs(args.output_dir, exist_ok=True)
        plot_all_transitions(data, args.output_dir)
        plot_single_families(data, args.output_dir)
        print(f"Wrote time-estimation panels to {args.output_dir}")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Figure 2 time estimation requires a GPU: every panel needs PEINT forward passes. "
            "To redraw the panels from the shipped table instead, pass --from-csv."
        )

    from peint.simulation import load_model
    from peint.time_mle.t_mle import estimate_transition_times

    peint_caching.set_cache_dir("_cache_peint")
    peint_caching.set_read_only(False)

    device = torch.device("cuda")
    checkpoint = args.checkpoint or str(cfg.require(cfg.PEINT_CHECKPOINT))
    transitions_dir = str(cfg.require(cfg.TRANSITIONS_DIR))
    families = held_out_families(cfg.NONTRAIN_FAMILIES_FILE, args.num_families)

    re_estimated_dir = estimate_transition_times(
        transitions_dir=transitions_dir,
        families=families,
        model_checkpoint_path=checkpoint,
        lr=args.lr,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
    )["output_transitions_dir"]

    data, representative = collect_time_estimates(families, transitions_dir, re_estimated_dir)
    print(f"Collected {len(data)} transitions across {data.family.nunique()} families.")

    _apply_paper_style()
    plot_all_transitions(data, args.output_dir)
    plot_single_families(data, args.output_dir)

    if representative is None:
        print("No transition matched the representative band; skipping the NLL panel.")
    else:
        model, vocab = load_model(
            model_checkpoint_path=checkpoint, use_cached_model=False, device=device
        )
        nlls, times = nll_curve(
            model=model, vocab=vocab, x=representative["x"], y=representative["y"],
            max_time=args.max_nll_time, device=device,
        )
        plot_nll_curve(nlls, times, representative["t"], args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    data.to_csv(os.path.join(args.output_dir, "figure2_time_estimation.csv"), index=False)
    print(f"Wrote time-estimation panels to {args.output_dir}")


if __name__ == "__main__":
    main()
