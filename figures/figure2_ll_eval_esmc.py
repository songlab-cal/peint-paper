"""Figure 2 likelihood eval with ESM-C PEINT folded in — held-out AND in-family, one plot each.

Companion to the peint repo's ``figure2_ll_eval.py``. Per-site log-likelihoods of held-out
transitions in bins of evolutionary time, mean per-site likelihood per model:

    Random guess, WAG, LG (4 rate categories), PEINT (ESM2), PEINT (ESM-C)

The classical models + ESM2 PEINT were already scored on this exact split (same seed-42
family split, same transitions, same ``model_checkpoints/peint.ckpt``) and their per-site
outputs live in the peint repo's ``_cache_peint``. We read those cached per-site dirs
directly — the WAG/LG *training* cache keys embed the original machine's hardcoded paths and
no longer resolve, but the per-family *eval* outputs are on disk and are what the plot needs.
Only the ESM-C arm is computed fresh, via the same
``evaluate_peint_model_transitions_log_likelihood__cached`` used for ESM2 (per-family cached,
so a rerun is a no-op once done).

Two ways to produce the panels:

* **Recompute** -- score the transitions. Env: ``peint-esmc`` (Biohub ESM-C backbone +
  sentencepiece); needs a GPU, ``HF_HOME``, a checkpoint, and the
  ``peint_transitions_{aligned,unaligned}`` roles. Run from the repo root::

      HF_HOME=/path/to/hf_cache python -m figures.figure2_ll_eval_esmc

  Each run also writes ``<stem>.csv`` -- the per-(model, time bin) totals behind the panel.

* **Replot** -- ``--from-csv`` redraws both panels from those tables. No GPU, no checkpoint,
  no transitions, so it works at the ``figure_data`` tier::

      python -m figures.figure2_ll_eval_esmc --from-csv

Both paths converge on ``plot_mean_likelihood`` with identical arrays, so a difference
between them is a bug rather than a choice.
"""

import argparse
import glob
import shutil
import subprocess
import json
import os
import random

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
# Editable TrueType text in the PDF for Illustrator (not outlined Type3).
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from cherryml import caching as cherryml_caching
from peint import caching as peint_caching
from peint.evaluation import (
    evaluate_peint_model_transitions_log_likelihood__cached,
)
from peint.io import (
    read_transitions,
    read_transitions_log_likelihood_per_site,
)
from peint.utils import (
    get_quantile_idx,
    get_quantization_points_from_geometric_grid,
)

import paper_config as cfg
from paper.model_style import model_colors

# The peint repo ships the data and the _cache_peint cache alongside the peint
# package, so it is resolved from the installed package rather than hardcoded.
PEINT_REPO = str(cfg.PEINT_REPO)


def _p(*parts):
    """Resolve a peint-repo-relative path, falling back to the data deposit.

    These inputs live in the model repo on the machine that trained the model, and under
    ``LOCAL_DATA/peint/`` for anyone who unpacked them from the deposit. Authoritative tree
    first, so this machine keeps resolving to the exact same strings and the _cache_peint
    hashes -- which key on absolute argument paths -- stay warm.
    """
    here = os.path.join(PEINT_REPO, *parts)
    shipped = os.path.join(str(cfg.LOCAL_DATA), "peint", *parts)
    if cfg.LOCAL_DATA_ONLY:
        return shipped
    if os.path.exists(here):
        return here
    return shipped if os.path.exists(shipped) else here


# Data dirs (in the peint repo), matching figure2_ll_eval.py.
ALIGNED_TEST_TRANSITIONS_DIR = _p("local_data/aligned/test_transitions_dir")
UNALIGNED_TEST_TRANSITIONS_DIR = _p("local_data/unaligned/test_transitions_dir/output_transitions_dir")
UNALIGNED_TEST_ALIGNMENT_MASK_DIR = _p("local_data/unaligned/test_alignment_mask_dir")
ALIGNED_TRAIN_TRANSITIONS_DIR = _p("local_data/aligned/train_transitions_dir")
ALIGNED_TRAIN_SITE_RATES_4CAT_DIR = _p(
    "local_data/aligned/train_site_rates_4cat_dir/output_site_rates_dir")
CACHE_DIR = _p("_cache_peint")
CHERRYML_CACHE_DIR = _p("_cache_cherryml")

# ESM-C PEINT (A3, 60k). Self-describing checkpoint (encoder_backbone="esmc-biohub").
# Not shipped with the paper deposit; set PEINT_PAPER_ESMC_CHECKPOINT to your copy.
ESMC_CHECKPOINT = cfg.ESMC_SIM_CHECKPOINT

# Family split — identical to figure2_ll_eval.build_family_split, kept here so the split is
# reproduced without importing the peint top-level script. Verified to reproduce the exact
# families the cached per-site outputs were computed on (553 test + 553 train-held-out).
HELD_OUT_CAS = ["5e2r_1_A", "1ekj_1_C"]
NUM_TRAIN_FAMILIES = 14500
SPLIT_SEED = 42

# Cached per-site eval functions to READ (not recompute) for the base models. Keyed by the
# display name used in the plot; value is the peint cache subdir of that eval function.
CACHED_EVAL_FUNCS = {
    "Random guess": "evaluate_uniform_random_guess_model_transitions_log_likelihood__cached",
    "WAG": "evaluate_wag_model_transitions_log_likelihood__cached",
    "LG (4 rate categories)": "evaluate_lg_model_transitions_log_likelihood__cached",
    "PEINT (ESM2)": "evaluate_peint_model_transitions_log_likelihood__cached",
}

# Canonical model->color from the shared map, so this panel matches every other figure.
_C = model_colors()
MODEL_COLORS = {
    "Random guess": "gray",
    "WAG": _C["WAG"],
    "LG (4 rate categories)": _C["LG4X"],
    "PEINT (ESM2)": _C["PEINT (ESM2)"],
    "PEINT (ESM-C)": _C["PEINT (ESM-C)"],
}
# Plot order (Random first as the floor, ESM-C last).
PLOT_ORDER = ["Random guess", "WAG", "LG (4 rate categories)", "PEINT (ESM2)", "PEINT (ESM-C)"]

# The two panels this module emits. Shared by the recompute and the --from-csv paths so the
# labels and file stems cannot drift apart.
PANEL_SPECS = (
    ("Test families (held out)", "figure2_likelihood_eval_test_esmc"),
    ("Train held-out subset (in-family)", "figure2_likelihood_eval_train_held_out_esmc"),
)


def build_family_split(transitions_dir):
    all_families = sorted(
        f[: -len(".txt")] for f in os.listdir(transitions_dir) if f.endswith(".txt")
    )
    random.Random(SPLIT_SEED).shuffle(all_families)
    families_train = sorted(all_families[:NUM_TRAIN_FAMILIES])
    families_test = sorted(all_families[NUM_TRAIN_FAMILIES:])
    families_test = sorted(families_test + HELD_OUT_CAS)
    families_train = [f for f in families_train if f not in HELD_OUT_CAS]
    subset_idx = random.Random(SPLIT_SEED).sample(
        range(len(families_train)), len(families_test)
    )
    train_held_out_subset = [families_train[i] for i in subset_idx]
    assert len(families_train) == 14498, len(families_train)
    assert len(families_test) == 553, len(families_test)
    return families_train, families_test, train_held_out_subset


def discover_cached_per_site_dir(func_subdir, needed_families, exclude=()):
    """Find the cached per-site dir for an eval function that covers all needed families.

    The peint _cache_peint holds one output_...per_site_dir per (args) hash; several may
    exist (smoke subsets, full runs). Pick the one whose family files are a superset of the
    families this plot needs; raise if none is.

    ``exclude`` drops known-irrelevant dirs. It matters for the PEINT eval function, which
    is shared by the ESM2 and ESM-C checkpoints: both write a full-coverage dir under the
    same subdir, so without excluding the ESM-C one the "PEINT (ESM2)" curve can silently
    become a second ESM-C curve. Ambiguity that ``exclude`` does not resolve is an error
    rather than a coin flip.
    """
    needed = set(needed_families)
    exclude = {os.path.realpath(e) for e in exclude}
    pattern = os.path.join(
        CACHE_DIR, func_subdir, "*/*/*/*/output_transitions_log_likelihood_per_site_dir"
    )
    covering = []
    for entry in sorted(glob.glob(pattern)):
        if os.path.realpath(entry) in exclude:
            continue
        have = {f[: -len(".txt")] for f in os.listdir(entry) if f.endswith(".txt")}
        if needed <= have:
            covering.append((entry, len(have)))
    if not covering:
        raise FileNotFoundError(
            f"No cached per-site dir under {func_subdir} covers all {len(needed)} families."
        )
    if len(covering) > 1:
        listing = "\n  ".join(f"{e} ({n} families)" for e, n in covering)
        raise RuntimeError(
            f"{len(covering)} cached per-site dirs under {func_subdir} cover all "
            f"{len(needed)} families, so the choice would depend on glob order:\n  "
            f"{listing}\nPass the intended one explicitly (--peint-esm2-per-site-dir)."
        )
    return covering[0][0]


def usable_mpi_processes(requested):
    """Largest process count `mpirun` will actually accept here, at most `requested`.

    cherryml fits the WAG/LG rate matrices by shelling out to `mpirun -np N`. If the machine
    or the Slurm allocation has fewer slots than N, Open MPI refuses to launch, `os.system`
    swallows the message, the counting step writes nothing, and the run dies much later with
    a CacheUsageError about a missing result.txt. That is a miserable thing to debug, so probe
    for it up front and step down instead.
    """
    if requested <= 1:
        return 1
    if shutil.which("mpirun") is None:
        return 1
    n = requested
    while n > 1:
        probe = subprocess.run(["mpirun", "-np", str(n), "true"],
                               capture_output=True, timeout=120)
        if probe.returncode == 0:
            break
        n //= 2
    if n != requested:
        print(f"  NOTE: mpirun cannot allocate {requested} slots here; using {n}. "
              f"Ask for more cores, or pass --num-processes {n} to skip this probe.")
    return n


def compute_baseline_per_site_dir(name, families, families_train, num_processes):
    """Fit and score a classical baseline, rather than looking one up in a warm cache.

    Everything these need is in the deposit: the rate matrices are fit on
    aligned/train_transitions_dir (LG additionally on train_site_rates_4cat_dir), then scored
    on the held-out test transitions. Cached like everything else, so this is paid once.
    """
    from peint import models

    num_processes = usable_mpi_processes(num_processes)

    if name == "Random guess":
        return models.evaluate_uniform_random_guess_model_transitions_log_likelihood__cached(
            transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR, families=families,
        )["output_transitions_log_likelihood_per_site_dir"]

    if name == "WAG":
        model_dir = models.train_wag_model__cached(
            train_transitions_dir=ALIGNED_TRAIN_TRANSITIONS_DIR,
            families=families_train, num_processes=num_processes,
        )["output_model_dir"]
        return models.evaluate_wag_model_transitions_log_likelihood__cached(
            transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR, families=families,
            model_dir=model_dir, num_processes=num_processes, condition_on_non_gap=True,
        )["output_transitions_log_likelihood_per_site_dir"]

    if name == "LG (4 rate categories)":
        model_dir = models.train_lg_model__cached(
            train_transitions_dir=ALIGNED_TRAIN_TRANSITIONS_DIR,
            train_site_rates_dir=ALIGNED_TRAIN_SITE_RATES_4CAT_DIR,
            families=families_train, num_processes=num_processes,
        )["output_model_dir"]
        return models.evaluate_lg_model_transitions_log_likelihood__cached(
            transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
            site_rates_dir=ALIGNED_TRAIN_SITE_RATES_4CAT_DIR, families=families,
            model_dir=model_dir, num_processes=num_processes, condition_on_non_gap=True,
        )["output_transitions_log_likelihood_per_site_dir"]

    raise KeyError(f"no compute path for baseline {name!r}")


def peint_per_site_dir(checkpoint, families, device, batch_size):
    """Compute (cached) PEINT per-site log-likelihoods for these families."""
    return evaluate_peint_model_transitions_log_likelihood__cached(
        transitions_dir=UNALIGNED_TEST_TRANSITIONS_DIR,
        aligned_transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
        alignment_mask_dir=UNALIGNED_TEST_ALIGNMENT_MASK_DIR,
        model_checkpoint_path=checkpoint,
        families=families,
        device=device,
        batch_size=batch_size,
    )["output_transitions_log_likelihood_per_site_dir"]


def esmc_per_site_dir(families, device, batch_size):
    return peint_per_site_dir(ESMC_CHECKPOINT, families, device, batch_size)


def accumulate_by_time_bin(families, per_site_dirs, quantization_points):
    """Total log-likelihood and site count per model, binned by evolutionary time.

    Transitions any model left unscored (NaN) are dropped for every model, so the comparison
    always runs over the same transitions (mirrors figure2_ll_eval.accumulate_by_time_bin).
    """
    totals = {name: np.zeros(len(quantization_points)) for name in per_site_dirs}
    counts = {name: np.zeros(len(quantization_points), dtype=int) for name in per_site_dirs}
    for family in tqdm(families, desc="binning"):
        transitions = read_transitions(os.path.join(ALIGNED_TEST_TRANSITIONS_DIR, family + ".txt"))
        per_site = {
            name: np.array(
                read_transitions_log_likelihood_per_site(os.path.join(d, family + ".txt"))
            )
            for name, d in per_site_dirs.items()
        }
        scored = ~np.isnan(np.stack(list(per_site.values()))).any(axis=(0, 2))
        for i, (_, _, t) in enumerate(transitions):
            if not scored[i]:
                continue
            bin_idx = get_quantile_idx(quantization_points, t)
            for name, ll in per_site.items():
                totals[name][bin_idx] += ll[i].sum()
                counts[name][bin_idx] += ll.shape[1]
    return totals, counts


def panel_table(totals, counts, quantization_points):
    """The tidy table behind one panel: one row per (model, time bin).

    ``total_ll`` and ``n_sites`` are exactly what ``plot_mean_likelihood`` consumes, so the
    table is a lossless record of the panel rather than a summary of it. The mean per-site
    likelihood it actually draws -- exp(total_ll / n_sites) -- is stored alongside so the
    file is readable on its own.
    """
    rows = []
    for name in PLOT_ORDER:
        if name not in totals:
            continue
        for i, t in enumerate(quantization_points):
            n = int(counts[name][i])
            rows.append({
                "model": name,
                "t_bin": float(t),
                "total_ll": float(totals[name][i]),
                "n_sites": n,
                "mean_per_site_likelihood": float(np.exp(totals[name][i] / n)) if n else np.nan,
            })
    return pd.DataFrame(rows)


def table_to_totals_counts(df):
    """Inverse of :func:`panel_table` -- rebuild what ``plot_mean_likelihood`` needs.

    The time grid is recovered from the table itself rather than recomputed, so a replot does
    not depend on the quantization grid still matching the run that produced the table.
    """
    quantization_points = sorted(float(t) for t in df["t_bin"].unique())
    index = {t: i for i, t in enumerate(quantization_points)}
    totals, counts = {}, {}
    for name, sub in df.groupby("model", sort=False):
        totals[name] = np.zeros(len(quantization_points))
        counts[name] = np.zeros(len(quantization_points), dtype=int)
        for t, total_ll, n_sites in zip(sub["t_bin"], sub["total_ll"], sub["n_sites"]):
            i = index[float(t)]
            totals[name][i] = float(total_ll)
            counts[name][i] = int(n_sites)
    return totals, counts, quantization_points


def load_panel_table(stem, out_dir):
    """Load the table this module writes, preferring the deposited copy.

    Mirrors ``figure3_structure_metrics._from_figure_data``: the same file ships in the
    deposit's ``figure_data`` tier, so the replot path and the recompute path converge on
    ``plot_mean_likelihood`` with identical arrays -- there is one plotting implementation,
    not two.
    """
    for cand in (os.path.join(str(cfg.FIGURE_DATA_DIR), stem + ".csv"),
                 os.path.join(out_dir, stem + ".csv")):
        if os.path.exists(cand):
            print(f"replotting {stem} from {cand}")
            return pd.read_csv(cand)
    raise SystemExit(
        f"No saved table for {stem}. Looked in {cfg.FIGURE_DATA_DIR} and {out_dir}. "
        f"Fetch the figure_data tier, or run without --from-csv to recompute it."
    )


def plot_mean_likelihood(totals, counts, quantization_points, output_path, title):
    sns.set_theme(style="white")
    plt.rcParams["xtick.bottom"] = True
    plt.rcParams["ytick.left"] = True
    plt.rcParams.update(
        {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
         "pdf.fonttype": 42, "ps.fonttype": 42}
    )
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    max_x = 0
    for name in PLOT_ORDER:
        if name not in totals:
            continue
        populated = counts[name] > 0
        xs = np.array(quantization_points)[populated]
        ys = np.exp(totals[name][populated] / counts[name][populated])
        if len(xs) > 0:
            max_x = max(max_x, xs.max())
        style = {"linewidth": 0.9, "legend": False, "color": MODEL_COLORS.get(name)}
        if name == "Random guess":
            style["linestyle"] = "--"
        sns.lineplot(x=xs, y=ys, label=name, ax=ax, **style)

    ticks_base = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    labels_base = [f"{t:.1f}" if i in [0, 4, 9] else "" for i, t in enumerate(ticks_base)]
    extra_ticks = list(range(2, int(max_x) + 1))
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.set_xlabel("Evolutionary time")
    ax.set_ylabel("Mean per-site likelihood")
    ax.set_title(title, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1e-1, max(max_x, 1.0))
    ax.set_xticks(ticks_base + extra_ticks, labels=labels_base + [str(t) for t in extra_ticks])
    ax.legend(fontsize=6.5, loc="upper right", frameon=True, ncol=1)
    ax.grid(which="major", axis="y", linestyle=":", linewidth=0.5)
    sns.despine(ax=ax, top=True, right=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    fig.savefig(output_path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {output_path} (+ .png)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--esmc-checkpoint", default=ESMC_CHECKPOINT)
    ap.add_argument("--limit-families", type=int, default=None,
                    help="Use only the first N families of each set (smoke test).")
    ap.add_argument("--no-esmc", action="store_true", help="Base models only (skip ESM-C).")
    ap.add_argument("--num-processes", type=int, default=4,
                    help="Processes for fitting/scoring the WAG and LG baselines.")
    ap.add_argument("--families-path", default=None,
                    help="JSON with explicit in_family / held_out_family lists, in the same "
                         "shape as generate_all_results --families_path. Omit to use the "
                         "paper's derived split (14,498 / 553).")
    ap.add_argument("--peint-esm2-per-site-dir", default=None,
                    help="Pin the ESM2 PEINT cached per-site dir instead of discovering it. "
                         "Needed with --no-esmc, where the ESM-C dir cannot be excluded "
                         "automatically and both cover every family.")
    ap.add_argument("--out-dir", default=str(cfg.FIGURES_DIR))
    ap.add_argument("--from-csv", "--replot", dest="from_csv", action="store_true",
                    help="Redraw both panels from the saved per-(model, time bin) tables "
                         "instead of scoring transitions. Same plotting code, same figure; "
                         "needs no GPU, no checkpoint and no transitions.")
    args = ap.parse_args()

    # Replot first: this path must not touch the transitions dirs, a checkpoint or a cache,
    # because the whole point is that it runs at the figure_data tier.
    if args.from_csv:
        os.makedirs(args.out_dir, exist_ok=True)
        for label, stem in PANEL_SPECS:
            totals, counts, quantization_points = table_to_totals_counts(
                load_panel_table(stem, args.out_dir))
            plot_mean_likelihood(totals, counts, quantization_points,
                                 os.path.join(args.out_dir, stem + ".pdf"), label)
        return

    peint_caching.set_cache_dir(CACHE_DIR)
    # cherryml has its own cache, and fitting the WAG/LG rate matrices goes through it. Without
    # this its cached functions hand back None output dirs and training dies in os.stat.
    cherryml_caching.set_cache_dir(CHERRYML_CACHE_DIR)
    cherryml_caching.set_read_only(False)
    peint_caching.set_read_only(False)

    if args.families_path:
        # Same JSON convention as benchmarks/generate_all_results --families_path:
        #   in_family        seen in training -> the matrices are fit on these, and they are
        #                    the "train held-out subset" curve
        #   held_out_family  never seen -> the held-out curve
        # Giving the families explicitly is what makes a reduced run reproducible: the file
        # is the record, so nothing depends on directory order or a subsample rule.
        with open(cfg.require(args.families_path)) as fh:
            spec = json.load(fh)
        families_train = sorted(spec["in_family"])
        families_test = sorted(spec["held_out_family"])
        train_held_out_subset = families_train
        print(f"Using {args.families_path}: {len(families_train)} in-family, "
              f"{len(families_test)} held-out.")
    else:
        families_train, families_test, train_held_out_subset = build_family_split(
            ALIGNED_TEST_TRANSITIONS_DIR)
    if args.limit_families is not None:
        families_test = families_test[: args.limit_families]
        train_held_out_subset = train_held_out_subset[: args.limit_families]
    all_families = sorted(set(families_test) | set(train_held_out_subset))
    print(f"test={len(families_test)} in-family={len(train_held_out_subset)}")

    # ESM-C first (per-family cached under the ESM-C checkpoint key, so a rerun is a
    # no-op): its dir has to be known before the ESM2 lookup, because both checkpoints
    # cache under the same PEINT eval function and both now cover every family.
    per_site_dirs = {}
    esmc_dir = None
    if not args.no_esmc:
        print("Scoring ESM-C PEINT (cached) ...")
        esmc_dir = esmc_per_site_dir(all_families, args.device, args.batch_size)
        print(f"  PEINT (ESM-C)            <- {esmc_dir}")

    # Base models: read the cached per-site dirs directly (one dir serves both family sets).
    for name, subdir in CACHED_EVAL_FUNCS.items():
        if name == "PEINT (ESM2)" and args.peint_esm2_per_site_dir:
            d = args.peint_esm2_per_site_dir
        else:
            try:
                d = discover_cached_per_site_dir(
                    subdir, all_families, exclude=[esmc_dir] if esmc_dir else ()
                )
            except FileNotFoundError:
                print(f"  {name}: no cached per-site dir; computing from the transitions ...")
                if name == "PEINT (ESM2)":
                    d = peint_per_site_dir(str(cfg.PEINT_CHECKPOINT), all_families,
                                           args.device, args.batch_size)
                else:
                    d = compute_baseline_per_site_dir(
                        name, all_families, families_train, args.num_processes)
        per_site_dirs[name] = d
        print(f"  {name:24s} <- {d[len(CACHE_DIR) + 1:][:64]}...")

    if esmc_dir is not None:
        per_site_dirs["PEINT (ESM-C)"] = esmc_dir

    quantization_points = [float(q) for q in get_quantization_points_from_geometric_grid()]

    panel_families = {
        "figure2_likelihood_eval_test_esmc": families_test,
        "figure2_likelihood_eval_train_held_out_esmc": train_held_out_subset,
    }
    for label, stem in PANEL_SPECS:
        families = panel_families[stem]
        print(f"\n=== {label}: binning {len(families)} families ===")
        totals, counts = accumulate_by_time_bin(families, per_site_dirs, quantization_points)
        for name in PLOT_ORDER:
            if name not in totals:
                continue
            sites = counts[name].sum()
            mean_ll = totals[name].sum() / sites if sites else float("nan")
            print(f"  {name:24s} mean per-site LL = {mean_ll:+.4f} over {sites} sites")
        os.makedirs(args.out_dir, exist_ok=True)
        # Save the table BEFORE plotting, so an expensive run is never lost to a plotting
        # error, and so the panel can be redrawn later with --from-csv. This file is what the
        # deposit should ship in figure_data (a few KB) -- see FINDINGS P1-8.
        table_path = os.path.join(args.out_dir, stem + ".csv")
        panel_table(totals, counts, quantization_points).to_csv(table_path, index=False)
        print(f"Wrote {table_path}")
        plot_mean_likelihood(
            totals, counts, quantization_points, os.path.join(args.out_dir, stem + ".pdf"), label
        )


if __name__ == "__main__":
    main()
