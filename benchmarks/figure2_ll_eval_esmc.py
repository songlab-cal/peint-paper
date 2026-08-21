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

Env: ``peint-esmc`` (has the Biohub ESM-C backbone + sentencepiece). Needs a GPU and
``HF_HOME`` set for the ESM-C backbone. Run from anywhere::

    HF_HOME=/scratch/users/akoehl/hf_cache \
      python /scratch/users/akoehl/peint-paper/benchmarks/figure2_ll_eval_esmc.py
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import matplotlib as mpl
mpl.use("Agg")
# Editable TrueType text in the PDF for Illustrator (not outlined Type3).
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# The peint repo ships the data, the _cache_peint cache, and the protevo package.
PEINT_REPO = "/scratch/users/akoehl/peint"
sys.path.insert(0, PEINT_REPO)

from protevo import caching as protevo_caching  # noqa: E402
from protevo.evaluation import (  # noqa: E402
    evaluate_peint_model_transitions_log_likelihood__cached,
)
from protevo.io import (  # noqa: E402
    read_transitions,
    read_transitions_log_likelihood_per_site,
)
from protevo.utils import (  # noqa: E402
    get_quantile_idx,
    get_quantization_points_from_geometric_grid,
)


def _p(*parts):
    return os.path.join(PEINT_REPO, *parts)


# Data dirs (in the peint repo), matching figure2_ll_eval.py.
ALIGNED_TEST_TRANSITIONS_DIR = _p("local_data/aligned/test_transitions_dir")
UNALIGNED_TEST_TRANSITIONS_DIR = _p("local_data/unaligned/test_transitions_dir/output_transitions_dir")
UNALIGNED_TEST_ALIGNMENT_MASK_DIR = _p("local_data/unaligned/test_alignment_mask_dir")
CACHE_DIR = _p("_cache_peint")
CHERRYML_CACHE_DIR = _p("_cache_cherryml")

# ESM-C PEINT (A3, 60k). Self-describing checkpoint (encoder_backbone="esmc-biohub").
ESMC_CHECKPOINT = (
    "/scratch/users/yufan.cao/protevo_ablations/esmc/"
    "20260729-5e5d20h960d-esmc-14498fams-esmc/epoch=4-step=60000.ckpt"
)

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

# Canonical model->color, matched to the paper's ESM-C figures (seaborn "deep" indices:
# WAG=0, LG4X=3, PEINT (ESM2)=2 green, PEINT (ESM-C)=9 cyan).
_DEEP = sns.color_palette("deep")
MODEL_COLORS = {
    "Random guess": "gray",
    "WAG": _DEEP[0],
    "LG (4 rate categories)": _DEEP[3],
    "PEINT (ESM2)": _DEEP[2],
    "PEINT (ESM-C)": _DEEP[9],
}
# Plot order (Random first as the floor, ESM-C last).
PLOT_ORDER = ["Random guess", "WAG", "LG (4 rate categories)", "PEINT (ESM2)", "PEINT (ESM-C)"]


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


def discover_cached_per_site_dir(func_subdir, needed_families):
    """Find the cached per-site dir for an eval function that covers all needed families.

    The peint _cache_peint holds one output_...per_site_dir per (args) hash; several may
    exist (smoke subsets, full runs). Pick the one whose family files are a superset of the
    families this plot needs; raise if none is.
    """
    needed = set(needed_families)
    pattern = os.path.join(
        CACHE_DIR, func_subdir, "*/*/*/*/output_transitions_log_likelihood_per_site_dir"
    )
    best, best_have = None, -1
    for entry in glob.glob(pattern):
        have = {f[: -len(".txt")] for f in os.listdir(entry) if f.endswith(".txt")}
        if needed <= have and len(have) > best_have:
            best, best_have = entry, len(have)
    if best is None:
        raise FileNotFoundError(
            f"No cached per-site dir under {func_subdir} covers all {len(needed)} families."
        )
    return best


def esmc_per_site_dir(families, device, batch_size):
    """Compute (cached) the ESM-C PEINT per-site log-likelihoods for these families."""
    return evaluate_peint_model_transitions_log_likelihood__cached(
        transitions_dir=UNALIGNED_TEST_TRANSITIONS_DIR,
        aligned_transitions_dir=ALIGNED_TEST_TRANSITIONS_DIR,
        alignment_mask_dir=UNALIGNED_TEST_ALIGNMENT_MASK_DIR,
        model_checkpoint_path=ESMC_CHECKPOINT,
        families=families,
        device=device,
        batch_size=batch_size,
    )["output_transitions_log_likelihood_per_site_dir"]


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
    ap.add_argument("--out-dir", default="/scratch/users/akoehl/peint-paper/figures/output")
    args = ap.parse_args()

    protevo_caching.set_cache_dir(CACHE_DIR)
    protevo_caching.set_read_only(False)

    _, families_test, train_held_out_subset = build_family_split(ALIGNED_TEST_TRANSITIONS_DIR)
    if args.limit_families is not None:
        families_test = families_test[: args.limit_families]
        train_held_out_subset = train_held_out_subset[: args.limit_families]
    all_families = sorted(set(families_test) | set(train_held_out_subset))
    print(f"test={len(families_test)} in-family={len(train_held_out_subset)}")

    # Base models: read the cached per-site dirs directly (one dir serves both family sets).
    per_site_dirs = {}
    for name, subdir in CACHED_EVAL_FUNCS.items():
        d = discover_cached_per_site_dir(subdir, all_families)
        per_site_dirs[name] = d
        print(f"  {name:24s} <- {d[len(CACHE_DIR) + 1:][:64]}...")

    # ESM-C: computed fresh (per-family cached under the ESM-C checkpoint key).
    if not args.no_esmc:
        print("Scoring ESM-C PEINT (fresh) ...")
        esmc_dir = esmc_per_site_dir(all_families, args.device, args.batch_size)
        per_site_dirs["PEINT (ESM-C)"] = esmc_dir
        print(f"  PEINT (ESM-C)            <- {esmc_dir}")

    quantization_points = [float(q) for q in get_quantization_points_from_geometric_grid()]

    for label, families, stem in (
        ("Test families (held out)", families_test, "figure2_likelihood_eval_test_esmc"),
        ("Train held-out subset (in-family)", train_held_out_subset, "figure2_likelihood_eval_train_held_out_esmc"),
    ):
        print(f"\n=== {label}: binning {len(families)} families ===")
        totals, counts = accumulate_by_time_bin(families, per_site_dirs, quantization_points)
        for name in PLOT_ORDER:
            if name not in totals:
                continue
            sites = counts[name].sum()
            mean_ll = totals[name].sum() / sites if sites else float("nan")
            print(f"  {name:24s} mean per-site LL = {mean_ll:+.4f} over {sites} sites")
        os.makedirs(args.out_dir, exist_ok=True)
        plot_mean_likelihood(
            totals, counts, quantization_points, os.path.join(args.out_dir, stem + ".pdf"), label
        )


if __name__ == "__main__":
    main()
