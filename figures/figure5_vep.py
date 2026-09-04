import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import matplotlib as mpl
import scipy
from tqdm import tqdm
import os
import json
from pathlib import Path

from protevo.vep._vep_utils import (
    PROTEINGYM_DIR,
    _format_time_dir_suffix,
    _discover_time_dirs,
)
from paper.plot_style import _set_publication_style
import paper_config as cfg
from paper.model_style import base_lm_pair_colors, model_colors
from paper import vep

# Scored results now live under test_lls/production/ (archived runs under test_lls/archive/).
# Figure functions below resolve `<TEST_LLS_PRODUCTION>/<run_name>/spearman_results.csv`.

"""Consolidated plotting utilities and figure generators for VEP analysis.

All plotting-related helpers are defined in this file to avoid redundant
style configuration across modules. Use `_set_publication_style()` to
apply Illustrator-friendly, publication-ready defaults before plotting.
"""

# Figures and their input data both live inside the peint-paper repo (anchored via
# __file__), git-committed, so nothing is written to the model (protevo) repo:
#   * figures                -> peint-paper/figures/<category>/
#   * per-run scored results -> peint-paper/local_data/vep/test_lls/production/<run>/spearman_results.csv
# The ProteinGym family reference (transition pairs) is still read from the model
# repo's ProteinGym3, which PEINT is meant to be used in conjunction with.
PAPER_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PAPER_ROOT / "figures"
# Via cfg.LOCAL_DATA rather than PAPER_ROOT/local_data: the two are the same by default, but
# only the former moves when someone unpacks the deposit somewhere else.
VEP_RESULTS_DIR = Path(cfg.LOCAL_DATA) / "vep" / "test_lls" / "production"


def _fig_path(category, filename):
    """Resolve figures/<category>/<filename>, creating the subdir if needed."""
    out_dir = FIG_DIR / category
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename


# Assay types in the order the figures present them.
ASSAY_TYPE_ORDER = [
    "OrganismalFitness", "Stability", "Expression", "Activity", "Binding",
]


def _paired_bars(df_plot, hue_order, palette, group_size, alpha):
    """Bars grouped into touching sub-groups of ``group_size`` within each assay type.

    Layout per category: sub-groups of `group_size` bars with no gap inside a sub-group and
    a small gap between sub-groups, so a (base pLM, PEINT) pair reads as one unit. Heights
    are the mean over families and the error bars its standard error — the same statistics
    seaborn's ``estimator=np.mean, errorbar="se"`` produces.
    """
    cats = list(dict.fromkeys(df_plot["assay_type"]))
    stats = df_plot.groupby(["assay_type", "model"])["spearman"].agg(["mean", "sem"])

    n = len(hue_order)
    n_groups = int(np.ceil(n / group_size))
    span = 0.8                      # fraction of the category slot the bars occupy
    gap = 0.05                      # between sub-groups
    bar_w = (span - gap * (n_groups - 1)) / n
    start = -span / 2 + bar_w / 2

    ax = plt.gca()
    for i, model in enumerate(hue_order):
        g, m = divmod(i, group_size)
        off = start + g * (group_size * bar_w + gap) + m * bar_w
        xs, ys, es = [], [], []
        for c_i, cat in enumerate(cats):
            if (cat, model) not in stats.index:
                continue
            row = stats.loc[(cat, model)]
            xs.append(c_i + off)
            ys.append(row["mean"])
            es.append(0.0 if pd.isna(row["sem"]) else row["sem"])
        color = palette[i] if isinstance(palette, (list, tuple)) else (
            palette.get(model) if isinstance(palette, dict) else None)
        ax.bar(xs, ys, width=bar_w, yerr=es, label=model, color=color, alpha=alpha,
               linewidth=0.5, edgecolor="black",
               error_kw={"linewidth": 1.0, "capsize": 2, "capthick": 1.0})
    ax.set_xticks(range(len(cats)))
    ax.set_xticklabels(cats)
    return ax


def _plot_proteingym_spearman_comparision(
    df_results_all: pd.DataFrame,
    model_names: dict | None = None,
    palette: list | dict | None = None,
    figsize: tuple[int, int] = (12, 5),
    dpi: int = 240,
    alpha: float = 0.8,
    group_size: int | None = None,
):
    """Plot ProteinGym average Spearman by assay type with model comparison.

    ``group_size`` bundles consecutive hues into touching sub-groups separated by a small
    gap — pass 2 to read the six-model chart as three (base pLM, PEINT) pairs instead of six
    evenly spaced bars. Seaborn spaces hues uniformly with no hook for this, so that path
    draws the bars itself; the aggregation (mean over families, SE error bars) is identical.

    Style is controlled externally via `_set_publication_style()`.
    """
    df_plot = df_results_all.copy()
    # Assay types read in a fixed, meaningful order rather than order-of-appearance.
    if "assay_type" in df_plot.columns:
        present = [a for a in ASSAY_TYPE_ORDER if a in set(df_plot["assay_type"])]
        present += [a for a in dict.fromkeys(df_plot["assay_type"]) if a not in present]
        df_plot["assay_type"] = pd.Categorical(df_plot["assay_type"], categories=present, ordered=True)
        df_plot = df_plot.sort_values("assay_type")
    hue_order = None
    if model_names is not None:
        df_plot = df_plot[df_plot["model"].isin(model_names.keys())]
        df_plot = df_plot.copy()
        df_plot.loc[:, "model"] = df_plot["model"].map(model_names)
        hue_order = [model_names[k] for k in model_names.keys()]
    else:
        hue_order = list(dict.fromkeys(df_plot["model"].tolist()))

    plt.figure(figsize=figsize, dpi=dpi)
    if group_size:
        ax = _paired_bars(df_plot, hue_order, palette, group_size, alpha)
    else:
        ax = sns.barplot(
            x="assay_type",
            y="spearman",
            hue="model",
            hue_order=hue_order,
            data=df_plot,
            alpha=alpha,
            estimator=np.mean,
            errorbar="se",
            capsize=0.15,
            err_kws={"linewidth": 1.0},
            linewidth=0.5,
            edgecolor="black",
            palette=palette,
        )

    ax.set_xlabel("Assay Type")
    ax.set_ylabel("Mean Spearman Correlation")
    # ax.set_title(
    #     f'ProteinGym Average Spearman by Assay Type: {df_plot["family"].unique().size} Families'
    # )
    ax.set_title(f"Mean Spearman Correlation by Assay Type")
    # Outside the axes: anchored inside (0.8) the box covered the last assay group.
    leg = ax.legend(
        title="Model: Mean Spearman",
        bbox_to_anchor=(1.01, 1.0),
        loc="upper left",
        frameon=True,
    )
    for legline in leg.legend_handles:
        try:
            legline.set_alpha(0.9)
        except Exception:
            pass

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return ax


def _make_spearman_plot(
    run_names: dict[str, dict],
    model_names: dict[str, str],
    figsize: tuple[int, int] = (8, 3),
    palette: list | dict | None = None,
    save_path: str | None = None,
    alpha: float = 0.8,
    save_spearman_path: str | None = None,
    group_size: int | None = None,
):
    output_dir = VEP_RESULTS_DIR
    if save_spearman_path is None:
        save_spearman_path = output_dir / "spearman_results.csv"
    # Load all families
    transition_dir = (
        PROTEINGYM_DIR
        / "_cache_cherryml/create_test_transition_pairs/3a13efc22507796bfec09150d959917b6e034c3c29639e25c0cadf0f02921d4c/output_transition_pairs_dir/"
    )
    families = [
        f.split(".")[0] for f in os.listdir(transition_dir) if f.endswith(".txt")
    ]

    df_results_all = []
    families_overlap = set(families)
    for name, info in run_names.items():
        run_dir = info.get("run_name")
        t_wag = info.get("t_wag")
        if t_wag is None:
            per_assay_spearman_fpath = output_dir / run_dir / "spearman_results.csv"
        else:
            # Convert provided t_wag to dotted string per spec
            t_wag_str = str(t_wag).replace(".", "_")
            per_assay_spearman_fpath = (
                output_dir
                / run_dir
                / f"wag_corrected_spearman_results_t_{t_wag_str}.csv"
            )
        if not per_assay_spearman_fpath.exists():
            print(f"{name}: {per_assay_spearman_fpath} not exists")
        df = pd.read_csv(per_assay_spearman_fpath)
        df["model"] = name
        df_results_all.append(df)
        families_overlap = set(df["family"]).intersection(families_overlap)
    df_results_all = pd.concat(df_results_all, ignore_index=True)
    df_results_all = df_results_all[
        df_results_all["family"].isin(families_overlap)
    ].reset_index(drop=True)

    if save_spearman_path is not None:
        df_results_all.to_csv(save_spearman_path, index=False)

    # Compute average Spearman per model by averaging the mean Spearman per assay_type
    df_means_by_assay = df_results_all.groupby(["model", "assay_type"], as_index=False)[
        "spearman"
    ].mean()
    df_model_avg = df_means_by_assay.groupby("model", as_index=False)["spearman"].mean()
    # Append the average to the display model label directly, so legend shows it
    df_model_avg["model_display"] = df_model_avg["model"].map(model_names)
    avg_map = {
        k: f"{model_names[k]}: {v:.3f}"
        for k, v in zip(
            df_model_avg["model"].tolist(), df_model_avg["spearman"].tolist()
        )
    }
    # Rebuild plotting dataframe with updated model labels containing averages
    df_plot = df_results_all.copy()
    df_plot = df_plot[df_plot["model"].isin(model_names.keys())].copy()
    df_plot.loc[:, "model"] = df_plot["model"].map(avg_map)

    # Now call the plotting function on the modified labels
    ax = _plot_proteingym_spearman_comparision(
        df_plot, model_names=None, palette=palette, figsize=figsize, alpha=alpha,
        group_size=group_size,
    )

    # Standardize axis aesthetics
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(True, alpha=0.3, which="both", linewidth=0.25)

    if save_path is not None:
        save_path = Path(save_path)
        fig = ax.get_figure()
        fig.tight_layout()
        # Save PNG (preserve existing behavior)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        # Also save a PDF twin (vector, Illustrator-editable text)
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path} and {pdf_path}")

    plt.show()


def _plot_spearman_by_mutational_depth(
    df_results_all: pd.DataFrame,
    model_names: dict | None = None,
    palette: list | dict | None = None,
    figsize: tuple[int, int] = (8, 5),
    dpi: int = 240,
    alpha: float = 0.8,
):
    """Plot Spearman correlation by mutational depth with model comparison.

    For each mutational depth, shows mean Spearman across families with error bars.
    """
    df_plot = df_results_all.copy()
    hue_order = None
    if model_names is not None:
        df_plot = df_plot[df_plot["model"].isin(model_names.keys())]
        df_plot = df_plot.copy()
        df_plot.loc[:, "model"] = df_plot["model"].map(model_names)
        hue_order = [model_names[k] for k in model_names.keys()]
    else:
        hue_order = list(dict.fromkeys(df_plot["model"].tolist()))

    # Define mutational depth order (numeric order with "5+" at the end)
    depth_order = sorted(
        [d for d in df_plot["mutational_depth"].unique() if d != "5+"],
        key=lambda x: int(x) if isinstance(x, str) and x.isdigit() else float(x),
    )
    if "5+" in df_plot["mutational_depth"].unique():
        depth_order.append("5+")

    plt.figure(figsize=figsize, dpi=dpi)
    ax = sns.pointplot(
        x="mutational_depth",
        y="spearman",
        hue="model",
        hue_order=hue_order,
        data=df_plot,
        order=depth_order,
        estimator=np.mean,
        errorbar="se",
        capsize=0.1,
        markers="o",
        linestyles="-",
        dodge=0.2,
        palette=palette,
        alpha=alpha,
    )

    ax.set_xlabel("Mutational Depth")
    ax.set_ylabel("Mean Spearman Correlation")
    ax.set_title("Mean Spearman Correlation by Mutational Depth")
    leg = ax.legend(
        title="Model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=True,
    )
    for legline in leg.legend_handles:
        try:
            legline.set_alpha(0.9)
        except Exception:
            pass

    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    return ax


def _make_esm_vs_peint_plot(
    run_names: dict[str, dict],
    model_names: dict[str, str],
    figsize: tuple[int, int] = (6, 5),
    palette: list | dict | None = None,
    save_path: str | None = None,
    alpha: float = 0.8,
    dot_size: int = 50,
    assay_type: str | None = "OrganismalFitness",
):
    """Create a scatterplot comparing ESM vs PEINT performance per family.

    Each family is represented as a dot with ESM Spearman on x-axis and PEINT
    Spearman on y-axis. If assay_type is None, different assay types are colored
    with different hues. If assay_type is specified, only that assay type is shown
    with uniform coloring and no legend.

    Parameters
    ----------
    run_names : dict[str, str]
        Dictionary mapping model names to their run directory names.
        Must contain exactly two models: one ESM and one PEINT.
    model_names : dict[str, str]
        Mapping from run_names keys to display names for the plot.
    figsize : tuple[int, int]
        Figure size for the plot. Default is (8, 6).
    palette : list | dict | None
        Optional color palette for assay types. If None, uses default seaborn palette.
    save_path : str | None
        If provided, save the plot to this path (PNG and PDF).
    alpha : float
        Alpha transparency for scatter points. Default is 0.8.
    dot_size : int
        Size of the scatter points. Default is 50.
    assay_type : str | None
        If specified, only show data for this assay type. If None, show all assay types
        with different colors. Default is "OrganismalFitness".
    """
    if len(run_names) != 2:
        raise ValueError("run_names must contain exactly two models (ESM and PEINT)")

    output_dir = VEP_RESULTS_DIR

    # Load all families
    transition_dir = (
        PROTEINGYM_DIR
        / "_cache_cherryml/create_test_transition_pairs/3a13efc22507796bfec09150d959917b6e034c3c29639e25c0cadf0f02921d4c/output_transition_pairs_dir/"
    )
    families = [
        f.split(".")[0] for f in os.listdir(transition_dir) if f.endswith(".txt")
    ]

    df_results_all = []
    families_overlap = set(families)
    for name, info in run_names.items():
        run_dir = info.get("run_name")
        t_wag = info.get("t_wag")
        if t_wag is None:
            per_assay_spearman_fpath = output_dir / run_dir / "spearman_results.csv"
        else:
            t_wag_str = str(t_wag).replace(".", "_")
            per_assay_spearman_fpath = (
                output_dir
                / run_dir
                / f"wag_corrected_spearman_results_t_{t_wag_str}.csv"
            )
        if not per_assay_spearman_fpath.exists():
            print(f"{name}: {per_assay_spearman_fpath} not exists")
            continue
        df = pd.read_csv(per_assay_spearman_fpath)
        df["model"] = name
        df_results_all.append(df)
        families_overlap = set(df["family"]).intersection(families_overlap)

    if not df_results_all:
        raise ValueError("No valid data files found for any of the specified models")

    df_results_all = pd.concat(df_results_all, ignore_index=True)
    df_results_all = df_results_all[
        df_results_all["family"].isin(families_overlap)
    ].reset_index(drop=True)

    # Pivot the data to have ESM and PEINT as separate columns
    df_pivot = df_results_all.pivot_table(
        index=["family", "assay_type"], columns="model", values="spearman"
    ).reset_index()

    # Get the model keys from run_names
    model_keys = list(run_names.keys())
    if len(model_keys) != 2:
        raise ValueError("Expected exactly 2 models in run_names")

    # Check if we have data for both models
    missing_models = [m for m in model_keys if m not in df_pivot.columns]
    if missing_models:
        raise ValueError(f"Missing data for models: {missing_models}")

    # Remove rows where either model has NaN values
    df_plot = df_pivot.dropna(subset=model_keys).copy()

    if df_plot.empty:
        raise ValueError("No common families with valid data for both models")

    # Filter by assay_type if specified
    if assay_type is not None:
        df_plot = df_plot[df_plot["assay_type"] == assay_type].copy()
        if df_plot.empty:
            raise ValueError(f"No data found for assay_type '{assay_type}'")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=240)

    # Determine if we're plotting multiple assay types or just one
    if assay_type is None:
        # Plot multiple assay types with different colors
        assay_types = sorted(df_plot["assay_type"].unique())
        if palette is None:
            assay_colors = sns.color_palette("Set1", n_colors=len(assay_types))
            color_map = {assay: assay_colors[i] for i, assay in enumerate(assay_types)}
        elif isinstance(palette, list):
            if len(palette) < len(assay_types):
                raise ValueError(
                    f"Palette list must have at least {len(assay_types)} colors"
                )
            color_map = {assay: palette[i] for i, assay in enumerate(assay_types)}
        else:
            color_map = {assay: palette.get(assay, None) for assay in assay_types}

        # Plot scatter points for each assay type
        for assay_type_plot in assay_types:
            subset = df_plot[df_plot["assay_type"] == assay_type_plot]
            ax.scatter(
                subset[model_keys[0]],  # ESM on x-axis
                subset[model_keys[1]],  # PEINT on y-axis
                label=assay_type_plot,
                color=color_map[assay_type_plot],
                alpha=alpha,
                s=dot_size,
                edgecolor="black",
                linewidths=0.3,
            )
    else:
        # Plot single assay type with uniform color
        ax.scatter(
            df_plot[model_keys[0]],  # ESM on x-axis
            df_plot[model_keys[1]],  # PEINT on y-axis
            color="steelblue",
            alpha=alpha,
            s=dot_size,
            edgecolor="black",
            linewidths=0.3,
        )

    # Add diagonal line (y = x) for reference
    min_val = min(df_plot[model_keys[0]].min(), df_plot[model_keys[1]].min())
    max_val = max(df_plot[model_keys[0]].max(), df_plot[model_keys[1]].max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, linewidth=1)

    # Count families where PEINT is better than ESM
    peint_better = (df_plot[model_keys[1]] > df_plot[model_keys[0]]).sum()
    total_families = len(df_plot)
    peint_better_pct = (peint_better / total_families) * 100

    # Set labels and title
    ax.set_xlabel(
        f"{model_names.get(model_keys[0], model_keys[0])} Spearman Correlation"
    )
    ax.set_ylabel(
        f"{model_names.get(model_keys[1], model_keys[1])} Spearman Correlation"
    )
    title = f"PEINT > ESM2: {peint_better} / {total_families} ({peint_better_pct:.1f}%) families"
    if assay_type is not None:
        if assay_type == "OrganismalFitness":
            title += " (Organismal Fitness)"
        else:
            title += f" ({assay_type})"
    ax.set_title(title)

    # Set equal aspect ratio and limits
    ax.set_aspect("equal", adjustable="box")

    # Set axis limits based on data range
    if assay_type is None:
        # For multiple assay types, use standard 0-1 range
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # For single assay type, use max + 0.1 as upper limit
        max_spearman = max(df_plot[model_keys[0]].max(), df_plot[model_keys[1]].max())
        upper_limit = max_spearman + 0.1
        ax.set_xlim(0, upper_limit)
        ax.set_ylim(0, upper_limit)

    # Add grid
    ax.grid(True, alpha=0.3, which="both", linewidth=0.25)

    # Add legend only if plotting multiple assay types
    if assay_type is None:
        leg = ax.legend(
            title="Assay Type",
            bbox_to_anchor=(0.98, 0.02),
            loc="lower right",
            frameon=True,
        )
        for legline in leg.legend_handles:
            try:
                legline.set_alpha(0.9)
            except Exception:
                pass

    # Standardize axis aesthetics
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)

    plt.tight_layout()

    # Save if path provided
    if save_path is not None:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path} and {pdf_path}")

    plt.show()

    return df_plot


def _make_per_family_spearman_plot(
    run_names: dict[str, dict],
    model_names: dict[str, str] | None = None,
    output_dir: Path | None = None,
    assay_type: str = "OrganismalFitness",
    sort_by_model: str = "PEINT",
    figsize: tuple[int, int] = (12, 3),
    dpi: int = 300,
    palette: list | dict | None = None,
    save_path: str | None = None,
    dot_size: int = 50,
    best_times_json: str | None = None,
    best_times_display_name: str | None = None,
    display_full_family_name: bool = False,
):
    """Create a per-family Spearman correlation scatterplot.

    Parameters
    ----------
    run_names : dict[str, str]
        Dictionary mapping model names to their run directory names.
    model_names : dict[str, str] | None
        Optional mapping from run_names keys to display names for the plot.
        If None, uses the keys from run_names directly.
    output_dir : Path | None
        Root directory containing the results. Defaults to VEP_RESULTS_DIR.
    assay_type : str
        Filter results to only include this assay type. Default is "OrganismalFitness".
    sort_by_model : str
        Model name (from model_names values) to use for sorting families.
        Default is "PEINT".
    figsize : tuple[int, int]
        Figure size for the plot. Default is (15, 8).
    dot_size : int
        Size of the dots in the plot. Default is 36.
    save_path : str | None
        If provided, save the plot to this path.
    best_times_json : str | None
        Path to JSON file with best times data from make_per_family_spearman_time_plot.
    best_times_display_name : str | None
        Display name for the best times data. Required if best_times_json is provided.
    display_full_family_name : bool
        If True, display the full family name (including the suffix) on the x-axis.
        If False, display the prefix before the first underscore. Default is False.
    """

    if output_dir is None:
        output_dir = VEP_RESULTS_DIR

    if model_names is None:
        model_names = {k: k for k in run_names.keys()}

    # Validate best_times_json parameters
    if best_times_json is not None and best_times_display_name is None:
        raise ValueError(
            "best_times_display_name is required when best_times_json is provided"
        )

    # Load data for all models
    df_results_all = []
    families_overlap = None

    for model_key, info in run_names.items():
        run_dir = info.get("run_name")
        t_wag = info.get("t_wag")
        if t_wag is None:
            per_assay_spearman_fpath = output_dir / run_dir / "spearman_results.csv"
        else:
            t_wag_str = str(t_wag).replace(".", "_")
            per_assay_spearman_fpath = (
                output_dir
                / run_dir
                / f"wag_corrected_spearman_results_t_{t_wag_str}.csv"
            )
        if not per_assay_spearman_fpath.exists():
            print(
                f"Warning: {model_key}: {per_assay_spearman_fpath} does not exist, skipping"
            )
            continue

        df = pd.read_csv(per_assay_spearman_fpath)
        df["model"] = model_key
        df_results_all.append(df)

        # Track families that exist across all models
        if families_overlap is None:
            families_overlap = set(df["family"])
        else:
            families_overlap = families_overlap.intersection(set(df["family"]))

    if not df_results_all:
        raise ValueError("No valid data files found for any of the specified models")

    # Combine all data
    df_combined = pd.concat(df_results_all, ignore_index=True)

    # Load best times data if provided
    if best_times_json is not None:
        try:
            with open(best_times_json, "r") as f:
                best_times_data = json.load(f)

            # Create DataFrame from best times data
            best_times_rows = []
            for family, data in best_times_data["best_results"].items():
                if family in families_overlap:
                    best_times_rows.append(
                        {
                            "family": family,
                            "assay_type": best_times_data["assay_type"],
                            "spearman": data["best_spearman"],
                            "model": best_times_display_name,
                        }
                    )

            if best_times_rows:
                df_best_times = pd.DataFrame(best_times_rows)
                df_combined = pd.concat([df_combined, df_best_times], ignore_index=True)
                print(
                    f"Loaded best times data for {len(best_times_rows)} families from {best_times_json}"
                )
            else:
                print(f"Warning: No matching families found in {best_times_json}")
        except Exception as e:
            print(
                f"Warning: Could not load best times data from {best_times_json}: {e}"
            )

    # Filter by assay type and families that exist in all models
    df_filtered = df_combined[
        (df_combined["assay_type"] == assay_type)
        & (df_combined["family"].isin(families_overlap))
    ].reset_index(drop=True)

    if df_filtered.empty:
        raise ValueError(
            f"No data found for assay_type '{assay_type}' across all models"
        )

    # Map model names for display
    df_filtered = df_filtered.copy()
    # Update model_names to include best_times_display_name if provided
    if best_times_display_name is not None:
        model_names[best_times_display_name] = best_times_display_name
    df_filtered["model_display"] = df_filtered["model"].map(model_names)

    # Find the sort model key from the display name
    sort_model_key = None
    for key, display_name in model_names.items():
        if display_name == sort_by_model:
            sort_model_key = key
            break

    if sort_model_key is None:
        raise ValueError(
            f"sort_by_model '{sort_by_model}' not found in model_names values"
        )

    # Get sorting order based on sort_by_model
    sort_data = df_filtered[df_filtered["model"] == sort_model_key].copy()
    family_order = sort_data.sort_values("spearman", ascending=False)["family"].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Colors
    models_to_plot = list(model_names.values())
    if palette is None:
        base_colors = sns.color_palette("Set2", n_colors=len(models_to_plot))
        color_map = {m: base_colors[i] for i, m in enumerate(models_to_plot)}
    elif isinstance(palette, list):
        color_map = {m: palette[i] for i, m in enumerate(models_to_plot)}
    else:
        color_map = {m: palette.get(m, None) for m in models_to_plot}

    # Small jitter to avoid overlap between models at same family index
    jitter_width = 0.15
    offsets = np.linspace(-jitter_width, jitter_width, num=len(models_to_plot))

    for i, model_display in enumerate(models_to_plot):
        model_data = df_filtered[df_filtered["model_display"] == model_display]
        x_positions = [family_order.index(family) for family in model_data["family"]]
        x_positions = np.array(x_positions, dtype=float) + offsets[i]

        ax.scatter(
            x_positions,
            model_data["spearman"],
            label=model_display,
            color=color_map[model_display],
            alpha=0.8,
            s=dot_size,
            edgecolor="black",
            linewidths=0.3,
        )

    # Set x-axis labels (prefix before first underscore)
    ax.set_xticks(range(len(family_order)))
    if display_full_family_name:
        short_labels = family_order
    else:
        short_labels = [
            "_".join(f.split("_")[:3]) if isinstance(f, str) else f for f in family_order
        ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Protein Family")
    ax.set_ylabel("Spearman Correlation")
    if assay_type == "OrganismalFitness":
        assay_type_label = "Organismal Fitness"
    else:
        assay_type_label = assay_type
    ax.set_title(f"Per-Family Spearman Correlation ({assay_type_label})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    leg = ax.legend(
        bbox_to_anchor=(0.8, 1), loc="upper left", frameon=True, title="Model"
    )
    for legline in leg.legend_handles:
        try:
            legline.set_alpha(0.9)
        except Exception:
            pass

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(True, alpha=0.3, which="both", linewidth=0.25)
    fig.tight_layout()

    # Save if path provided (PNG + PDF twin)
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path} and {pdf_path}")

    plt.show()

    return df_filtered


def _make_per_family_spearman_time_plot(
    run_prefix: str,
    times: list[float] | None = None,
    output_dir: Path | None = None,
    assay_type: str = "OrganismalFitness",
    sort_by_time: float = 1.0,
    figsize: tuple[int, int] = (12, 3),
    dpi: int = 240,
    palette: list | dict | None = None,
    save_path: str | None = None,
    dot_size: int = 36,
    highlight_time: float | None = None,
    save_json: bool = True,
):
    """Create a per-family Spearman correlation scatterplot across transition times.

    For each family on the x-axis, plot a separate dot for each transition time
    provided in ``times``. Families are ordered by their Spearman values at
    ``sort_by_time`` (descending).

    Parameters
    ----------
    run_prefix : str
        Prefix of the checkpoint run directory before the time suffix (e.g.
        "20250528_162531-ft_217fams-mix1.0-epoch=5-step=5000").
    times : list[float] | None
        List of transition times evaluated. If None, defaults to
        [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0].
    output_dir : Path | None
        Root directory containing the per-time sub-directories. Defaults to
        VEP_RESULTS_DIR.
    assay_type : str
        Filter results to only include this assay type. Default is
        "OrganismalFitness".
    sort_by_time : float
        Time value from ``times`` to use for sorting families.
    figsize : tuple[int, int]
        Figure size for the plot. Default is (16, 6).
    dpi : int
        Figure DPI. Default is 240.
    palette : list | dict | None
        Optional color mapping for times. If a list, it must have length equal
        to ``len(times)``; if a dict, it should map time values to colors.
    dot_size : int
        Size of the dots in the plot. Default is 36.
    save_path : str | None
        If provided, save the plot to this path (PNG) and a PDF twin.
    highlight_time : float | None
        Time value to highlight in the plot. If None, no time will be highlighted.
    save_json : bool
        If True, save a JSON file with the best time and Spearman for each family.
    """
    if times is None:
        times = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    times = sorted(times)

    if output_dir is None:
        output_dir = VEP_RESULTS_DIR

    # Load per-time per-family results
    df_time_all = []
    families_overlap: set | None = None
    for t in times:
        t_suffix = _format_time_dir_suffix(t)
        run_dir = Path(output_dir) / f"{run_prefix}-{t_suffix}"
        results_path = run_dir / "spearman_results.csv"
        if not results_path.exists():
            print(
                f"Warning: Missing results for t={t} at {results_path}, skipping this time."
            )
            continue
        df_t = pd.read_csv(results_path)
        # Expect columns include: family, assay_type, spearman, ...
        if (
            "family" not in df_t.columns
            or "spearman" not in df_t.columns
            or "assay_type" not in df_t.columns
        ):
            print(
                f"Warning: Unexpected format in {results_path}; required columns missing, skipping this time."
            )
            continue
        df_t = df_t[df_t["assay_type"] == assay_type].copy()
        if df_t.empty:
            print(
                f"Warning: No rows for assay_type '{assay_type}' at t={t}, skipping this time."
            )
            continue
        df_t["time"] = float(t)
        df_time_all.append(df_t)

        # Track family overlap across times
        fams_t = set(df_t["family"])
        families_overlap = (
            fams_t
            if families_overlap is None
            else families_overlap.intersection(fams_t)
        )

    if not df_time_all:
        raise ValueError(
            "No valid per-time data files found for any of the specified times"
        )

    # Combine and filter to families present across all available times
    df_combined = pd.concat(df_time_all, ignore_index=True)
    if families_overlap is None or len(families_overlap) == 0:
        raise ValueError(
            "No common families across the available times after filtering by assay_type"
        )

    df_filtered = df_combined[df_combined["family"].isin(families_overlap)].reset_index(
        drop=True
    )

    # Establish family order based on spearman at sort_by_time
    if sort_by_time not in set(df_filtered["time"].unique()):
        # Fall back to the max available time if desired time wasn't loaded
        available_times = sorted(df_filtered["time"].unique().tolist())
        fallback_time = available_times[-1]
        print(
            f"Info: sort_by_time={sort_by_time} not available; using {fallback_time} for sorting."
        )
        sort_time = fallback_time
    else:
        sort_time = sort_by_time

    sort_data = df_filtered[df_filtered["time"] == sort_time].copy()
    family_order = sort_data.sort_values("spearman", ascending=False)["family"].tolist()

    # Figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Build color map for times
    times_available = sorted(df_filtered["time"].unique().tolist())
    if palette is None:
        base_colors = sns.color_palette("viridis", n_colors=len(times_available))
        time_to_color = {t: base_colors[i] for i, t in enumerate(times_available)}
    elif isinstance(palette, list):
        if len(palette) < len(times_available):
            raise ValueError("Palette list must have >= number of times available")
        time_to_color = {t: palette[i] for i, t in enumerate(times_available)}
    else:
        # dict mapping time -> color (fallback: None allowed but will error in scatter)
        time_to_color = {t: palette.get(t, None) for t in times_available}

    # Jitter to separate times visually at the same family index
    jitter_width = 0.15
    offsets = np.linspace(-jitter_width, jitter_width, num=len(times_available))

    for i, t in enumerate(times_available):
        df_t = df_filtered[df_filtered["time"] == t]
        x_positions = [family_order.index(fam) for fam in df_t["family"]]
        x_positions = np.array(x_positions, dtype=float) + offsets[i]

        ax.scatter(
            x_positions,
            df_t["spearman"],
            label=f"t={t}",
            color=time_to_color[t],
            alpha=0.8,
            s=dot_size,
            edgecolor="black",
            linewidths=0.3,
        )
    if highlight_time is not None:
        df_highlight = df_filtered[df_filtered["time"] == highlight_time]
        x_positions = [family_order.index(fam) for fam in df_highlight["family"]]
        x_positions = np.array(x_positions, dtype=float) + offsets[i]
        ax.scatter(
            x_positions,
            df_highlight["spearman"],
            label=f"t={highlight_time}",
            color="hotpink",
            marker="x",
            alpha=1.0,
            s=dot_size,
            edgecolor="black",
            linewidths=0.6,
        )

    # X-axis: family labels (prefix before first underscore)
    ax.set_xticks(range(len(family_order)))
    short_labels = [
        "_".join(f.split("_")[:3]) if isinstance(f, str) else f for f in family_order
    ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Protein Family")
    ax.set_ylabel("Spearman Correlation")
    assay_type_label = (
        "Organismal Fitness" if assay_type == "OrganismalFitness" else assay_type
    )
    ax.set_title(f"Per-Family Spearman vs Time ({assay_type_label})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    leg = ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, title="Time"
    )
    for legline in leg.legend_handles:
        try:
            legline.set_alpha(0.9)
        except Exception:
            pass

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(True, alpha=0.3, which="both", linewidth=0.25)
    fig.tight_layout()

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path} and {pdf_path}")

    # Save JSON with best time and Spearman for each family
    if save_json:
        best_results = {}
        for family in family_order:
            family_data = df_filtered[df_filtered["family"] == family]
            if not family_data.empty:
                best_idx = family_data["spearman"].idxmax()
                best_row = family_data.loc[best_idx]
                best_results[family] = {
                    "best_time": float(best_row["time"]),
                    "best_spearman": float(best_row["spearman"]),
                }

        json_data = {
            "run_prefix": run_prefix,
            "assay_type": assay_type,
            "best_results": best_results,
        }

        if save_path:
            json_path = save_path.with_suffix(".json")
        else:
            json_path = Path(f"{run_prefix}_best_times.json")

        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"Best times JSON saved to: {json_path}")

    plt.show()

    return df_filtered


def _make_per_family_spearman_time_plot_best_vs_default(
    run_prefix: str,
    default_t: float = 1.0,
    output_dir: Path | None = None,
    assay_type: str = "OrganismalFitness",
    sort_by_time: float = 1.0,
    figsize: tuple[int, int] = (12, 3),
    dpi: int = 240,
    palette: list | dict | None = None,
    save_path: str | None = None,
    dot_size: int = 36,
):
    """Create a per-family Spearman correlation scatterplot comparing default time vs best time.

    For each family on the x-axis, plot two dots:
    1. Default time (t=default_t, fixed for all families)
    2. Best time (the time with highest Spearman correlation for that family)

    Parameters
    ----------
    run_prefix : str
        Prefix of the checkpoint run directory before the time suffix (e.g.
        "20250528_162531-ft_217fams-mix1.0-epoch=5-step=5000").
    default_t : float
        Default time to plot for all families. Default is 1.0.
    output_dir : Path | None
        Root directory containing the per-time sub-directories. Defaults to
        VEP_RESULTS_DIR.
    assay_type : str
        Filter results to only include this assay type. Default is
        "OrganismalFitness".
    sort_by_time : float
        Time value to use for sorting families.
    figsize : tuple[int, int]
        Figure size for the plot. Default is (12, 3).
    dpi : int
        Figure DPI. Default is 240.
    palette : list | dict | None
        Optional color mapping. If a list, should have 2 colors [default_color, best_color].
        If a dict, should map "default" and "best" to colors.
    dot_size : int
        Size of the dots in the plot. Default is 36.
    save_path : str | None
        If provided, save the plot to this path (PNG) and a PDF twin.
    """
    if output_dir is None:
        output_dir = VEP_RESULTS_DIR

    # Auto-discover all available time directories
    time_dirs = _discover_time_dirs(output_dir, run_prefix, times=None)
    if not time_dirs:
        raise ValueError(
            f"No time directories found for run_prefix={run_prefix} under {output_dir}"
        )

    times = [t for t, _ in time_dirs]
    print(f"Discovered {len(times)} time directories: {times}")

    # Load per-time per-family results
    df_time_all = []
    families_overlap: set | None = None
    for t, time_dir in time_dirs:
        results_path = time_dir / "spearman_results.csv"
        if not results_path.exists():
            print(
                f"Warning: Missing results for t={t} at {results_path}, skipping this time."
            )
            continue
        df_t = pd.read_csv(results_path)
        # Expect columns include: family, assay_type, spearman, ...
        if (
            "family" not in df_t.columns
            or "spearman" not in df_t.columns
            or "assay_type" not in df_t.columns
        ):
            print(
                f"Warning: Unexpected format in {results_path}; required columns missing, skipping this time."
            )
            continue
        df_t = df_t[df_t["assay_type"] == assay_type].copy()
        if df_t.empty:
            print(
                f"Warning: No rows for assay_type '{assay_type}' at t={t}, skipping this time."
            )
            continue
        df_t["time"] = float(t)
        df_time_all.append(df_t)

        # Track family overlap across times
        fams_t = set(df_t["family"])
        families_overlap = (
            fams_t
            if families_overlap is None
            else families_overlap.intersection(fams_t)
        )

    if not df_time_all:
        raise ValueError(
            "No valid per-time data files found for any of the discovered times"
        )

    # Combine and filter to families present across all available times
    df_combined = pd.concat(df_time_all, ignore_index=True)
    if families_overlap is None or len(families_overlap) == 0:
        raise ValueError(
            "No common families across the available times after filtering by assay_type"
        )

    df_filtered = df_combined[df_combined["family"].isin(families_overlap)].reset_index(
        drop=True
    )

    # Check if default_t exists in the data
    if default_t not in set(df_filtered["time"].unique()):
        available_times = sorted(df_filtered["time"].unique().tolist())
        raise ValueError(
            f"default_t={default_t} not found in available times: {available_times}"
        )

    # For each family, find the best t (highest Spearman)
    best_t_per_family = {}
    for family in families_overlap:
        family_data = df_filtered[df_filtered["family"] == family]
        if not family_data.empty:
            best_idx = family_data["spearman"].idxmax()
            best_row = family_data.loc[best_idx]
            best_t_per_family[family] = float(best_row["time"])

    # Filter data to only include default_t and best_t for each family
    # Add a marker column to distinguish "default" vs "best" rows
    df_plot_data = []
    for family in families_overlap:
        # Add default_t data
        default_data = df_filtered[
            (df_filtered["family"] == family) & (df_filtered["time"] == default_t)
        ].copy()
        if not default_data.empty:
            default_data["point_type"] = "default"
            df_plot_data.append(default_data)

        # Add best_t data (always, even if same as default_t - they will overlap visually)
        best_t = best_t_per_family[family]
        best_data = df_filtered[
            (df_filtered["family"] == family) & (df_filtered["time"] == best_t)
        ].copy()
        if not best_data.empty:
            best_data["point_type"] = "best"
            df_plot_data.append(best_data)

    if not df_plot_data:
        raise ValueError("No data to plot after filtering to default_t and best_t")

    df_plot = pd.concat(df_plot_data, ignore_index=True)

    # Establish family order based on spearman at sort_by_time
    if sort_by_time not in set(df_filtered["time"].unique()):
        # Fall back to the max available time if desired time wasn't loaded
        available_times = sorted(df_filtered["time"].unique().tolist())
        fallback_time = available_times[-1]
        print(
            f"Info: sort_by_time={sort_by_time} not available; using {fallback_time} for sorting."
        )
        sort_time = fallback_time
    else:
        sort_time = sort_by_time

    sort_data = df_filtered[df_filtered["time"] == sort_time].copy()
    family_order = sort_data.sort_values("spearman", ascending=False)["family"].tolist()

    # Figure and axes
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # Build color map for default_t and best_t
    if palette is None:
        # Default colors: blue for default, orange for best
        default_color = sns.color_palette("Set1")[0]  # Blue
        best_color = sns.color_palette("Set1")[1]  # Orange
    elif isinstance(palette, list):
        if len(palette) < 2:
            raise ValueError("Palette list must have at least 2 colors")
        default_color = palette[0]
        best_color = palette[1]
    else:
        # dict mapping "default" and "best" to colors
        default_color = palette.get("default", sns.color_palette("Set1")[0])
        best_color = palette.get("best", sns.color_palette("Set1")[1])

    # Jitter to separate default_t and best_t visually at the same family index
    jitter_width = 0.15
    offsets = np.linspace(-jitter_width, jitter_width, num=2)

    # Plot default_t data (using point_type marker)
    df_default = df_plot[df_plot["point_type"] == "default"]
    if not df_default.empty:
        x_positions = [family_order.index(fam) for fam in df_default["family"]]
        x_positions = np.array(x_positions, dtype=float) + offsets[0]

        ax.scatter(
            x_positions,
            df_default["spearman"],
            label=f"t={default_t}",
            color=default_color,
            alpha=0.8,
            s=dot_size,
            edgecolor="black",
            linewidths=0.3,
        )

    # Plot best_t data (using point_type marker)
    df_best = df_plot[df_plot["point_type"] == "best"]
    if not df_best.empty:
        x_positions = [family_order.index(fam) for fam in df_best["family"]]
        x_positions = np.array(x_positions, dtype=float) + offsets[1]

        ax.scatter(
            x_positions,
            df_best["spearman"],
            label="Best t",
            color=best_color,
            alpha=0.8,
            s=dot_size,
            edgecolor="black",
            linewidths=0.3,
        )

    # X-axis: family labels (prefix before first underscore)
    ax.set_xticks(range(len(family_order)))
    short_labels = [
        "_".join(f.split("_")[:3]) if isinstance(f, str) else f for f in family_order
    ]
    ax.set_xticklabels(short_labels, rotation=45, ha="right")
    ax.set_xlabel("Protein Family")
    ax.set_ylabel("Spearman Correlation")
    assay_type_label = (
        "Organismal Fitness" if assay_type == "OrganismalFitness" else assay_type
    )
    ax.set_title(f"Per-Family Spearman: Default t={default_t} vs Best t ({assay_type_label})")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    leg = ax.legend(
        bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, title="Time"
    )
    for legline in leg.legend_handles:
        try:
            legline.set_alpha(0.9)
        except Exception:
            pass

    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(True, alpha=0.3, which="both", linewidth=0.25)
    fig.tight_layout()

    # Save if requested
    if save_path:
        save_path = Path(save_path)
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path} and {pdf_path}")

    plt.show()

    return df_plot


# Public figure generators (thin wrappers centralizing processing per plot type)
def make_spearman_figure():
    palettes_all = sns.color_palette()
    palettes = [
        palettes_all[4],
        palettes_all[2],
        palettes_all[1],
    ]
    run_names = {
        "esm_150m": {"run_name": "ESM2_150M", "t_wag": None},
        "peint": {
            "run_name": "peint_esm2_150m",
            "t_wag": None,
        },
        # "peint_optimal_t_per_site": {
        #     "run_name": "20250922_112511-ft_217fams-hhfilter90-epoch=5-step=4000-t_optimal_per_site",
        #     "t_wag": None,
        # },
        # 650M arm, kept for reference; the captions specify 150M.
        # "esm_650m": {"run_name": "ESM2_650M", "t_wag": None},
        # "peint": {"run_name": "peint_650m", "t_wag": None},
    }
    model_names = {
        "esm_150m": "ESM2",
        "peint": "PEINT",
    }
    save_path = _fig_path("spearman_agg", "spearman_plot_esm_150m.png")
    return _make_spearman_plot(
        run_names=run_names,
        model_names=model_names,
        palette=palettes,
        save_path=save_path,
        figsize=(10, 4),
        # save_spearman_path=MAIN_DIR / "protevo/vep/figures" / "spearman_results.csv",
    )


def make_per_family_spearman_figure():
    output_root = VEP_RESULTS_DIR
    run_names_example = {
        "esm_150m": {"run_name": "ESM2_150M", "t_wag": None},
        "peint": {
            "run_name": "peint_esm2_150m",
            "t_wag": None,
        },
        # "peint_optimal_t_per_site": {
        #     "run_name": "20250922_112511-ft_217fams-hhfilter90-epoch=5-step=4000-t_optimal_per_site",
        #     "t_wag": None,
        # },
        # "esm_650m": {"run_name": "ESM2_650M", "t_wag": None},
        # "peint": {"run_name": "peint_650m", "t_wag": None},
    }
    model_names_example = {
        "esm_150m": "ESM2",
        "peint": "PEINT",
    }
    plt.rcParams.update(
        {
            "axes.labelsize": 12,
            "axes.titlesize": 12,
            "xtick.labelsize": 6,
            "ytick.labelsize": 8,
            "legend.title_fontsize": 12,
            "legend.fontsize": 12,
        }
    )
    palettes_all = sns.color_palette()
    palettes = [
        palettes_all[4],
        palettes_all[2],
        palettes_all[1],
    ]
    save_path = _fig_path("per_family", "per_family_spearman_plot_esm_150m.png")
    return _make_per_family_spearman_plot(
        run_names=run_names_example,
        model_names=model_names_example,
        output_dir=Path(output_root),
        assay_type="OrganismalFitness",
        sort_by_model="PEINT",
        save_path=save_path,
        palette=palettes,
        dot_size=50,
        figsize=(12, 5),
        display_full_family_name=True,
    )


def make_per_family_spearman_time_figure():
    output_root = VEP_RESULTS_DIR
    default_times = [
        0.05,
        0.1,
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.8,
        0.9,
        1.0,
        1.5,
        2.0,
        2.5,
        3.0,
        3.5,
        4.0,
        4.5,
        5.0,
    ]
    default_run_prefix = "20250528_162531-ft_217fams-mix1.0-epoch=5-step=5000"
    save_path = _fig_path("time", "per_family_spearman_time_plot.png")
    plt.rcParams.update({"xtick.labelsize": 6})
    return _make_per_family_spearman_time_plot(
        run_prefix=default_run_prefix,
        times=default_times,
        output_dir=Path(output_root),
        assay_type="OrganismalFitness",
        sort_by_time=1.0,
        save_path=save_path,
        dot_size=50,
        figsize=(12, 5),
    )


def make_per_family_spearman_time_figure_best_vs_default():
    output_root = VEP_RESULTS_DIR
    default_run_prefix = "20250922_112511-ft_217fams-hhfilter90-epoch=5-step=4000"
    save_path = _fig_path("time", "per_family_spearman_time_plot_best_vs_default.png")
    plt.rcParams.update({"xtick.labelsize": 6})
    return _make_per_family_spearman_time_plot_best_vs_default(
        run_prefix=default_run_prefix,
        default_t=1.0,
        output_dir=Path(output_root),
        assay_type="OrganismalFitness",
        sort_by_time=1.0,
        save_path=save_path,
        dot_size=50,
        figsize=(12, 5),
    )


def make_esm_vs_peint_figure():
    run_names_example = {
        "esm_150m": {"run_name": "ESM2_150M", "t_wag": None},
        "peint": {"run_name": "peint_esm2_150m", "t_wag": None},
    }
    model_names_example = {
        "esm_150m": "ESM2",
        "peint": "PEINT",
    }
    assay_type = "OrganismalFitness"
    save_path = _fig_path(
        "scatter", f"esm_vs_peint_scatter_{assay_type}_esm_150m.png"
    )
    return _make_esm_vs_peint_plot(
        run_names=run_names_example,
        model_names=model_names_example,
        save_path=save_path,
        assay_type=assay_type,
        figsize=(6, 5),
    )


def make_spearman_by_mutational_depth_figure(per_assay_type=False):
    """Generate Spearman correlation by mutational depth figure.

    Parameters
    ----------
    per_assay_type : bool
        If True, generate separate plots for each assay type.
        If False (default), generate a single plot with all assay types combined.
    """
    palettes_all = sns.color_palette()
    palettes = [
        palettes_all[4],
        palettes_all[2],
        palettes_all[1],
    ]
    run_names = {
        "esm_150m": {"run_name": "ESM2_150M", "t_wag": None},
        "peint": {
            "run_name": "peint_esm2_150m",
            "t_wag": None,
        },
        # 650M arm, kept for reference; the captions specify 150M.
        # "esm_650m": {"run_name": "ESM2_650M", "t_wag": None},
        # "peint": {"run_name": "peint_650m", "t_wag": None},
    }
    model_names = {
        "esm_150m": "ESM2",
        "peint": "PEINT",
    }
    save_path = _fig_path("mutational_depth", "spearman_by_mutational_depth_esm_150m.png")
    output_dir = VEP_RESULTS_DIR

    # Load mutational depth data for all models
    df_results_all = []
    for name, info in run_names.items():
        run_dir = info.get("run_name")
        t_wag = info.get("t_wag")

        depth_fpath = output_dir / run_dir / "spearman_by_mutation_depth.csv"
        if not depth_fpath.exists():
            print(f"{name}: {depth_fpath} not found, skipping")
            continue

        df = pd.read_csv(depth_fpath)
        df["model"] = name
        df_results_all.append(df)

    if not df_results_all:
        raise ValueError("No valid mutation depth data files found")

    df_results_all = pd.concat(df_results_all, ignore_index=True)

    if per_assay_type:
        # Generate separate plots for each assay type
        assay_types = sorted(df_results_all["assay_type"].unique())
        axes = []

        for assay_type in assay_types:
            print(f"\nGenerating plot for assay type: {assay_type}")
            df_assay = df_results_all[df_results_all["assay_type"] == assay_type].copy()

            # Skip if not enough data
            if df_assay.empty:
                print(f"  No data for {assay_type}, skipping")
                continue

            # Call plotting function
            ax = _plot_spearman_by_mutational_depth(
                df_results_all=df_assay,
                model_names=model_names,
                palette=palettes,
                figsize=(8, 5),
                alpha=0.8,
            )

            # Update title to include assay type
            if assay_type == "OrganismalFitness":
                assay_type_label = "Organismal Fitness"
            else:
                assay_type_label = assay_type
            ax.set_title(
                f"Mean Spearman Correlation by Mutational Depth ({assay_type_label})"
            )

            # Standardize axis aesthetics
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            ax.tick_params(width=0.5, length=2)
            ax.grid(True, alpha=0.3, which="both", linewidth=0.25)

            # Save with assay type in filename
            assay_type_safe = assay_type.replace(" ", "_").replace("/", "_")
            save_path = _fig_path(
                "mutational_depth",
                f"spearman_by_mutational_depth_{assay_type_safe}.png",
            )
            fig = ax.get_figure()
            fig.tight_layout()
            # Save PNG
            fig.savefig(save_path, dpi=400, bbox_inches="tight")
            # Save PDF
            pdf_path = save_path.with_suffix(".pdf")
            fig.savefig(pdf_path, bbox_inches="tight")
            print(f"  Plot saved to: {save_path} and {pdf_path}")

            axes.append(ax)

        plt.show()
        return axes
    else:
        # Generate single plot with all assay types combined
        ax = _plot_spearman_by_mutational_depth(
            df_results_all=df_results_all,
            model_names=model_names,
            palette=palettes,
            figsize=(8, 5),
            alpha=0.8,
        )

        # Standardize axis aesthetics
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.tick_params(width=0.5, length=2)
        ax.grid(True, alpha=0.3, which="both", linewidth=0.25)
        fig = ax.get_figure()
        fig.tight_layout()
        # Save PNG
        fig.savefig(save_path, dpi=400, bbox_inches="tight")
        # Save PDF
        pdf_path = save_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight")
        print(f"Plot saved to: {save_path} and {pdf_path}")

        plt.show()

        return ax


# ---------------------------------------------------------------------------
# Official-release baselines + base-vs-PEINT / multi-model comparisons
# ---------------------------------------------------------------------------
# Baselines come from ProteinGym's released per-variant zero-shot scores via
# protevo.vep.official_baselines (the maintained canonical source). Each released
# column is materialized as an ordinary `test_lls/production/<column>/` run dir so
# the existing bar machinery (_make_spearman_plot -> side-by-side bars per assay
# type with SE error bars) works unchanged and any released model can sit next to
# PEINT checkpoints scored with compute_fitness.


def materialize_official_baselines(columns, overwrite=False):
    """Write `test_lls/production/<col>/spearman_results.csv` for each released
    column (schema `family, assay_type, spearman`), computing all requested columns
    in a single pass over the release. Returns the list of run names (== columns).
    """
    from protevo.vep.official_baselines import released_zero_shot_spearman

    prod = VEP_RESULTS_DIR
    todo = [
        c
        for c in columns
        if overwrite or not (prod / c / "spearman_results.csv").exists()
    ]
    if todo:
        df = released_zero_shot_spearman(todo).rename(columns={"DMS_id": "family"})
        for col in todo:
            sub = df[df["model"] == col][["family", "assay_type", "spearman"]]
            (prod / col).mkdir(parents=True, exist_ok=True)
            sub.to_csv(prod / col / "spearman_results.csv", index=False)
    return list(columns)


def _spearman_bar_figure(run_names, model_names, save_name, palette=None, figsize=(11, 5),
                         group_size=None):
    """Shared driver: materialize any official baselines referenced, then draw the
    side-by-side-by-assay-type bar chart (SE error bars, class-avg in the legend).

    `run_names[key]["run_name"]` is a `test_lls/production` sub-dir; set
    `run_names[key]["official"] = True` for released-baseline columns so they are
    materialized on demand.
    """
    official_cols = [
        info["run_name"] for info in run_names.values() if info.get("official")
    ]
    if official_cols:
        materialize_official_baselines(official_cols)
    run_names = {k: {"run_name": v["run_name"], "t_wag": v.get("t_wag")} for k, v in run_names.items()}
    save_path = _fig_path("spearman_agg", save_name)
    return _make_spearman_plot(
        run_names=run_names,
        model_names=model_names,
        palette=palette,
        save_path=save_path,
        figsize=figsize,
        save_spearman_path=_fig_path("spearman_agg", save_name.replace(".png", ".csv")),
        group_size=group_size,
    )


def make_base_vs_peint_spearman_figure(
    base_column, peint_run, base_label="ESM (base)", peint_label="PEINT", save_name=None, palette=None
):
    """Default 1v1: a base backbone (official release) vs PEINT trained on it.

    Side-by-side bars per assay type with SE error bars.
    """
    palettes_all = sns.color_palette()
    palette = palette or [palettes_all[7], palettes_all[0]]  # base = grey, PEINT = blue
    run_names = {
        "base": {"run_name": base_column, "official": True},
        "peint": {"run_name": peint_run},
    }
    model_names = {"base": base_label, "peint": peint_label}
    return _spearman_bar_figure(
        run_names, model_names, save_name or f"base_vs_peint_{base_column}.png",
        palette=palette, figsize=(10, 4),
    )


def make_multimodel_spearman_figure(methods, save_name="multimodel_spearman.png", palette=None,
                                    group_size=None):
    """N-model bar chart: side-by-side bars per assay type (SE error bars).

    `methods` is an ordered list of dicts, each either
      {"label": ..., "official": "<COLUMN>"}   released zero-shot baseline, or
      {"label": ..., "run": "<run_dir>"}       a compute_fitness result dir.
    """
    run_names, model_names = {}, {}
    for i, m in enumerate(methods):
        key = f"m{i}"
        if "official" in m:
            run_names[key] = {"run_name": m["official"], "official": True}
        else:
            run_names[key] = {"run_name": m["run"]}
        model_names[key] = m["label"]
    return _spearman_bar_figure(run_names, model_names, save_name, palette=palette,
                                figsize=(12, 5), group_size=group_size)


# Our comparison: raw ESM-C / ESM2-150M (released) vs PEINT trained on each.
# PEINT run dirs are compute_fitness outputs copied under test_lls/production/.
_ESMC_ESM2_CONFIG = {
    "esmc": {"base_column": "ESMC-300M", "peint_run": "peint_esmc300m", "label": "ESM-C 300M"},
    "esm2_150m": {"base_column": "ESM2_150M", "peint_run": "peint_esm2_150m", "label": "ESM2 150M"},
}


def make_esmc_esm2_comparison():
    """Produce the 1v1 base-vs-PEINT figures for each backbone plus the combined
    4-model bar chart (raw ESM-C, PEINT/ESM-C, raw ESM2-150M, PEINT/ESM2-150M)."""
    palettes_all = sns.color_palette()
    for tier, cfg in _ESMC_ESM2_CONFIG.items():
        make_base_vs_peint_spearman_figure(
            base_column=cfg["base_column"],
            peint_run=cfg["peint_run"],
            base_label=cfg["label"],
            peint_label=f"PEINT ({cfg['label']})",
            save_name=f"base_vs_peint_{tier}.png",
        )
    # One hue family per backbone (raw = light, PEINT = dark), from the shared map.
    pairs = base_lm_pair_colors()
    palette = [*pairs["ESM-C 300M"], *pairs["ESM2-150M"]]
    methods = [
        {"label": "ESM-C 300M", "official": "ESMC-300M"},
        {"label": "PEINT (ESM-C 300M)", "run": "peint_esmc300m"},
        {"label": "ESM2 150M", "official": "ESM2_150M"},
        {"label": "PEINT (ESM2 150M)", "run": "peint_esm2_150m"},
    ]
    return make_multimodel_spearman_figure(
        methods, save_name="multimodel_esmc_esm2.png", palette=palette
    )


# ---------------------------------------------------------------------------
# Multi-model VEP comparison across base LMs (adds ESM2-650M)
# ---------------------------------------------------------------------------
# A "new version" of the multi-model figures that also carries the 650M ESM2
# tier and, per Antoine, is organized by frozen base LM: each backbone (ESM2-150M,
# ESM-C 300M, ESM2-650M) contributes its released zero-shot baseline and the PEINT
# model trained on it, in the paired order ESM2-150 | PEINT ESM2-150 | ESM-C |
# PEINT ESM-C | ESM2-650 | PEINT ESM2-650. Config + data derivation live in
# paper.vep; these functions are thin plotting + orchestration.


def make_multimodel_by_base_lm(save_name="multimodel_by_base_lm.png"):
    """Two bar charts: (1) grouped by base LM (base pLM vs PEINT, bar = class-
    averaged Spearman over assay types, SE across assay types); (2) the assay-type-
    resolved 6-model chart (paired hues). Both restrict to the family set common to
    all six runs.
    """
    materialize_official_baselines([b for _, b, _ in vep.BASE_LM_CONFIG])
    run_names, _ = vep.base_lm_run_and_model_names()
    df = vep.load_spearman_overlap(VEP_RESULTS_DIR, run_names)
    n_fam = df["family"].nunique()

    # class-average building block: per (model key, assay type) mean Spearman.
    per_assay = df.groupby(["key", "assay_type"], as_index=False)["spearman"].mean()
    per_assay["base_lm"] = per_assay["key"].str.split("|").str[0]
    per_assay["kind"] = per_assay["key"].str.endswith("|peint").map(
        {True: "PEINT", False: "Base pLM"}
    )
    lm_order = [lm for lm, _, _ in vep.BASE_LM_CONFIG]

    # --- Figure 1: grouped by base LM (base vs PEINT), class-avg over assay types ---
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(
        data=per_assay, x="base_lm", y="spearman", hue="kind",
        order=lm_order, hue_order=["Base pLM", "PEINT"],
        estimator=np.mean, errorbar="se", capsize=0.15,
        err_kws={"linewidth": 1.0}, linewidth=0.5, edgecolor="black",
        palette=[_BASE_GRAY, model_colors()["PEINT (ESM2)"]], alpha=0.85, ax=ax,
    )
    class_avg = per_assay.groupby(["base_lm", "kind"])["spearman"].mean()
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xlabel("Base language model")
    ax.set_ylabel("Class-averaged Spearman")
    ax.set_title(f"VEP: base pLM vs PEINT by backbone ({n_fam} families)")
    ax.legend(title="", frameon=True, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    fig.tight_layout()
    save_path = _fig_path("spearman_agg", save_name)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    per_assay.to_csv(save_path.with_suffix(".csv"), index=False)
    print(f"Plot saved to: {save_path} and {save_path.with_suffix('.pdf')}")
    print("\nClass-averaged Spearman (base_lm, kind):")
    print(class_avg.round(3).to_string())

    # --- Figure 2: assay-type-resolved 6-model bar chart (paired hues) ---
    methods = []
    for lm, base_col, peint_run in vep.BASE_LM_CONFIG:
        methods.append({"label": lm, "official": base_col})
        methods.append({"label": f"PEINT ({lm})", "run": peint_run})
    # group_size=2: BASE_LM_CONFIG emits (base pLM, PEINT) consecutively, so pairing
    # adjacent hues puts each backbone and the PEINT model built on it side by side.
    make_multimodel_spearman_figure(
        methods, save_name="multimodel_esm2_150_650_esmc.png",
        palette=vep.base_lm_palette(), group_size=2,
    )
    return ax


def make_mutational_depth_by_base_lm(save_name="mutational_depth_by_base_lm.png"):
    """Spearman-by-mutational-depth pointplot for all six base-LM entries (base pLM
    + PEINT per backbone). Base-pLM depth curves are materialized from the release;
    everything is restricted to the family set common to all six runs.
    """
    run_names, model_names = vep.base_lm_run_and_model_names()
    peint_run = next(rd for k, rd in run_names.items() if k.endswith("|peint"))
    peint_fams = set(pd.read_csv(
        VEP_RESULTS_DIR / peint_run / "spearman_by_mutation_depth.csv"
    )["family"])
    vep.materialize_official_depth(
        VEP_RESULTS_DIR,
        [rd for k, rd in run_names.items() if k.endswith("|base")],
        families=peint_fams,
    )

    dfs, overlap = [], None
    for key, rd in run_names.items():
        df = pd.read_csv(VEP_RESULTS_DIR / rd / "spearman_by_mutation_depth.csv")
        df["model"] = key
        dfs.append(df)
        fams = set(df["family"])
        overlap = fams if overlap is None else (overlap & fams)
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["family"].isin(overlap)].reset_index(drop=True)

    ax = _plot_spearman_by_mutational_depth(
        df_results_all=df, model_names=model_names, palette=vep.base_lm_palette(),
        figsize=(8, 5), alpha=0.85,
    )
    ax.set_title(f"Mean Spearman by Mutational Depth ({df['family'].nunique()} families)")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2)
    ax.grid(True, alpha=0.3, which="both", linewidth=0.25)
    fig = ax.get_figure()
    fig.tight_layout()
    save_path = _fig_path("mutational_depth", save_name)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    print(f"Plot saved to: {save_path} and {save_path.with_suffix('.pdf')}")
    return ax


# ---------------------------------------------------------------------------
# Overall Spearman vs total model size (params scatter)
# ---------------------------------------------------------------------------
# Class-averaged Spearman on the common core set of assays vs TOTAL parameter count
# (log x). Base single-sequence pLMs in gray; PEINT models colored by backbone family,
# with a connector from each base pLM to the PEINT model built on it — the connector's
# horizontal run is the parameter cost of PEINT's head, its rise is the accuracy lift.
#
# x is the whole model, not just the frozen backbone. For PEINT that is
# backbone + (2 encoder + 2 decoder layers + lm_head + time embedding); the head adds
# only 8-16%, which is the point of the panel. The three ESM2/ESM-C counts below were
# measured off the checkpoints rather than taken from the published round numbers, so a
# base pLM and the PEINT model built on it are counted the same way:
#
#     peint_150m      ckpt vep/peint_150m/epoch=5-step=4000.ckpt
#                     148,161,274 frozen +  23,432,353 head = 171,593,627   (+15.8%)
#     peint_650m      ckpt vep/peint_650m/epoch=12-step=10000.ckpt
#                     651,085,494 frozen +  93,559,073 head = 744,644,567   (+14.4%)
#     peint_esmc300m  ckpt esmc-biohub/1e1d-ep_13-step_4130.ckpt (1 enc + 1 dec)
#                     332,997,184 frozen +  26,882,944 head = 359,880,128    (+8.1%)
#
# (summed over state_dict, split on the "model.esm." prefix; do NOT infer these from
# checkpoint file size — the encoder-stripped checkpoints of the vESM runs, since dropped,
# carried optimizer state and implied a head about 3x too large.) ProGen2 / Tranception keep
# their published counts.
#
# Data sources: ESM2 / ESM-C from the ProteinGym release (official_baselines); PEINT via
# compute_fitness.
_PARAMS_MODELS = [
    # (label, run_dir_in_local_data, total_params, is_peint, family, pair_key)
    ("ESM2-150M", "ESM2_150M", 148_161_274, False, "esm2", "esm2-150"),
    ("ESM2-650M", "ESM2_650M", 651_085_494, False, "esm2-650", "esm2-650"),
    ("ESM-C 300M", "ESMC-300M", 332_997_184, False, "esmc", "esmc-300"),
    ("ProGen2-small", "Progen2_small", 151e6, False, "progen", None),
    ("ProGen2-medium", "Progen2_medium", 764e6, False, "progen", None),
    ("ProGen2-large", "Progen2_large", 2.7e9, False, "progen", None),
    ("ProGen2-xlarge", "Progen2_xlarge", 6.4e9, False, "progen", None),
    ("Tranception-S", "Tranception_S_no_retrieval", 85e6, False, "tranception", None),
    ("Tranception-M", "Tranception_M_no_retrieval", 300e6, False, "tranception", None),
    ("Tranception-L", "Tranception_L_no_retrieval", 700e6, False, "tranception", None),
    ("PEINT (ESM2-150M)", "peint_150m", 171_593_627, True, "esm2", "esm2-150"),
    ("PEINT (ESM2-650M)", "peint_650m", 744_644_567, True, "esm2-650", "esm2-650"),
    ("PEINT (ESM-C 300M)", "peint_esmc300m", 359_880_128, True, "esmc", "esmc-300"),
]
_BASE_GRAY = "#9a9a9a"

# Label placement overrides, (dx, dy) in points, plus optional horizontal alignment.
# The ESM-C pair and ESM2-650M sit close together in both x and y, so their default
# right-of-marker labels overlap; these push them apart.
_PARAMS_LABEL_OFFSET = {
    "PEINT (ESM-C 300M)": (-8, 7, "right"),
    "ESM-C 300M": (-8, -12, "right"),
    "ESM2-650M": (7, -13, "left"),
    "PEINT (ESM2-650M)": (8, 2, "left"),
}


def _family_color():
    """Backbone family -> PEINT marker color, from the shared model_style map."""
    c = model_colors()
    return {
        "esm2": c["PEINT (ESM2)"],
        "esm2-650": c["PEINT (ESM2-650M)"],
        "esmc": c["PEINT (ESM-C)"],
    }


def _core_overall_spearman(models, aggregate="class", assay_type=None):
    """Overall Spearman per model on the assays scored by *every* model (common core set).

    assay_type=None uses all classes: aggregate='class' -> mean-of-class-means (ProteinGym
    Average_Spearman), 'flat' -> mean over assays. Setting assay_type restricts to that one
    ProteinGym function class and returns the mean over its assays (common to all models).
    Returns (dict label->spearman, sorted common family list).
    """
    per = {}
    for label, run, *_ in models:
        f = VEP_RESULTS_DIR / run / "spearman_results.csv"
        if not f.exists():
            raise FileNotFoundError(f"{label}: missing {f} (stage it into local_data first)")
        df = pd.read_csv(f).rename(columns={"DMS_id": "family"})
        if assay_type is not None:
            df = df[df["assay_type"] == assay_type]
        per[label] = df.set_index("family")
    common = sorted(set.intersection(*[set(df.index) for df in per.values()]))

    def agg(df):
        d = df.loc[common]
        if assay_type is not None or aggregate == "flat":
            return d["spearman"].mean()
        return d.groupby("assay_type")["spearman"].mean().mean()

    return {label: agg(df) for label, df in per.items()}, common


def make_params_vs_spearman_figure(save_name=None, aggregate="class", assay_type=None):
    """Scatter of overall Spearman vs total model params (log x), gray base pLMs vs
    color-by-family PEINT, with a connector from each base pLM to the PEINT model built
    on it: the run is the head's parameter cost, the rise is the accuracy lift.

    assay_type restricts to one ProteinGym function class (e.g. 'OrganismalFitness')."""
    from matplotlib.lines import Line2D

    vals, common = _core_overall_spearman(_PARAMS_MODELS, aggregate, assay_type)
    if assay_type is not None:
        ylab = f"Spearman ρ ({assay_type})"
        title = f"VEP {assay_type} vs model size — ProteinGym DMS ({len(common)} assays)"
        save_name = save_name or f"params_vs_spearman_{assay_type}.png"
    else:
        ylab = "Class-averaged Spearman ρ" if aggregate == "class" else "Mean Spearman ρ"
        title = f"VEP vs model size — ProteinGym DMS ({len(common)} core assays)"
        save_name = save_name or "params_vs_spearman.png"
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    family_color = _family_color()

    # base -> PEINT connectors, paired by pair_key. Base and PEINT now sit at different
    # x (the head's cost), so these slope up and to the right.
    base = {k: (p, vals[l]) for l, _, p, ip, _f, k in _PARAMS_MODELS if not ip and k}
    peint = {k: (p, vals[l]) for l, _, p, ip, _f, k in _PARAMS_MODELS if ip and k}
    for k in set(base) & set(peint):
        (xb, yb), (xp, yp) = base[k], peint[k]
        ax.plot([xb, xp], [yb, yp], color="0.8", lw=1.0, zorder=1)

    for label, _, p, is_peint, fam, _k in _PARAMS_MODELS:
        y = vals[label]
        color = family_color[fam] if is_peint else _BASE_GRAY
        ax.scatter(p, y, s=190 if is_peint else 120, marker="o", color=color,
                   edgecolor="white", linewidth=0.8, zorder=3 if is_peint else 2)
        dx, dy, ha = _PARAMS_LABEL_OFFSET.get(label, (6, 5 if is_peint else -11, "left"))
        ax.annotate(label, (p, y),
                    xytext=(dx, dy), textcoords="offset points", ha=ha,
                    fontsize=7, color=color if is_peint else "0.45",
                    fontweight="bold" if is_peint else "normal")

    ax.set_xscale("log")
    ax.set_xticks([1e8, 3e8, 1e9, 3e9, 1e10])
    ax.set_xticklabels(["100M", "300M", "1B", "3B", "10B"])
    ax.set_xlim(7.0e7, 8.5e9)
    ax.set_xlabel("Total parameters (frozen backbone + task head)")
    ax.set_ylabel(ylab)
    ax.set_title(title)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BASE_GRAY, markersize=8, label="base pLM (zero-shot)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=family_color["esm2"], markersize=8, label="PEINT / ESM2-150M"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=family_color["esmc"], markersize=8, label="PEINT / ESM-C"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=family_color["esm2-650"], markersize=8, label="PEINT / ESM2-650M"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, lw=0.5)
    fig.tight_layout()

    save_path = _fig_path("params", save_name)
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    pd.DataFrame(
        [(l, int(p), "PEINT" if ip else "base", fam, vals[l])
         for l, _, p, ip, fam, _k in _PARAMS_MODELS],
        columns=["model", "total_params", "type", "family", "spearman"],
    ).to_csv(save_path.with_suffix(".csv"), index=False)
    print(f"wrote {save_path} | {len(common)} core assays")
    for l, _, _p, _ip, _f, _k in _PARAMS_MODELS:
        print(f"  {l:22s} {vals[l]:.4f}")
    return ax


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate various types of plots for VEP analysis"
    )

    # Main plot type selection
    parser.add_argument(
        "--plot",
        type=str,
        choices=[
            "agg",
            "family",
            "time_family",
            "esm_vs_peint",
            "mutant_depth",
            "time_family_best_vs_default",
            "esmc_esm2",
            "by_base_lm",
            "params",
        ],
        required=True,
        help=(
            "Type of plot to generate: 'agg' for aggregate comparison, 'time' for time series, "
            "'family' for per-family comparison, 'time_family' for per-family across times, "
            "'esm_params' for ESM2 vs PEINT across parameter sizes, "
            "'esm_vs_peint' for ESM vs PEINT scatterplot by family, "
            "'mutant_depth' for Spearman by mutational depth, "
            "'transition_times' for transition times distribution"
        ),
    )

    args = parser.parse_args()
    _set_publication_style()

    if args.plot == "agg":
        make_spearman_figure()

    elif args.plot == "family":
        make_per_family_spearman_figure()

    elif args.plot == "time_family":
        make_per_family_spearman_time_figure()

    elif args.plot == "esm_vs_peint":
        make_esm_vs_peint_figure()

    elif args.plot == "mutant_depth":
        make_spearman_by_mutational_depth_figure()

    elif args.plot == "time_family_best_vs_default":
        make_per_family_spearman_time_figure_best_vs_default()

    elif args.plot == "esmc_esm2":
        make_esmc_esm2_comparison()

    elif args.plot == "by_base_lm":
        make_multimodel_by_base_lm()
        make_mutational_depth_by_base_lm()

    elif args.plot == "params":
        make_params_vs_spearman_figure()
        make_params_vs_spearman_figure(assay_type="OrganismalFitness")

if __name__ == "__main__":
    main()
