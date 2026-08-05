"""Domain-level generalization test (approach #2): JSD restricted to the *novel* Pfam domain.

The family-level test asks whether a held-out family as a whole is novel. This one goes finer: it
locates each Pfam domain directly on ``seq1`` (pyhmmer, Pfam gathering thresholds), labels each
domain novel or seen by whether its Pfam accession appears anywhere in the 14,498 training
families, and then averages the per-site JSD over just that domain's alignment columns. So a single
held-out family can contribute both a "novel domain" JSD and a "seen domain" JSD, and — where a
family carries one of each — the two can be compared *within the same family*, the tightest
possible difficulty control.

The per-site JSD comes from ``paper.jsd.family_jsd`` (the paper's single JSD definition); seq1
residue ranges are mapped to alignment columns via ``paper.generalization.seq1_residue_to_column``.

Assumes the alignment pipeline has been run. First run fetches + decompresses the Pfam HMMs
(~400 MB) and scans the held-out seq1s (cached thereafter). Run from the repo root::

    python -m benchmarks.generalization_jsd_domain
"""

from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from paper import generalization as gen
from paper.jsd import REAL_OTHER_SPLIT
from paper.splits import generate_tree_split
from figures.figure3_conservation import DEFAULT_CONSERVATION_THRESHOLD, model_msa_dirs

FOCUS_MODELS = ["PEINT (Progressive)", REAL_OTHER_SPLIT, "LG+S256"]


def build_domain_jsd_table(force: bool = False) -> pd.DataFrame:
    """One row per (family, domain-class, model): mean JSD over that class's domain columns.

    domain-class is 'novel' or 'seen'. Columns of a domain = alignment columns whose seq1 residue
    lies in the domain's seq1 range AND that were scored as conserved by ``family_jsd``.
    """
    cache = Path(cfg.GENERALIZATION_DIR) / "domain_jsd_heldout.csv"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    cache.parent.mkdir(parents=True, exist_ok=True)

    families = gen.eval_families()
    domains = gen.scan_pfam_domains(families)
    # A held-out domain is "novel" only if its Pfam family is absent from training under BOTH the
    # hmmscan-on-seq1 vocabulary (same method as the held-out calls) and the SIFTS structure-mapped
    # vocabulary. Taking the union of the two training vocabularies is the conservative choice: it
    # minimizes false "novel" labels from either method missing a family.
    train_vocab = gen.hmm_pfam_vocab(gen.train_families()) | gen.partition("pfam_family")["train_vocab"]
    msa_dirs = model_msa_dirs()
    tree_dir = str(cfg.require(cfg.TREE_DIR))

    rows = []
    skipped = 0
    for family in families:
        hits = domains.get(family, [])
        if not hits:
            continue
        try:
            _, per_site = gen.fast_family_jsd(msa_dirs, family, generate_tree_split(tree_dir, family),
                                              DEFAULT_CONSERVATION_THRESHOLD)
            res2col = gen.seq1_residue_to_column(family)
        except (FileNotFoundError, ValueError, KeyError):
            skipped += 1
            continue
        scored_cols = {int(i) for i in per_site.index}

        # accumulate scored columns per domain-class for this family
        cols_by_class = {"novel": set(), "seen": set()}
        for h in hits:
            cls = "novel" if h["acc"] not in train_vocab else "seen"
            dom_cols = {res2col[r] for r in range(h["start"], h["end"] + 1) if r in res2col}
            cols_by_class[cls] |= (dom_cols & scored_cols)

        for cls, cols in cols_by_class.items():
            if not cols:
                continue
            idx = [str(c) for c in sorted(cols)]
            sub = per_site.loc[idx]
            for model in per_site.columns:
                rows.append({
                    "family": family, "domain_class": cls, "model": model,
                    "jsd": float(sub[model].mean()), "n_sites": len(idx),
                })
    if skipped:
        print(f"(skipped {skipped} families with no computable JSD)")
    df = pd.DataFrame(rows)
    df.to_csv(cache, index=False)
    return df


def _plot(df: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style

    _set_publication_style()
    plot_df = df[df["model"].isin(FOCUS_MODELS)].copy()
    plot_df["model"] = pd.Categorical(plot_df["model"], categories=FOCUS_MODELS, ordered=True)
    fig, ax = plt.subplots(figsize=(1.6 * len(FOCUS_MODELS), 3))
    sns.boxplot(data=plot_df, x="model", y="jsd", hue="domain_class",
                hue_order=["seen", "novel"], palette={"seen": "#9ecae1", "novel": "#fb6a4a"},
                showfliers=False, width=0.7, linewidth=0.5, ax=ax)
    sns.stripplot(data=plot_df, x="model", y="jsd", hue="domain_class", hue_order=["seen", "novel"],
                  dodge=True, size=2, alpha=0.4, palette={"seen": "#3182bd", "novel": "#cb181d"},
                  ax=ax, legend=False)
    ax.set_xlabel("")
    ax.set_ylabel("Mean JSD over domain sites", fontsize=9)
    ax.set_title("Novel vs seen Pfam domain (within held-out families)", fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[:2], l[:2], fontsize=7, frameon=False)
    sns.despine(ax=ax)
    out = Path(cfg.GENERALIZATION_DIR)
    fig.savefig(out / "generalization_jsd_domain.pdf", bbox_inches="tight")
    fig.savefig(out / "generalization_jsd_domain.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    gen.ensure_annotations()
    df = build_domain_jsd_table()
    n_novel_dom = df[df.domain_class == "novel"]["family"].nunique()
    n_seen_dom = df[df.domain_class == "seen"]["family"].nunique()
    print(f"Domain JSD: {df['family'].nunique()} families "
          f"({n_novel_dom} contribute a novel domain, {n_seen_dom} a seen domain)")

    # unpaired: novel-domain vs seen-domain JSD, per model
    records = []
    for model, g in df.groupby("model"):
        stat = gen.mann_whitney(
            g.loc[g.domain_class == "novel", "jsd"].values,
            g.loc[g.domain_class == "seen", "jsd"].values,
        )
        records.append({"model": model, **stat})
    stats = pd.DataFrame(records)
    stats.to_csv(Path(cfg.GENERALIZATION_DIR) / "generalization_jsd_domain_stats.csv", index=False)

    # paired: families carrying BOTH a novel and a seen domain (same-family difficulty control)
    print("\nNovel-domain vs seen-domain JSD (median; lower is better), per model:")
    for _, r in stats.iterrows():
        flag = "  <-- PEINT" if r["model"].startswith("PEINT") else (
            "  <-- Real baseline" if r["model"] == REAL_OTHER_SPLIT else "")
        print(f"    {r['model']:<28} novel={r.get('median_novel', float('nan')):.3f} "
              f"seen={r.get('median_seen', float('nan')):.3f} "
              f"n(novel/seen)={r['n_novel']}/{r['n_seen']} p={r['p']:.3g} "
              f"delta={r['cliffs_delta']:+.2f}{flag}")

    both = _paired_within_family(df)
    _plot(df)
    _plot_paired(both)
    print(f"\nWrote tables + figures to {cfg.GENERALIZATION_DIR}")


def _paired_within_family(df: pd.DataFrame) -> pd.DataFrame:
    """Wilcoxon signed-rank on families that carry both a novel and a seen domain.

    This is the tightest difficulty control: within a single held-out family, the novel domain and
    the seen domain share tree, depth and evolutionary regime, so any novel/seen JSD difference is
    the domain's novelty alone. Returns the per-(family, model) paired table (novel, seen columns).
    """
    from scipy.stats import wilcoxon

    wide = df.pivot_table(index=["family", "model"], columns="domain_class", values="jsd")
    both = wide.dropna(subset=["novel", "seen"])
    both.to_csv(Path(cfg.GENERALIZATION_DIR) / "generalization_jsd_domain_paired.csv")
    if both.empty:
        print("\n(no family carries both a novel and a seen domain — paired test skipped)")
        return both
    print(f"\nPaired within-family (families with both domain types), per model:")
    for model, g in both.groupby(level="model"):
        if len(g) < 5:
            continue
        try:
            _, p = wilcoxon(g["novel"], g["seen"])
        except ValueError:
            p = float("nan")
        print(f"    {model:<28} n_pairs={len(g)} "
              f"novel={g['novel'].median():.3f} seen={g['seen'].median():.3f} p={p:.3g}")
    return both


def _plot_paired(both: pd.DataFrame) -> None:
    """Per-family seen->novel domain-JSD slopes, PEINT beside Real — the paired control, visualized.

    Each grey line is one held-out family (its seen-domain vs novel-domain JSD); the bold red line is
    the median. PEINT's slope sitting on top of Real's shows the novel-domain penalty is intrinsic to
    the data, not specific to PEINT.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wilcoxon
    from paper.plot_style import _set_publication_style

    if both.empty:
        return
    _set_publication_style()
    panel_models = ["PEINT (Progressive)", REAL_OTHER_SPLIT]
    labels = {"PEINT (Progressive)": "PEINT", REAL_OTHER_SPLIT: "Real (eval subtree)"}
    fig, axes = plt.subplots(1, len(panel_models), figsize=(2.5 * len(panel_models), 3), sharey=True)
    n_pairs = 0

    for ax, model in zip(axes, panel_models):
        if model not in both.index.get_level_values("model"):
            continue
        g = both.xs(model, level="model")
        n_pairs = len(g)
        for _, row in g.iterrows():
            ax.plot([0, 1], [row["seen"], row["novel"]], color="#b0b0b0", lw=0.4, alpha=0.6, zorder=1)
        med_seen, med_novel = g["seen"].median(), g["novel"].median()
        ax.plot([0, 1], [med_seen, med_novel], color="#cb181d", lw=2, marker="o", ms=4, zorder=3)
        try:
            _, p = wilcoxon(g["novel"], g["seen"])
        except ValueError:
            p = float("nan")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["seen\ndomain", "novel\ndomain"], fontsize=8)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(f"{labels[model]}\n" + r"$\Delta$med=" + f"{med_novel - med_seen:+.3f}, p={p:.2g}",
                     fontsize=8)
        sns.despine(ax=ax)
    axes[0].set_ylabel("Mean JSD over domain sites", fontsize=9)
    fig.suptitle(f"Paired novel vs seen Pfam domain, within held-out families (n={n_pairs})",
                 fontsize=8, y=1.06)
    fig.subplots_adjust(wspace=0.25)
    fig.tight_layout(w_pad=2.0)
    out = Path(cfg.GENERALIZATION_DIR)
    fig.savefig(out / "generalization_jsd_domain_paired.pdf", bbox_inches="tight")
    fig.savefig(out / "generalization_jsd_domain_paired.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
