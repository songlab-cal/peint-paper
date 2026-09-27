"""Stratify the paper's metrics by how far each held-out family sits from training in sequence space.

The Pfam split asks whether a family's *labels* were in training. This asks whether its *sequences*
were, using the DIAMOND search in ``benchmarks.heldout_train_similarity``, and re-runs the same
novel-vs-seen comparisons on the answer. Run that benchmark's ``blastp`` stage first::

    python -m benchmarks.generalization_seqdistance
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from paper import generalization as gen
from paper import transition_similarity as ts
from benchmarks.heldout_train_similarity import family_distance_table

# Three readings of "far from training", because they are not the same families and a claim that
# holds for one is not automatically true of the others.
#
#   no_homolog  nothing in the entire training set aligns to any sequence of this family. The
#               strongest statement available, and the smallest stratum.
#   sparse      a family's typical sequence reaches no training family at all above 30% identity.
#               Targets thin neighbourhoods rather than distant ones: a family can have one good
#               match and still be isolated.
#   low_identity the typical sequence's best training match is under 30% identity. The plain
#               reading of "little sequence identity to train".
SCHEMES = {
    "no_homolog": lambda t: t["no_hit_pct"] >= 100.0,
    "sparse": lambda t: t["med_n_fam_ge30"] <= 0,
    "low_identity": lambda t: t["median_pident"] < 30.0,
}

# Model sets differ per table because the three result generations named their arms differently.
METRICS = {
    "omegafold_plddt": {
        "value": "plddt",
        "label": "OmegaFold pLDDT",
        "models": ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"],
        "higher_is_better": True,
        "baseline": "Real",
    },
    "family_jsd": {
        "value": "jsd",
        "label": "Mean JSD vs. real",
        "models": ["LG+S256", "PEINT (Progressive)", "Real (other split)"],
        "higher_is_better": False,
        "baseline": "Real (other split)",
    },
    # ESM-C was simulated into its own mafft frame, with its own Real reference, so its JSD is a
    # separate arm rather than another column beside PEINT (ESM2). Comparing it against the rev1
    # Real would mix frames; each arm is read against the Real of the frame it was computed in.
    "family_jsd_esmc": {
        "value": "jsd",
        "label": "Mean JSD vs. real (ESM-C frame)",
        "models": ["PEINT (ESM-C)", "Real (other split)"],
        "higher_is_better": False,
        "baseline": "Real (other split)",
    },
    # ESM-IF, both readouts. Approach 2 (self-consistency, each sequence on its OWN OmegaFold
    # structure) is template-free and is the one to lead with. Approach 1 threads onto the fixed GT
    # structure, and its likelihood tracks identity-to-seq1, so on these families it reports Real
    # *improving* simply because Real's leaves sit closer to seq1 there -- read it with the
    # esmif_a1_identity column, not on its own.
    "esmif_recovery": {
        "value": "recovery",
        "label": "ESM-IF sequence recovery (self-consistency)",
        "models": ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"],
        "higher_is_better": True,
        "baseline": "Real",
    },
    "esmif_sc_ll": {
        "value": "ll",
        "label": "ESM-IF log-likelihood (self-consistency)",
        "models": ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"],
        "higher_is_better": True,
        "baseline": "Real",
    },
    "esmif_gt_ll": {
        "value": "ll",
        "label": "ESM-IF log-likelihood (threaded on GT structure)",
        "models": ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"],
        "higher_is_better": True,
        "baseline": "Real",
    },
    "af2rank_plddt": {
        "value": "plddt",
        "label": "AF2Rank pLDDT",
        "models": ["LG+S256", "PEINT (Progressive)", "Real (other split)"],
        "higher_is_better": True,
        "baseline": "Real (other split)",
    },
}


def load_omegafold() -> pd.DataFrame:
    for d in (cfg.FIGURE_DATA_DIR, cfg.FIGURES_DIR):
        p = Path(d) / "figure3_omegafold_plddt_ecdf.csv"
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("figure3_omegafold_plddt_ecdf.csv not found")


def load_family_jsd() -> pd.DataFrame:
    p = Path(cfg.GENERALIZATION_DIR) / "family_jsd_heldout.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- run benchmarks.generalization_jsd_family first")
    return (pd.read_csv(p, index_col=0).reset_index(names="family")
            .melt(id_vars="family", var_name="model", value_name="jsd").dropna(subset=["jsd"]))


def load_family_jsd_esmc() -> pd.DataFrame:
    """Family JSD in the ESM-C frame. Produced by collect_family_jsd_fast over the rev2 mafft
    dirs; cached because recomputing it needs paper.jsd, which imports a `peint` package that the
    installed checkout provides under the name `protevo`."""
    p = Path(cfg.GENERALIZATION_DIR) / "family_jsd_heldout_esmc.csv"
    if not p.exists():
        raise FileNotFoundError(f"{p} -- see the module docstring for how it is produced")
    return (pd.read_csv(p, index_col=0).reset_index(names="family")
            .melt(id_vars="family", var_name="model", value_name="jsd").dropna(subset=["jsd"]))


def _load_esmif(name: str) -> pd.DataFrame:
    for d in (Path(cfg.FIGURE_DATA_DIR) / "esmif", Path(cfg.FIGURES_DIR) / "esmif"):
        p = d / name
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(f"{name} not found -- run benchmarks.esmif_validation first")


def load_esmif_selfconsistency() -> pd.DataFrame:
    return _load_esmif("esmif_selfconsistency.csv")


def load_esmif_gt() -> pd.DataFrame:
    return _load_esmif("esmif_gt_likelihood.csv")


def load_af2rank() -> pd.DataFrame:
    p = Path(cfg.RESULTS_R1_DIR) / "af2rank_comparisons.csv"
    if not p.exists():
        raise FileNotFoundError(str(p))
    d = pd.read_csv(p)
    # pLDDT ships on 0-1 here and 0-100 in the OmegaFold table; rescale so the axis label is honest.
    if d["plddt"].max() <= 1.5:
        d = d.assign(plddt=d["plddt"] * 100.0)
    return d[["family", "model", "plddt"]]


LOADERS = {"omegafold_plddt": load_omegafold, "family_jsd": load_family_jsd,
           "family_jsd_esmc": load_family_jsd_esmc, "af2rank_plddt": load_af2rank,
           "esmif_recovery": load_esmif_selfconsistency,
           "esmif_sc_ll": load_esmif_selfconsistency, "esmif_gt_ll": load_esmif_gt}


def strata(level: str) -> pd.DataFrame:
    """Per-family distance table plus one novel/seen column per scheme."""
    t = family_distance_table(level)
    for name, rule in SCHEMES.items():
        t[name] = np.where(rule(t), "novel", "seen")
    return t


def _covariate_frame(families) -> pd.DataFrame:
    """Family -> query length, for the covariate-matched repeat of each test.

    A family far from training may simply be shorter or shallower, and a metric that tracks length
    would then look like it tracks novelty. Matching on length is the same control the Pfam
    stratification uses, so the two analyses are read on the same terms.
    """
    cov = gen.family_covariates(list(families))
    return pd.DataFrame(
        [{"family": f, "length": c.get("length", np.nan)} for f, c in cov.items()]
    )


def _excess_over_baseline(stats: pd.DataFrame, spec: dict) -> pd.DataFrame:
    """How much worse a model gets on distant families *than the real data does*.

    The distant families turn out to be harder for everything, the Real arm included, so a model
    losing ground there is not by itself a generalization failure -- it may just be a harder set of
    proteins. The quantity that separates the two is the difference of differences: the model's
    novel-minus-seen change, minus the same change for Real. Signed so that positive always means
    "degrades more than the real data", whichever direction the metric runs in.
    """
    stats = stats.copy()
    stats["delta_median"] = stats["median_novel"] - stats["median_seen"]
    base = spec.get("baseline")
    out = []
    for scheme, g in stats.groupby("scheme"):
        g = g.copy()
        ref = g.loc[g["model"] == base, "delta_median"]
        if ref.empty:
            g["excess_vs_real"] = np.nan
        elif spec["higher_is_better"]:
            g["excess_vs_real"] = float(ref.iloc[0]) - g["delta_median"]
        else:
            g["excess_vs_real"] = g["delta_median"] - float(ref.iloc[0])
        out.append(g)
    return pd.concat(out, ignore_index=True)


def run(metric: str, level: str, out_dir: Path) -> pd.DataFrame:
    spec = METRICS[metric]
    df = LOADERS[metric]()
    t = strata(level)
    cov = _covariate_frame(t["family"])
    df = df.merge(t[["family", *SCHEMES]], on="family", how="inner").merge(cov, on="family", how="left")

    rows = []
    for scheme in SCHEMES:
        for model in spec["models"]:
            g = df[df["model"] == model]
            if g.empty:
                continue
            a = g.loc[g[scheme] == "novel", spec["value"]]
            b = g.loc[g[scheme] == "seen", spec["value"]]
            if len(a) < 5 or len(b) < 5:
                continue
            stat = gen.mann_whitney(a.to_numpy(), b.to_numpy())
            # Repeat on a length-matched subsample, so a difference that is really about length
            # shows up as a p-value that collapses here while the unmatched one looked significant.
            m = g.set_index("family")
            try:
                sub = gen.matched_subsample(m, m[scheme] == "novel", covariate="length")
                ms = gen.mann_whitney(
                    sub.loc[sub[scheme] == "novel", spec["value"]].to_numpy(),
                    sub.loc[sub[scheme] == "seen", spec["value"]].to_numpy(),
                )
            except (ValueError, KeyError):
                ms = {"p": np.nan, "n_novel": 0, "n_seen": 0}
            rows.append({"metric": metric, "level": level, "scheme": scheme, "model": model,
                         **stat, "p_matched": ms["p"], "n_matched": ms["n_novel"]})

    stats = pd.DataFrame(rows)
    if stats.empty:
        return stats
    stats = _excess_over_baseline(stats, spec)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats.to_csv(out_dir / f"seqdistance_{metric}_{level}_stats.csv", index=False)

    for scheme in SCHEMES:
        plotted = df.rename(columns={scheme: "_stratum_src"})
        n_novel = (t[scheme] == "novel").sum()
        if n_novel < 5:
            continue
        gen.plot_grouped_by_novelty(
            plotted, spec["value"], f"seqdistance_{metric}_{level}_{scheme}",
            model_order=[m for m in spec["models"] if m in set(df["model"])],
            stratum_col="_stratum_src", value_label=spec["label"],
            title=f"{scheme.replace('_', ' ')} ({level}-length): {n_novel} of {len(t)} held-out families",
            out_dir=str(out_dir),
        )
    return stats


# Families grouped by how many DISTINCT training families their typical sequence reaches. Counting
# distinct families rather than hits is deliberate: one training family contributes ~1,000
# near-duplicates, so a raw hit count measures database redundancy, not how well connected a
# held-out family is to training.
HIT_BINS = [
    ("0 hits", lambda v: v == 0),
    ("1-2 hits", lambda v: v.between(1, 2)),
    ("3-4 hits", lambda v: v.between(3, 4)),
    ("5-10 hits", lambda v: v.between(5, 10)),
    (">10 hits", lambda v: v > 10),
]


def report_by_hit_count(out_dir: Path, level: str = "full",
                        metrics=("esmif_recovery", "esmif_sc_ll", "esmif_gt_ll",
                                 "omegafold_plddt")) -> pd.DataFrame:
    """Metric medians against how many training families a held-out family actually reaches.

    A binary distant/close split cannot show whether a metric degrades smoothly with distance from
    training or falls off a cliff at zero. It does the latter here: the 0-hit bin is far below the
    rest, while 3-4, 5-10 and >10 are indistinguishable, so connectivity saturates almost
    immediately. Reading Real down the same column is what separates "this region is hard for the
    whole pLM and folding stack" from "this model fails here".
    """
    t = family_distance_table(level).set_index("family")
    groups = [(name, set(t[rule(t["med_n_families"])].index)) for name, rule in HIT_BINS]

    rows = []
    for name, fams in groups:
        sub = t.reindex([f for f in fams if f in t.index])
        rows.append({"metric": "identity_to_train", "bin": name, "n_families": len(sub),
                     "median_pident": round(sub["median_pident"].median(), 2),
                     "mean_pident": round(sub["median_pident"].mean(), 2),
                     "median_no_hit_pct": round(sub["no_hit_pct"].median(), 2)})
    for metric in metrics:
        spec = METRICS[metric]
        try:
            df = LOADERS[metric]()
        except FileNotFoundError as exc:
            print(f"  ({metric} skipped: {exc})")
            continue
        for name, fams in groups:
            rec = {"metric": metric, "bin": name}
            for model in spec["models"]:
                g = df[(df["model"] == model) & (df["family"].isin(fams))]
                if g.empty:
                    continue
                rec[f"{model}__median"] = round(g[spec["value"]].median(), 4)
                rec[f"{model}__mean"] = round(g[spec["value"]].mean(), 4)
                rec["n_families"] = g["family"].nunique()
            rows.append(rec)
    out = pd.DataFrame(rows)
    # Share of families each bin holds, per metric: the bins are very unequal (the 0-hit bin is
    # 18% of families, the >10 bin 8%), so a row read without it looks like four equal groups.
    # Computed within each metric because the metric tables do not all cover the same families.
    out["pct_families"] = (
        100 * out["n_families"] / out.groupby("metric")["n_families"].transform("sum")
    ).round(1)
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / f"seqdistance_by_hit_count_{level}.csv", index=False)
    print(f"\nMetrics by number of distinct training families reached ({level}-length):")
    for metric in out["metric"].unique():
        d = out[out["metric"] == metric].dropna(axis=1, how="all").copy()
        d["bin"] = [f"{b} ({int(n)}, {p:.1f}%)"
                    for b, n, p in zip(d["bin"], d["n_families"], d["pct_families"])]
        print(f"\n[{metric}]")
        print(d.drop(columns=["metric", "n_families", "pct_families"]).to_string(index=False))
    return out


# Coarser than HIT_BINS: three grades of connectivity to training, for the ESM-IF panel. The fine
# bins showed 3-4, 5-10 and >10 to be indistinguishable, so collapsing them loses nothing and the
# three groups that remain are the ones that actually differ.
ESMIF_GRADES = [
    ("0 hits", lambda v: v == 0),
    ("1-10 hits", lambda v: v.between(1, 10)),
    (">10 hits", lambda v: v > 10),
]
ESMIF_MODELS = ["LG+S256", "PEINT (ESM2)", "PEINT (ESM-C)", "Real"]


def plot_esmif_by_hit_count(out_dir: Path, level: str = "full") -> None:
    """ESM-IF self-consistency against connectivity to training, graded in three steps.

    Both rows are template-free: each sequence is scored on its own OmegaFold structure, so
    nothing here is conditioned on the experimental structure. Real is plotted beside the models
    on purpose -- it is natural sequence, so wherever Real falls too, the drop belongs to that
    region of sequence/structure space rather than to any model.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style
    from paper.model_style import model_colors

    _set_publication_style()
    t = family_distance_table(level).set_index("family")
    grade = {}
    for name, rule in ESMIF_GRADES:
        for f in t[rule(t["med_n_families"])].index:
            grade[f] = name

    df = load_esmif_selfconsistency()
    df = df[df["model"].isin(ESMIF_MODELS)].copy()
    df["grade"] = df["family"].map(grade)
    df = df.dropna(subset=["grade"])
    df["model"] = pd.Categorical(df["model"], categories=ESMIF_MODELS, ordered=True)

    order = [g for g, _ in ESMIF_GRADES]
    # Family counts come from one model's rows, not from len(df): every model contributes a row
    # per family, so counting the frame would report four times the families there are.
    ref = df[df["model"] == "Real"]
    total = ref["family"].nunique()
    ticks = []
    for g in order:
        n = ref.loc[ref["grade"] == g, "family"].nunique()
        ticks.append(f"{g}\n{n} families ({100 * n / total:.0f}%)")

    colors = model_colors()
    rows = [("ll", "ESM-IF log-likelihood\n(self-consistency)"),
            ("recovery", "ESM-IF sequence recovery\n(self-consistency)")]
    fig, axes = plt.subplots(len(rows), 1, figsize=(9.5, 7.2), sharex=True)

    for ax, (value, ylabel) in zip(axes, rows):
        sns.boxplot(data=df, x="grade", y=value, hue="model", order=order,
                    hue_order=ESMIF_MODELS, palette={m: colors[m] for m in ESMIF_MODELS},
                    showfliers=False, width=0.74, linewidth=0.5, ax=ax)
        # Each model's median in the BEST-connected grade, carried across as a dotted reference.
        # The panel is about how far each arm falls as connectivity drops, and the arms sit at
        # very different absolute levels, so a per-model reference is the only way to compare
        # those falls by eye.
        for m in ESMIF_MODELS:
            v = df[(df["model"] == m) & (df["grade"] == ">10 hits")][value].median()
            ax.axhline(v, color=colors[m], lw=0.6, ls=":", alpha=0.7, zorder=0)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("")
        ax.legend_.remove() if ax.legend_ else None
        sns.despine(ax=ax)

    axes[-1].set_xticks(range(len(order)))
    axes[-1].set_xticklabels(ticks, fontsize=9)
    axes[-1].set_xlabel("Distinct training families reached by the typical held-out sequence",
                        fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[m]) for m in ESMIF_MODELS]
    axes[0].legend(handles, ESMIF_MODELS, fontsize=8, frameon=False, ncol=len(ESMIF_MODELS),
                   loc="lower center", bbox_to_anchor=(0.5, 1.01))
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"esmif_by_hit_count_{level}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir}/esmif_by_hit_count_{level}.{{pdf,png}}")


def plot_similarity_cdf(out_dir: Path, levels=("full", "domain")) -> None:
    """CDFs of identity to the closest training sequence, Pfam-seen vs Pfam-novel.

    The CDF is what a median hides: the Pfam-novel curve sits at 0% for most of its mass, but its
    upper tail reaches all the way to 100%, so the same set contains families with no training
    homolog at all and families with a near-identical one.

    Left panel counts sequences, right panel counts families by their median, because the two
    answer different questions -- how much of the evaluation data has a close relative in
    training, and how many families are close as a whole.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style
    from benchmarks.heldout_train_similarity import fasta_path, work_dir

    _set_publication_style()
    pf = gen.partition("pfam_family")
    fig, axes = plt.subplots(len(levels), 2, figsize=(9.2, 3.5 * len(levels)), squeeze=False)

    for i, level in enumerate(levels):
        t = family_distance_table(level).set_index("family")
        groups = {
            "Pfam seen": (set(pf["seen"]), "#4878cf"),
            "Pfam novel": (set(pf["novel"]), "#c44e52"),
        }
        prof = pd.read_csv(work_dir() / f"density_{level}.csv.gz", usecols=["qseqid", "pident_rank1"])
        ids = [ln[1:].strip() for ln in ts._open_text(fasta_path("test", level)) if ln.startswith(">")]
        # Reindex onto every submitted query: a sequence with no training hit belongs at 0%, and
        # dropping it would quietly lift every curve by the no-hit rate of its group.
        d = pd.DataFrame({"qseqid": ids}).merge(prof, on="qseqid", how="left")
        d["family"] = d["qseqid"].str.split(ts.ID_SEP).str[0]
        d["pident"] = d["pident_rank1"].fillna(0.0)

        for ax, (unit, getter) in zip(axes[i], [
            ("sequences", lambda fams: d.loc[d["family"].isin(fams), "pident"].to_numpy()),
            ("families (per-family median)",
             lambda fams: t.reindex([f for f in fams if f in t.index])["median_pident"].dropna().to_numpy()),
        ]):
            for label, (fams, color) in groups.items():
                x = np.sort(getter(fams))
                if not x.size:
                    continue
                ax.step(np.concatenate([[0], x]),
                        np.concatenate([[0], 100 * np.arange(1, x.size + 1) / x.size]),
                        where="post", lw=1.5, color=color, label=f"{label} (n={x.size:,})")
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.axvline(95, color="0.5", ls=":", lw=0.8)
            ax.set_xlabel("% identity to closest training sequence", fontsize=9)
            ax.set_ylabel(f"Cumulative % of {unit}", fontsize=9)
            ax.set_title(f"{level}-length queries, by {unit.split(' ')[0]}", fontsize=9)
            # Opaque frame, not frameon=False: the novel-group curves jump to ~95% at x=0 and run
            # straight through the upper-left corner, so an unframed legend is read over the top of
            # them and its swatches take on the colour of whatever line is behind them.
            ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.92,
                      edgecolor="0.8", borderpad=0.4)
            sns.despine(ax=ax)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"similarity_to_train_cdf.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir}/similarity_to_train_cdf.{{pdf,png}}")


def pfam_crosstab(t: pd.DataFrame, out_dir: Path, level: str) -> pd.DataFrame:
    """Where the Pfam-label novelty split and the sequence-distance split agree, and where not.

    They are not the same families, and the difference is not noise. Pfam calls a family novel when
    its domain *accession* never appears in training, but Pfam splits one superfamily across many
    accessions: 2a6c_1_B is novel on PF13744 (HTH_37) while training is full of HTH domains under
    PF01381, PF13560 and PF13443 -- all clan CL0123 -- and some of its sequences are 100% identical
    to training sequences. 58 of the 81 family-novel families share a clan with training.

    So the label axis over-calls novelty for a sizeable minority, and this table is what shows it.
    """
    pf, pc = gen.partition("pfam_family"), gen.partition("pfam_clan")
    lut = {f: "pfam_novel" for f in pf["novel"]}
    lut.update({f: "pfam_seen" for f in pf["seen"]})
    lut.update({f: "pfam_unlabeled" for f in pf["unlabeled"]})
    t = t.copy()
    t["pfam"] = t["family"].map(lut).fillna("not_partitioned")
    t["pfam_clan_novel"] = t["family"].isin(pc["novel"])

    rows = []
    for label, mask in (
        ("pfam family-novel, clan SEEN", t["family"].isin(pf["novel"] - pc["novel"])),
        ("pfam family-novel AND clan-novel", t["family"].isin(pf["novel"] & pc["novel"])),
        ("pfam seen", t["pfam"] == "pfam_seen"),
        ("pfam unlabeled", t["pfam"] == "pfam_unlabeled"),
    ):
        sub = t[mask]
        if sub.empty:
            continue
        rows.append({
            "group": label, "n": len(sub),
            "median_closest_pident": round(sub["median_pident"].median(), 1),
            "mean_closest_pident": round(sub["median_pident"].mean(), 1),
            "pct_families_zero_hits": round(100 * (sub["no_hit_pct"] >= 100).mean(), 1),
            "pct_families_with_ge95_match": round(100 * (sub["max_pident"] >= 95).mean(), 1),
            "median_n_train_families_ge30": round(sub["med_n_fam_ge30"].median(), 1),
            **{f"pct_{k}": round(100 * (sub[k] == "novel").mean(), 1) for k in SCHEMES},
        })
    out = pd.DataFrame(rows)
    out.to_csv(out_dir / f"seqdistance_vs_pfam_{level}.csv", index=False)
    print("\nPfam-label novelty vs sequence distance:")
    print(out.to_string(index=False))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", choices=("full", "domain"), default="full")
    ap.add_argument("--metrics", nargs="*", default=list(METRICS))
    ap.add_argument("--cdf", action="store_true", help="Also draw the similarity-to-training CDFs.")
    ap.add_argument("--hit-bins", action="store_true",
                    help="Also tabulate metrics against the number of training families reached.")
    args = ap.parse_args()

    out_dir = Path(cfg.SIMILARITY_DIR)
    t = strata(args.level)
    print(f"Held-out families ({args.level}-length queries): {len(t)}")
    for name in SCHEMES:
        n = (t[name] == "novel").sum()
        print(f"  {name:<14} novel={n:>4}  seen={len(t) - n:>4}")
    t.to_csv(out_dir / f"seqdistance_strata_{args.level}.csv", index=False)
    pfam_crosstab(t, out_dir, args.level)
    if args.cdf:
        plot_similarity_cdf(out_dir)
    if args.hit_bins:
        report_by_hit_count(out_dir, args.level)
        plot_esmif_by_hit_count(out_dir, args.level)

    allstats = []
    for metric in args.metrics:
        try:
            s = run(metric, args.level, out_dir)
        except FileNotFoundError as exc:
            print(f"\n[{metric}] skipped: {exc}")
            continue
        if s.empty:
            print(f"\n[{metric}] no comparable strata")
            continue
        allstats.append(s)
        spec = METRICS[metric]
        arrow = "higher is better" if spec["higher_is_better"] else "lower is better"
        print(f"\n[{metric}] {spec['label']} ({arrow})")
        for scheme in s["scheme"].unique():
            print(f"  [{scheme}]")
            for _, r in s[s["scheme"] == scheme].iterrows():
                mark = "" if r["model"] == spec.get("baseline") else (
                    f" excess_vs_real={r['excess_vs_real']:+.3f}")
                print(f"    {r['model']:<24} distant={r['median_novel']:7.3f} "
                      f"close={r['median_seen']:7.3f} change={r['delta_median']:+.3f} "
                      f"n={r['n_novel']}/{r['n_seen']} p={r['p']:.3g} "
                      f"p_matched={r['p_matched']:.3g}{mark}")
    if allstats:
        pd.concat(allstats, ignore_index=True).to_csv(
            out_dir / f"seqdistance_all_stats_{args.level}.csv", index=False)
    print(f"\nWrote tables + panels to {out_dir}")


if __name__ == "__main__":
    main()
