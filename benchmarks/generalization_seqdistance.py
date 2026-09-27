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
           "af2rank_plddt": load_af2rank}


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
    args = ap.parse_args()

    out_dir = Path(cfg.SIMILARITY_DIR)
    t = strata(args.level)
    print(f"Held-out families ({args.level}-length queries): {len(t)}")
    for name in SCHEMES:
        n = (t[name] == "novel").sum()
        print(f"  {name:<14} novel={n:>4}  seen={len(t) - n:>4}")
    t.to_csv(out_dir / f"seqdistance_strata_{args.level}.csv", index=False)
    pfam_crosstab(t, out_dir, args.level)

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
