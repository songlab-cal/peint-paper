"""How similar is the held-out data to the training data, by DIAMOND search.

Builds a DIAMOND database from every training sequence and searches the held-out families'
evaluation sequences against it, reporting each query's closest training match. Run at two
granularities: the whole chain, and each Pfam domain sliced out on its own, so a chain that is
novel only outside its domains cannot hide behind a conserved one (or vice versa).

Stages are separate subcommands because the search is a cluster job::

    python -m benchmarks.heldout_train_similarity fasta
    python -m benchmarks.heldout_train_similarity makedb
    python -m benchmarks.heldout_train_similarity blastp
    python -m benchmarks.heldout_train_similarity report

``sbatch benchmarks/run_heldout_train_similarity.sbatch`` runs the first three on the epurdom
partition; ``report`` is cheap and runs anywhere.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from paper import generalization as gen
from paper import transition_similarity as ts

SAMPLE_SEED = 0


# ======================================================================================
# paths
# ======================================================================================
def work_dir() -> Path:
    d = Path(cfg.SIMILARITY_WORK_DIR)
    d.mkdir(parents=True, exist_ok=True)
    return d


def fasta_path(name: str, level: str) -> Path:
    return work_dir() / "fasta" / f"{name}_{level}.fasta"


def db_path(level: str) -> Path:
    return work_dir() / "db" / f"train_{level}"


def hits_path(level: str) -> Path:
    return work_dir() / "hits" / f"test_{level}.tsv.gz"


def _domains() -> dict:
    """Pfam domains on each family's seq1, keyed by family. Cached for all 15,051 families."""
    return gen.scan_pfam_domains(sorted(set(gen.train_families()) | set(gen.eval_families())))


# ======================================================================================
# stages
# ======================================================================================
def stage_fasta(args) -> None:
    domains = _domains()
    train = sorted(gen.train_families())
    held = sorted(gen.eval_families())
    print(f"train={len(train)} families (database), held-out={len(held)} families (queries)")

    for name, families, split in (("train", train, "train"), ("test", held, "test")):
        for level in ts.LEVELS:
            out = fasta_path(name, level)
            if out.exists() and not args.force:
                print(f"  {out.name} exists, skipping (use --force to rebuild)")
                continue
            cap = args.max_seqs_per_family if name == "test" else args.db_seqs_per_family
            print(f"  building {out.name}: {len(families)} families, split={split}, "
                  f"cap={cap or 'all'}/family")
            stats = ts.write_fasta(
                out, families, split, level, domains,
                max_seqs=cap, seed=SAMPLE_SEED, workers=args.workers,
                strict_frame=not args.no_frame_check,
            )
            print(f"    {stats['sequences']:,} sequences from {stats['families']:,} families "
                  f"-> {out} ({out.stat().st_size / 1e9:.2f} GB)")


def stage_makedb(args) -> None:
    for level in ts.LEVELS:
        fa = fasta_path("train", level)
        if not fa.exists():
            raise FileNotFoundError(f"{fa} missing -- run the `fasta` stage first.")
        print(f"  makedb {level}")
        ts.makedb(fa, db_path(level), threads=args.threads)


def stage_blastp(args) -> None:
    levels = ts.LEVELS if args.level == "all" else (args.level,)
    for level in levels:
        out = hits_path(level)
        if out.exists() and not args.force:
            print(f"  {out.name} exists, skipping (use --force to rerun)")
            continue
        query = fasta_path("test", level)
        if not query.exists():
            raise FileNotFoundError(f"{query} missing -- run the `fasta` stage first.")
        n_parts = args.query_chunks_domain if level == "domain" else args.query_chunks_full
        print(f"  blastp test x {level} ({n_parts} query chunk(s))")

        parts = ([query] if n_parts <= 1
                 else ts.split_fasta(query, n_parts, work_dir() / "fasta" / f"test_{level}_parts"))
        hit_parts = []
        for i, part in enumerate(parts):
            hp = out.parent / f"{out.name.replace('.tsv.gz', '')}.part{i}.tsv.gz"
            hit_parts.append(hp)
            if hp.exists() and not args.force:
                print(f"    part {i}: exists, skipping")
                continue
            print(f"    part {i+1}/{len(parts)}")
            ts.blastp(
                db_path(level), part, hp,
                threads=args.threads, sensitivity=args.sensitivity,
                max_target_seqs=args.max_target_seqs, evalue=args.evalue,
                block=args.block, index_chunks=args.index_chunks,
                tmpdir=str(work_dir() / "tmp"), hit_membuf=not args.no_hit_membuf,
            )
        if len(hit_parts) == 1:
            os.replace(hit_parts[0], out)
        else:
            ts.concat_gzip(hit_parts, out)
            for hp in hit_parts:
                hp.unlink()
        print(f"    -> {out} ({out.stat().st_size / 1e6:.0f} MB)")


# ======================================================================================
# report
# ======================================================================================
def build_table(force: bool = False) -> pd.DataFrame:
    """One row per submitted query, both levels, with its closest training match.

    Also labels each row by whether the family (and, at domain level, the specific Pfam family)
    was seen in training, so the sequence-space distance can be read against the label-space
    novelty split the rest of the generalization analysis uses.
    """
    cache = work_dir() / "per_sequence_hits.csv.gz"
    if cache.exists() and not force:
        return pd.read_csv(cache)

    part = gen.partition("pfam_family")
    fam_novelty = {f: "novel" for f in part["novel"]}
    fam_novelty.update({f: "seen" for f in part["seen"]})
    # Union of the hmmscan-on-seq1 and SIFTS training vocabularies, matching
    # benchmarks.generalization_jsd_domain: it minimizes false "novel" calls from either source.
    train_vocab = gen.hmm_pfam_vocab(gen.train_families()) | part["train_vocab"]

    frames = []
    for level in ts.LEVELS:
        tsv, fa = hits_path(level), fasta_path("test", level)
        if not (tsv.exists() and fa.exists()):
            print(f"  (skipping {level}: not searched yet)")
            continue
        print(f"  reducing {tsv.name}")
        d = ts.hits_table(tsv, fa)
        d["level"] = level
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No hit tables found -- run the `blastp` stage first.")

    d = pd.concat(frames, ignore_index=True)
    d["family_novelty"] = d["family"].map(fam_novelty)
    d["domain_novelty"] = np.where(
        d["pfam_acc"].notna(),
        np.where(d["pfam_acc"].isin(train_vocab), "seen", "novel"),
        None,
    )
    d.to_csv(cache, index=False)
    return d


def summarize(d: pd.DataFrame) -> pd.DataFrame:
    """Per-family medians of the closest-training-match statistics.

    Families, not sequences, are the unit: each contributes hundreds of closely related queries,
    so pooling sequences would weight families by how many they happen to have. Queries with no
    hit score 0, which is what "nothing in training resembles this" means on an identity scale;
    the no-hit rate is reported separately because a median hides it.
    """
    rows = []
    for level in sorted(d["level"].unique()):
        v = d[d["level"] == level].assign(
            hit=lambda x: x["bitscore"].notna(),
            pident0=lambda x: x["pident"].fillna(0.0),
            gident0=lambda x: x["gident"].fillna(0.0),
            qcov0=lambda x: x["qcov"].fillna(0.0),
        )
        g = v.groupby("family", sort=True)
        rows.append(pd.DataFrame({
            "level": level,
            "n_queries": g.size(),
            "hit_rate": 100 * g["hit"].mean(),
            "median_pident": g["pident0"].median(),
            "median_gident": g["gident0"].median(),
            "median_qcov": g["qcov0"].median(),
            "median_bitscore": g["bitscore"].median(),
        }).reset_index())
    return pd.concat(rows, ignore_index=True)


def plot(d: pd.DataFrame, fam: pd.DataFrame, out_dir: Path) -> None:
    """Two panels per granularity: the full distribution, and the novel/seen split."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style

    _set_publication_style()
    levels = [l for l in ts.LEVELS if l in set(fam["level"])]
    colors = {"full": "#4878cf", "domain": "#55a868"}
    nov_colors = {"seen": "#4878cf", "novel": "#c44e52"}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4))

    # Left: ECDF over sequences, one labelled line per granularity.
    ax = axes[0]
    for level in levels:
        x = np.sort(d.loc[d["level"] == level, "gident"].fillna(0.0).to_numpy())
        if x.size:
            ax.step(x, 100 * np.arange(1, x.size + 1) / x.size, where="post",
                    color=colors[level], lw=1.5,
                    label=f"{level}-length query (n={x.size:,})")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% identity to closest training sequence", fontsize=9)
    ax.set_ylabel("Cumulative % of held-out sequences", fontsize=9)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    sns.despine(ax=ax)

    # Right: per-family medians split by whether the Pfam family was seen in training.
    ax = axes[1]
    positions, ticks, data, facecolors = [], [], [], []
    pos = 1
    for level in levels:
        sub = fam[fam["level"] == level]
        key = "domain_novelty" if level == "domain" else "family_novelty"
        nov = (d[d["level"] == level].groupby("family")[key].first()
               if key in d.columns else None)
        merged = sub.merge(nov.rename("novelty"), on="family", how="left")
        for s in ("seen", "novel"):
            vals = merged.loc[merged["novelty"] == s, "median_gident"].dropna().to_numpy()
            if not vals.size:
                continue
            positions.append(pos)
            ticks.append(f"{level}\n{s}\n(n={vals.size})")
            data.append(vals)
            facecolors.append(nov_colors[s])
            pos += 1
        pos += 0.6
    if data:
        bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6,
                        showfliers=False, boxprops=dict(linewidth=0.5),
                        whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                        medianprops=dict(color="black", linewidth=1.0))
        for patch, c in zip(bp["boxes"], facecolors):
            patch.set_facecolor(c)
        ax.set_xticks(positions)
        ax.set_xticklabels(ticks, fontsize=7)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Per-family median % identity", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=nov_colors[s]) for s in ("seen", "novel")]
    ax.legend(handles, ["Pfam seen in training", "Pfam novel"],
              fontsize=7.5, frameon=False, loc="lower left")
    sns.despine(ax=ax)

    fig.suptitle("Held-out sequences vs their closest match in the training set (DIAMOND)",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"heldout_train_similarity.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir}/heldout_train_similarity.{{pdf,png}}")


def report_density(out_dir: Path) -> None:
    """Is the nearest training sequence an isolated match, or one of a crowd?

    Rank 1 alone cannot tell those apart, and they mean opposite things: a query whose closest
    training sequence is 50% identical and whose next-closest is 25% sits at the edge of the
    training distribution, while one where the next twenty are all ~50% sits inside it. Reports
    the identity profile down the ranked hit list, both raw and after collapsing each training
    family to its best hit, plus how many distinct training families clear each identity level.
    """
    for level in ts.LEVELS:
        tsv = hits_path(level)
        if not tsv.exists():
            continue
        cache = work_dir() / f"density_{level}.csv.gz"
        if cache.exists():
            prof = pd.read_csv(cache)
        else:
            print(f"  profiling hit density for {level}")
            prof = ts.density_profile(tsv)
            prof.to_csv(cache, index=False)
        if prof.empty:
            continue
        prof.to_csv(out_dir / f"heldout_train_similarity_density_{level}.csv.gz", index=False)

        ranks = [n for n in ts.DENSITY_RANKS if f"gident_rank{n}" in prof.columns]
        tbl = pd.DataFrame({
            "rank": ranks,
            "median_gident_raw": [round(prof[f"gident_rank{n}"].median(), 1) for n in ranks],
            "n_queries_raw": [int(prof[f"gident_rank{n}"].notna().sum()) for n in ranks],
            "median_gident_by_family": [round(prof[f"gident_fam{n}"].median(), 1) for n in ranks],
            "n_queries_by_family": [int(prof[f"gident_fam{n}"].notna().sum()) for n in ranks],
        })
        tbl.to_csv(out_dir / f"heldout_train_similarity_density_{level}_summary.csv", index=False)
        print(f"\n[{level}] identity of the Nth-closest training sequence / training family:")
        print(tbl.to_string(index=False))

        thr_cols = [c for c in prof.columns if c.startswith("n_fam_ge")]
        if thr_cols:
            print(f"[{level}] distinct training families within reach, per query (median / mean):")
            for c in thr_cols:
                print(f"    >={c.replace('n_fam_ge','')}% identity: "
                      f"{prof[c].median():.0f} / {prof[c].mean():.1f}")
        capped = 100 * (prof["n_hits"] >= prof["n_hits"].max()).mean()
        print(f"[{level}] queries at the --max-target-seqs cap: {capped:.1f}% "
              f"(a high value means the profile is truncated, not that the neighbourhood ends)")


def stage_report(args) -> None:
    out_dir = Path(cfg.SIMILARITY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build_table(force=args.force)
    fam = summarize(d)
    fam.to_csv(out_dir / "heldout_train_similarity_per_family.csv", index=False)

    rows = []
    for level in sorted(fam["level"].unique()):
        sub = fam[fam["level"] == level]
        v = d[d["level"] == level]
        hit = v.dropna(subset=["bitscore"])
        rows.append({
            "level": level,
            "n_families": len(sub),
            "n_sequences": len(v),
            "no_hit_pct": round(100 * v["bitscore"].isna().mean(), 2),
            "median_pident": round(sub["median_pident"].median(), 2),
            "median_gident": round(sub["median_gident"].median(), 2),
            "median_qcov": round(sub["median_qcov"].median(), 2),
            "q25_gident": round(sub["median_gident"].quantile(0.25), 2),
            "q75_gident": round(sub["median_gident"].quantile(0.75), 2),
            # How much of the query the alignment actually spans. Without these, a reader cannot
            # tell whether the identity above describes a whole protein or a short stretch of one
            # -- the first question anyone asks of a percent-identity number.
            # Denominator is queries WITH a hit: a no-hit query has no coverage to report, and
            # folding it in as a zero would conflate "matched a short stretch" with "matched
            # nothing", which the no_hit_pct column above already reports on its own.
            "pct_hits_qcov_ge80": round(100 * (hit["qcov"] >= 80).mean(), 2),
            "pct_hits_qcov_ge95": round(100 * (hit["qcov"] >= 95).mean(), 2),
            # Leakage check. The families are held out at the PDB level, so nothing here should
            # have a near-identical twin in training; these columns are what would show it if the
            # split leaked, and are worth reporting even when they come out at zero.
            "pct_ge95_identical": round(100 * (v["gident"] >= 95).mean(), 3),
            "pct_ge99_identical": round(100 * (v["gident"] >= 99).mean(), 3),
            "max_gident": round(v["gident"].max(), 2),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "heldout_train_similarity_summary.csv", index=False)
    print("\nClosest training match for held-out sequences (per-family medians):")
    print(table.to_string(index=False))

    report_density(out_dir)

    # Name the families behind any near-identical match, so a leak is traceable rather than a
    # percentage. Written only when there is something to write.
    leaks = d[d["gident"] >= 95]
    if not leaks.empty:
        cols = ["level", "qseqid", "sseqid", "pident", "gident", "qcov", "qlen", "family"]
        leaks[cols].sort_values("gident", ascending=False).to_csv(
            out_dir / "heldout_train_similarity_near_identical.csv", index=False)
        print(f"\n{len(leaks)} held-out queries are >=95% identical to a training sequence, "
              f"across {leaks['family'].nunique()} families -- see "
              f"heldout_train_similarity_near_identical.csv")

    # Novel vs seen, at both granularities.
    stats = []
    for level in sorted(fam["level"].unique()):
        key = "domain_novelty" if level == "domain" else "family_novelty"
        nov = d[d["level"] == level].groupby("family")[key].first().rename("novelty")
        merged = fam[fam["level"] == level].merge(nov, on="family", how="left")
        a = merged.loc[merged["novelty"] == "novel", "median_gident"].dropna()
        b = merged.loc[merged["novelty"] == "seen", "median_gident"].dropna()
        if len(a) and len(b):
            stats.append({"level": level, **gen.mann_whitney(a, b)})
    if stats:
        st = pd.DataFrame(stats)
        st.to_csv(out_dir / "heldout_train_similarity_novelty_stats.csv", index=False)
        print("\nNovel vs seen Pfam (Mann-Whitney on per-family medians):")
        print(st.to_string(index=False))

    plot(d, fam, out_dir)
    print(f"\nWrote tables + panel to {out_dir}")


# ======================================================================================
# CLI
# ======================================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("fasta", help="Extract sequences from the transition files.")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-seqs-per-family", type=int, default=0,
                   help="Cap on query sequences per held-out family (0 = all).")
    p.add_argument("--db-seqs-per-family", type=int, default=0,
                   help="Cap on database sequences per training family (0 = all). Capping the "
                        "database understates similarity to training, so leave it at 0.")
    p.add_argument("--no-frame-check", action="store_true",
                   help="Skip the seq1-length assertion (only if the a3m files are unavailable).")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=stage_fasta)

    p = sub.add_parser("makedb", help="Build the DIAMOND databases from the training fastas.")
    p.add_argument("--threads", type=int, default=32)
    p.set_defaults(func=stage_makedb)

    p = sub.add_parser("blastp", help="Search the held-out sequences against the training database.")
    p.add_argument("--level", choices=(*ts.LEVELS, "all"), default="all")
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--sensitivity", default="very-sensitive",
                   choices=("fast", "mid-sensitive", "sensitive", "more-sensitive",
                            "very-sensitive", "ultra-sensitive"))
    # Deep enough to see past a single training family's near-duplicates. At k=6, 96.7% of queries
    # hit the cap and 46.1% drew all six hits from one training family, so the list could not
    # distinguish an isolated nearest neighbour from a dense cluster of training data.
    p.add_argument("--max-target-seqs", type=int, default=100,
                   help="Training matches kept per query, best first (DIAMOND -k).")
    p.add_argument("--evalue", type=float, default=1e-3)
    p.add_argument("--block", type=float, default=None, help="DIAMOND -b (block size, GB).")
    p.add_argument("--index-chunks", type=int, default=None, help="DIAMOND -c.")
    p.add_argument("--no-hit-membuf", action="store_true",
                   help="Spill DIAMOND's intermediate hits to --tmpdir instead of holding them "
                        "in RAM. Only if the node has less memory than the temp files need disk.")
    # Peak memory scales with how many queries are in flight against a database block, so this is
    # the lever that keeps --hit-membuf inside the node. The domain search needs a smaller chunk:
    # its database is nearly twice the full-length one (24.7M vs 13.3M sequences) against twice the
    # queries, and in a single pass it peaked at 496 GB on a 512 GB node.
    p.add_argument("--query-chunks-full", type=int, default=2,
                   help="Split the full-length query set into this many DIAMOND runs.")
    p.add_argument("--query-chunks-domain", type=int, default=8,
                   help="Split the domain query set into this many DIAMOND runs.")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=stage_blastp)

    p = sub.add_parser("report", help="Reduce the hits, write tables and the panel.")
    p.add_argument("--force", action="store_true", help="Re-reduce the raw hit table.")
    p.set_defaults(func=stage_report)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
