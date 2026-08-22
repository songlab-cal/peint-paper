"""BLAST similarity of simulated leaves to the nr database.

Each simulated leaf was BLASTed against nr. Per sequence we take its **best hit** (max
percent identity); per family we take the **median** over that family's sequences; the panel
is the distribution of those family medians, one box per model. High values mean a model's
leaves look like proteins that already exist.

Two readings, because most classical-simulator leaves have no hit at all:

``hit``   -> blast_similarity.{pdf,png}. Median over sequences that returned >=1 BLAST
          hit; this is the original panel's definition. It conditions on a model-dependent
          subset: WAG keeps only ~5% of its sequences, and those are its most real-looking
          ones, so the classical models are flattered. Read it alongside the coverage table.
``all``   -> blast_similarity_with_nohit.{pdf,png}. No-hit sequences count as 0% identity.
          Honest but blunt: a family median drops to 0 as soon as half its sequences miss,
          so WAG and LG collapse onto zero.

The no-hit rate is itself a result and is written alongside the table.

    python -m benchmarks.blast_similarity                 # both variants
    python -m benchmarks.blast_similarity --variant hit
"""

import argparse
import os
import re

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from paper.model_style import model_colors
import paper_config as cfg

# BLAST tabular (outfmt 6) column order.
BLAST_COLS = ["qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
              "qstart", "qend", "sstart", "send", "evalue", "bitscore"]

# Query ids are "<family>_<seqN>_<model>.txt"; families are <pdb>_<idx>_<chain>.
QID = re.compile(r"^(?P<family>.+?)_(?P<seq>seq\d+)_(?P<model>.+)\.txt$")

# Model suffix in the query id -> display label. Order is the plotting order:
# classical simulators by increasing complexity, then the PEINT models.
MODELS = [
    ("wag", "WAG"),
    ("lg", "LG"),
    ("lg_c60", "LG+C60"),
    ("lg_s256", "LG+S256"),
    ("peint", "PEINT (ESM2)"),
    ("peint_esmc", "PEINT (ESM-C)"),
]
LABEL = dict(MODELS)
ORDER = [lbl for _, lbl in MODELS]


def _sources():
    """(blast tsv, query fasta) for each revision."""
    r1 = os.path.join(str(cfg.RESULTS_R1_DIR), "blast_sequences")
    r2 = os.path.join(str(cfg.RESULTS_R2_DIR), "blast_sequences")
    return [
        (os.path.join(r1, "revision1.csv"), os.path.join(r1, "result.txt")),
        (os.path.join(r2, "revision2.csv"), os.path.join(r2, "peint_esmc_blast.txt")),
    ]


def _parse_qid(series):
    ex = series.str.extract(QID)
    return ex["family"], ex["model"]


def best_hit_per_sequence(tsv):
    """Max percent identity per query sequence."""
    df = pd.read_csv(tsv, sep="\t", header=None, usecols=[0, 2],
                     names=["qseqid", "pident"], dtype={0: str, 2: "float32"})
    return df.groupby("qseqid", sort=False)["pident"].max()


def submitted_queries(fasta):
    """Every query id submitted, including the ones that returned no hit."""
    with open(fasta) as fh:
        return [ln[1:].strip() for ln in fh if ln.startswith(">")]


def build_table():
    """Per (model, family, sequence) best-hit identity, with no-hit rows kept as NaN."""
    frames = []
    for tsv, fasta in _sources():
        for p in (tsv, fasta):
            if not os.path.exists(p):
                raise FileNotFoundError(p)
        best = best_hit_per_sequence(tsv)
        qids = pd.Index(submitted_queries(fasta), name="qseqid")
        # Reindex onto every submitted query so no-hit sequences survive as NaN.
        s = best.reindex(qids)
        d = pd.DataFrame({"qseqid": qids, "pident": s.to_numpy()})
        d["family"], d["model_key"] = _parse_qid(d["qseqid"])
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    unknown = sorted(set(d["model_key"].dropna()) - set(LABEL))
    if unknown:
        raise ValueError(f"unmapped model suffixes in query ids: {unknown}")
    if d["family"].isna().any():
        bad = d.loc[d["family"].isna(), "qseqid"].head(3).tolist()
        raise ValueError(f"could not parse query ids, e.g. {bad}")
    d["model"] = d["model_key"].map(LABEL)
    return d[["model", "family", "qseqid", "pident"]]


def family_medians(d, variant):
    """Family-level median of the per-sequence best-hit identity."""
    x = d.copy()
    if variant == "all":
        x["pident"] = x["pident"].fillna(0.0)   # no detectable homolog
    else:
        x = x.dropna(subset=["pident"])
    return (x.groupby(["model", "family"])["pident"].median()
             .rename("median_max_pident").reset_index())


def plot(med, cov, out_stem, variant, threshold, n_fam_total):
    """Boxplot per model, annotated with what each box is actually computed from.

    Each row carries "<families> (<seq hit rate>)". Without it the panel is easy to
    misread: in the hits-only variant WAG's box rests on 163 of 542 families and 5% of its
    sequences, so its position reflects its luckiest sequences, not its typical ones.
    """
    colors = model_colors()
    present = [m for m in ORDER if m in set(med["model"])]
    data = [med.loc[med["model"] == m, "median_max_pident"].to_numpy() for m in present]

    fig, ax = plt.subplots(figsize=(4.3, 2.4))
    bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.62,
                    boxprops=dict(linewidth=0.5), whiskerprops=dict(linewidth=0.5),
                    capprops=dict(linewidth=0.5),
                    medianprops=dict(color="black", linewidth=0.8),
                    flierprops=dict(marker="x", markersize=2.5, markeredgewidth=0.4,
                                    markeredgecolor="0.35"))
    for patch, m in zip(bp["boxes"], present):
        patch.set_facecolor(colors[m])
    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", linewidth=0.8)

    ax.set_yticks(range(1, len(present) + 1), labels=present, fontsize=8)
    ax.invert_yaxis()                       # first model at the top, as in the original
    ax.set_xlim(0, 100)
    ax.set_xlabel("Median Max % Identity (BLAST nr)", fontsize=9)

    # Right-hand column: how much data each box rests on. The denominator is every family
    # that was BLASTed, not just those surviving in this variant.
    ax.text(103, 0.30, "families   seqs\n(of %d)   w/ hit" % n_fam_total,
            fontsize=5.6, color="0.35", va="center", ha="left", clip_on=False)
    for i, m in enumerate(present, start=1):
        n_fam = len(data[i - 1])
        pct = float(cov.loc[m, "with_hits"]) / float(cov.loc[m, "submitted"]) * 100
        weak = (variant == "hit") and n_fam < 0.5 * n_fam_total
        ax.text(103, i, f"{n_fam:>4d}      {pct:4.0f}%", fontsize=6.2, va="center",
                ha="left", clip_on=False,
                color="#b2182b" if weak else "0.25")
    sns.despine(ax=ax, top=True, right=True)
    for s in ax.spines.values():
        s.set_linewidth(0.5)
    ax.tick_params(width=0.5, length=2, labelsize=8)

    out_dir = str(cfg.FIGURES_DIR)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"{out_stem}.{ext}"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir}/{out_stem}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", choices=("all", "hit", "both"), default="both")
    ap.add_argument("--threshold", type=float, default=50.0,
                    help="Red reference line; pass a negative value to omit it.")
    args = ap.parse_args()

    d = build_table()
    out_dir = str(cfg.FIGURES_DIR)
    os.makedirs(out_dir, exist_ok=True)

    # The no-hit rate is a headline number in its own right.
    cov = (d.assign(hit=d["pident"].notna())
             .groupby("model")["hit"].agg(submitted="size", with_hits="sum").reindex(ORDER))
    cov["no_hit_pct"] = (100 * (1 - cov["with_hits"] / cov["submitted"])).round(1)
    cov.to_csv(os.path.join(out_dir, "blast_similarity_coverage.csv"))
    print("BLAST hit coverage per model:")
    print(cov.to_string())

    n_fam_all = d["family"].nunique()
    thr = None if args.threshold is not None and args.threshold < 0 else args.threshold
    for variant in (("hit", "all") if args.variant == "both" else (args.variant,)):
        med = family_medians(d, variant)
        stem = "blast_similarity" if variant == "hit" else "blast_similarity_with_nohit"
        med.to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
        print(f"\n[{variant}] family-median max %id, median over families "
              f"({med['family'].nunique()} families):")
        print(med.groupby("model")["median_max_pident"].median().reindex(ORDER).round(1).to_string())
        plot(med, cov, stem, variant, thr, n_fam_all)


if __name__ == "__main__":
    main()
