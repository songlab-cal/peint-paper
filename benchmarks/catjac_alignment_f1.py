"""Extended Data Fig. 8f — alignment F1 vs sequence identity, PEINT categorical Jacobian vs NW.

For each Pfam-seed domain pair, two alignments are scored against the seed alignment: one from
classical Needleman-Wunsch, one read out of PEINT's encoder-decoder categorical Jacobian. The
panel is mean F1 per percent-identity bin for each method.

The scoring is cached one file per family, so the panel redraws from that cache with no model
and no GPU::

    python -m benchmarks.catjac_alignment_f1 --from-csv       # from the shipped table
    python -m benchmarks.catjac_alignment_f1 --cache <dir>    # rebuild from the per-family cache

Each cache file has four lines::

    classical,<precision>,<recall>,<f1>
    peint,<precision>,<recall>,<f1>
    <aligned seq 1>,<aligned seq 2>
    <percent identity>

Regenerating the cache needs the model and the Pfam-A seed alignments; see the pipeline notes
that ship with the categorical-Jacobian data.
"""

import argparse
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

try:
    import paper_config as cfg
    DEFAULT_CACHE = str(cfg.LOCAL_DATA / "catjac" / "alignment_cache")
    DEFAULT_OUT = str(cfg.FIGURES_DIR)
    FIGURE_DATA = str(cfg.FIGURE_DATA_DIR)
    REPO_DATA = str(pathlib.Path(cfg.__file__).parent / "data" / "catjac")
except Exception:                                     # usable outside the repo too
    DEFAULT_CACHE, DEFAULT_OUT, FIGURE_DATA, REPO_DATA = "alignment_cache", ".", ".", "."

STYLE = {
    "linewidth": 1.5, "markeredgecolor": "k", "markeredgewidth": 1.5, "markersize": 8,
    "errorbar": ("sd", 1), "err_style": "bars",
    "err_kws": {"capsize": 2, "elinewidth": 1, "capthick": 1, "barsabove": True},
    "alpha": 1,
}


def read_cache(cache_dir: str) -> pd.DataFrame:
    """One row per (family, method) from the per-family cache files."""
    rows = []
    for name in sorted(os.listdir(cache_dir)):
        if not name.endswith(".txt"):
            continue
        with open(os.path.join(cache_dir, name)) as fh:
            lines = fh.readlines()
        if len(lines) < 4:
            continue
        s1, s2 = lines[2].strip().split(",")
        # a column that is a gap in both sequences carries no alignment information
        kept = [(c1, c2) for c1, c2 in zip(s1, s2) if not (c1 == "-" and c2 == "-")]
        n_gaps = sum(c1 == "-" for c1, _ in kept) + sum(c2 == "-" for _, c2 in kept)
        pid = float(lines[3].strip())
        fam = name[:-4]
        for line, method in ((lines[0], "classical"), (lines[1], "peint")):
            rows.append([float(line.strip().split(",")[-1]), fam, pid, n_gaps, len(kept), method])
    df = pd.DataFrame(rows, columns=["f1_score", "family", "pid", "num_gaps", "seq_len", "method"])
    df["pid_bin"] = df["pid"].round(1)
    return df


TABLE = "catjac_alignment_f1.csv"


def load_table(out_dir: str) -> pd.DataFrame:
    """Read the distilled per-family table, preferring the deposited copy."""
    for path in (os.path.join(REPO_DATA, TABLE), os.path.join(FIGURE_DATA, TABLE),
                 os.path.join(out_dir, TABLE)):
        if os.path.exists(path):
            print(f"  reading {path}")
            return pd.read_csv(path)
    raise SystemExit(
        f"No {TABLE} found. Run with --cache <dir> to rebuild it from the per-family cache."
    )


def plot(df: pd.DataFrame, out_dir: str) -> None:
    plt.figure(figsize=(5, 3))
    for method, label, colour in (("peint", "PEINT CatJac Alignment F1 Score", 0),
                                  ("classical", "NW Alignment F1 Score", 1)):
        sub = df[df.method == method]
        sns.lineplot(data=sub, x="pid_bin", y="f1_score", label=label, **STYLE)
        means = sub.groupby("pid_bin")["f1_score"].mean()
        plt.scatter(means.index, means.values, color=sns.color_palette()[colour],
                    s=40, edgecolor="black", zorder=5)
    plt.title("F1 Scores vs. %ID of Sequences")
    plt.xlabel("%ID", fontsize=10)
    plt.ylabel("Alignment F1 Score", fontsize=10)
    plt.xticks(np.arange(0, 1.1, 0.1), [f"{i*100:.0f}" for i in np.arange(0, 1.1, 0.1)], fontsize=9)
    plt.xticks(np.arange(0, 1.05, 0.05), minor=True)
    plt.yticks(np.arange(0, 1.1, 0.2), [f"{i*100:.0f}" for i in np.arange(0, 1.1, 0.2)], fontsize=9)
    plt.yticks(np.arange(0, 1.1, 0.1), minor=True)
    plt.grid(which="major", axis="y", linestyle="--", linewidth=0.4)
    plt.grid(which="major", axis="x", linestyle="--", linewidth=0.4)
    plt.legend(fontsize=8)
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        plt.savefig(os.path.join(out_dir, f"aln_f1_score.{ext}"), bbox_inches="tight", dpi=300)
    print(f"  wrote {os.path.join(out_dir, 'aln_f1_score')}.{{pdf,png}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default=DEFAULT_CACHE, help="directory of per-family cache files")
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--from-csv", "--replot", dest="from_csv", action="store_true",
                    help="plot from the distilled table instead of the per-family cache")
    a = ap.parse_args()

    if a.from_csv:
        df = load_table(a.output_dir)
        print(f"  {df.family.nunique()} families, {len(df)} scored alignments")
        for method in ("peint", "classical"):
            sub = df[df.method == method]
            print(f"    {method:<10} mean F1 {sub.f1_score.mean():.3f}  "
                  f"(low %ID <0.3: {sub[sub.pid < 0.3].f1_score.mean():.3f})")
        plot(df, a.output_dir)
        return

    if not os.path.isdir(a.cache):
        raise SystemExit(
            f"No alignment cache at {a.cache}. Fetch the catjac role, or pass --cache."
        )
    df = read_cache(a.cache)
    os.makedirs(a.output_dir, exist_ok=True)
    df.to_csv(os.path.join(a.output_dir, TABLE), index=False)
    print(f"  wrote {os.path.join(a.output_dir, TABLE)} ({len(df)} rows)")
    print(f"  {df.family.nunique()} families, {len(df)} scored alignments")
    for method in ("peint", "classical"):
        sub = df[df.method == method]
        print(f"    {method:<10} mean F1 {sub.f1_score.mean():.3f}  "
              f"(low %ID <0.3: {sub[sub.pid < 0.3].f1_score.mean():.3f})")
    plot(df, a.output_dir)


if __name__ == "__main__":
    main()
