"""Extended Data Fig. 8c — contact precision of Top-L categorical-Jacobian couplings.

For each test-set structure, the strongest categorical-Jacobian couplings are compared against
the true contact map and scored as precision-at-L. PEINT's encoder-decoder Jacobian is compared
against ESM's encoder-only Jacobian.

Scoring is cached one JSON per structure per arm. This distils those into a single table and
draws the panel; with the table present neither the JSONs nor a model are needed::

    python -m benchmarks.catjac_contact_precision                 # build the table, then plot
    python -m benchmarks.catjac_contact_precision --from-csv      # plot from the table alone

Each JSON holds ``metrics`` keyed ``<separation>_P@L*<fraction>`` for separations short,
medium, long and all, at L fractions 1.0, 0.5 and 0.2. The published panel is P@L*1.0.
"""

import argparse
import json
import os
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    import paper_config as cfg
    CATJAC = str(cfg.LOCAL_DATA / "catjac")
    DEFAULT_OUT = str(cfg.FIGURES_DIR)
    FIGURE_DATA = str(cfg.FIGURE_DATA_DIR)
    REPO_DATA = str(pathlib.Path(cfg.__file__).parent / "data" / "catjac")
except Exception:
    CATJAC, DEFAULT_OUT, FIGURE_DATA, REPO_DATA = "catjac", ".", ".", "."

TABLE = "catjac_contact_precision.csv"
SEPARATIONS = ("short", "medium", "long", "all")
FRACTIONS = ("1.0", "0.5", "0.2")


def read_arm(directory: str, model: str) -> list:
    """One row per (structure, separation, fraction) from an arm's per-structure JSONs.

    A structure whose scoring failed is stored as a JSON ``null``. Those are counted and
    reported rather than skipped silently, so the panel's n is always accounted for.
    """
    rows, empty = [], 0
    if not os.path.isdir(directory):
        return rows
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(directory, name)) as fh:
            d = json.load(fh)
        if not isinstance(d, dict) or not d.get("metrics"):
            empty += 1
            continue
        m = d.get("metrics", {})
        for sep in SEPARATIONS:
            for frac in FRACTIONS:
                key = f"{sep}_P@L*{frac}"
                if key in m and m[key] is not None:
                    rows.append([model, d.get("family", name), d.get("sequence_length"),
                                 sep, float(frac), float(m[key])])
    if empty:
        print(f"  {model}: {empty} structure(s) with no metrics, excluded")
    return rows


def build_table(catjac_dir: str) -> pd.DataFrame:
    rows = read_arm(os.path.join(catjac_dir, "peint"), "PEINT")
    rows += read_arm(os.path.join(catjac_dir, "esm"), "ESM")
    if not rows:
        raise SystemExit(
            f"No per-structure JSONs under {catjac_dir}/{{peint,esm}}. "
            f"Fetch the catjac role, or pass --from-csv to plot from the table."
        )
    return pd.DataFrame(rows, columns=["model", "family", "sequence_length",
                                       "separation", "l_fraction", "precision"])


def load_table(out_dir: str) -> pd.DataFrame:
    for path in (os.path.join(REPO_DATA, TABLE), os.path.join(FIGURE_DATA, TABLE),
                 os.path.join(out_dir, TABLE)):
        if os.path.exists(path):
            print(f"  reading {path}")
            return pd.read_csv(path)
    raise SystemExit(f"No {TABLE} found. Run without --from-csv to build it.")


def plot(df: pd.DataFrame, out_dir: str, separation: str, fraction: float) -> None:
    sub = df[(df.separation == separation) & (df.l_fraction == fraction)]
    fig, ax = plt.subplots(figsize=(3.2, 3.4))
    palette = {"PEINT": "#2ca02c", "ESM": "#9467bd"}     # green / purple, as published
    sns.boxplot(data=sub, x="model", y="precision", order=["PEINT", "ESM"],
                palette=palette, width=0.55, fliersize=1.5, linewidth=0.8, ax=ax)
    ax.set_ylabel(f"Contact precision  P@L*{fraction:g}  ({separation})", fontsize=9)
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", linewidth=0.4)
    for i, m in enumerate(["PEINT", "ESM"]):
        v = sub[sub.model == m].precision
        ax.text(i, 0.97, f"n={len(v)}\nmed {v.median():.3f}", ha="center", va="top", fontsize=7)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "catjac_contact_precision")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {stem}.{{pdf,png}}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catjac-dir", default=CATJAC)
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--from-csv", "--replot", dest="from_csv", action="store_true")
    ap.add_argument("--separation", default="all", choices=SEPARATIONS)
    ap.add_argument("--l-fraction", type=float, default=1.0)
    a = ap.parse_args()

    if a.from_csv:
        df = load_table(a.output_dir)
    else:
        df = build_table(a.catjac_dir)
        os.makedirs(a.output_dir, exist_ok=True)
        df.to_csv(os.path.join(a.output_dir, TABLE), index=False)
        print(f"  wrote {os.path.join(a.output_dir, TABLE)} ({len(df)} rows)")

    for model in ("PEINT", "ESM"):
        sub = df[(df.model == model) & (df.separation == a.separation)
                 & (df.l_fraction == a.l_fraction)]
        print(f"  {model:<6} n={len(sub):<4} mean {sub.precision.mean():.4f} "
              f"median {sub.precision.median():.4f}")
    plot(df, a.output_dir, a.separation, a.l_fraction)


if __name__ == "__main__":
    main()
