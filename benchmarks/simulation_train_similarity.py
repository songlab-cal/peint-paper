"""How close the simulation's own sequences are to training: roots and generated leaves.

The held-out transitions are not what the evaluation actually scores. Each simulation starts from
one root sequence and generates leaves from it, and those leaves are what every downstream metric
sees. So overlap between the held-out set and training only matters to the extent it reaches the
simulation, which it can do only through the root. This searches the roots and the generated
leaves against the same training database built by ``benchmarks.heldout_train_similarity``::

    python -m benchmarks.simulation_train_similarity fasta
    python -m benchmarks.simulation_train_similarity blastp
    python -m benchmarks.simulation_train_similarity report
"""

import argparse
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from paper import transition_similarity as ts
from benchmarks.heldout_train_similarity import db_path, work_dir

# Tree tips are named seqN; internal nodes are internal-N and the root, where present, is a row
# literally named "root". Only the tips are evaluated, so only the tips are queried here.
LEAF_RE = re.compile(r"^seq\d+$")

# Matched to the ESM-IF leaf cap, so the two analyses describe the same amount of each family.
DEFAULT_LEAVES_PER_FAMILY = 60
SAMPLE_SEED = 0

# Query set -> (alignment dir relative to a results root, which root). "root" is special-cased and
# read from ROOT_SEQ_DIR rather than from any alignment.
SOURCES = {
    "simroot": None,
    "simleaf_esm2": ("r1", "peint_progressive_dir"),
    "simleaf_esmc": ("r2", "peint_progressive_dir"),
    "realleaf": ("r1", "old_sequences"),
}
LABELS = {
    "simroot": "Simulation root",
    "simleaf_esm2": "PEINT (ESM2) leaves",
    "simleaf_esmc": "PEINT (ESM-C) leaves",
    "realleaf": "Real leaves",
}


def _msa_dir(which: str, sub: str) -> Path:
    root = cfg.RESULTS_R1_DIR if which == "r1" else cfg.RESULTS_R2_DIR
    return Path(root) / "mafft_add" / sub


def fasta_path(name: str) -> Path:
    return work_dir() / "fasta" / f"sim_{name}.fasta"


def hits_path(name: str) -> Path:
    return work_dir() / "hits" / f"sim_{name}.tsv.gz"


def read_fasta(path) -> dict:
    out, cur = {}, None
    for line in ts._open_text(path):
        if line.startswith(">"):
            cur = line[1:].strip()
            out[cur] = []
        else:
            out[cur].append(line.strip())
    return {k: "".join(v) for k, v in out.items()}


def root_sequence(family: str):
    """The simulation root, straight from ROOT_SEQ_DIR.

    Read from its own directory rather than located inside the simulated alignment, because the
    two generations disagree there: the ESM-C alignments carry the root as a row named ``root``,
    while the ESM2 ones mostly do not contain it at all. The root file is the one place both
    agree, and it is the sequence the simulation actually started from.
    """
    p = Path(cfg.ROOT_SEQ_DIR) / f"{family}.txt"
    if not p.exists():
        return None
    recs = read_fasta(p)
    if not recs:
        return None
    name, seq = next(iter(recs.items()))
    return name, seq.replace(ts.GAP, "").upper()


def leaf_sequences(family: str, which: str, sub: str, max_leaves: int, seed: int):
    """Ungapped tree tips from one model's simulated alignment, minus the root if it is present."""
    p = _msa_dir(which, sub) / f"{family}.txt"
    if not p.exists():
        return []
    recs = read_fasta(p)
    root = root_sequence(family)
    root_seq = root[1] if root else None
    leaves = []
    for name, aligned in recs.items():
        if not LEAF_RE.match(name):
            continue
        plain = aligned.replace(ts.GAP, "").upper()
        # Belt and braces: a tip carrying the root's exact sequence is the root under a tree label,
        # not a generated leaf, and counting it would import the root's similarity into the leaves.
        if plain and plain != root_seq:
            leaves.append((name, plain))
    if 0 < max_leaves < len(leaves):
        rng = random.Random(f"{seed}:{family}")
        leaves = sorted(rng.sample(leaves, max_leaves))
    return leaves


def families() -> list:
    """Families with both a root and an ESM-C simulation, so every arm describes the same set."""
    have_root = {p.stem for p in Path(cfg.ROOT_SEQ_DIR).glob("*.txt")}
    have_sim = {p.stem for p in _msa_dir("r2", "peint_progressive_dir").glob("*.txt")}
    return sorted(have_root & have_sim)


def stage_fasta(args) -> None:
    fams = families()
    print(f"{len(fams)} families with a root sequence and an ESM-C simulation")
    for name, src in SOURCES.items():
        out = fasta_path(name)
        if out.exists() and not args.force:
            print(f"  {out.name} exists, skipping")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".partial")
        n_seq = n_fam = 0
        with open(tmp, "w") as fh:
            for fam in fams:
                if src is None:
                    r = root_sequence(fam)
                    recs = [(r[0], r[1])] if r else []
                else:
                    recs = leaf_sequences(fam, *src, args.leaves_per_family, SAMPLE_SEED)
                if recs:
                    n_fam += 1
                for sid, seq in recs:
                    fh.write(f">{fam}{ts.ID_SEP}{sid}\n{seq}\n")
                    n_seq += 1
        tmp.replace(out)
        print(f"  {name:<14} {n_seq:>7,} sequences from {n_fam:>4} families -> {out.name}")


def stage_blastp(args) -> None:
    for name in SOURCES:
        out = hits_path(name)
        if out.exists() and not args.force:
            print(f"  {out.name} exists, skipping")
            continue
        q = fasta_path(name)
        if not q.exists():
            raise FileNotFoundError(f"{q} -- run the `fasta` stage first")
        print(f"  blastp {name}")
        ts.blastp(db_path("full"), q, out, threads=args.threads,
                  sensitivity=args.sensitivity, max_target_seqs=args.max_target_seqs,
                  evalue=args.evalue, block=args.block, index_chunks=args.index_chunks,
                  tmpdir=str(work_dir() / "tmp"), hit_membuf=not args.no_hit_membuf)


def build_table(force: bool = False) -> pd.DataFrame:
    cache = work_dir() / "sim_per_sequence_hits.csv.gz"
    if cache.exists() and not force:
        return pd.read_csv(cache)
    frames = []
    for name in SOURCES:
        tsv, fa = hits_path(name), fasta_path(name)
        if not (tsv.exists() and fa.exists()):
            print(f"  (skipping {name}: not searched yet)")
            continue
        d = ts.hits_table(tsv, fa)
        d["source"] = name
        frames.append(d)
    if not frames:
        raise FileNotFoundError("No simulation hit tables -- run `blastp` first")
    d = pd.concat(frames, ignore_index=True)
    d.to_csv(cache, index=False)
    return d


SOURCE_ORDER = ["simroot", "realleaf", "simleaf_esmc", "simleaf_esm2"]
# Short forms for axis ticks; the full LABELS collide once four sit side by side.
SHORT_LABELS = {"simroot": "Root", "realleaf": "Real",
                "simleaf_esmc": "PEINT\n(ESM-C)", "simleaf_esm2": "PEINT\n(ESM2)"}

ROOT_GRADES = [
    ("no homolog", lambda v: v <= 0),
    (">0-50%", lambda v: (v > 0) & (v < 50)),
    ("50-80%", lambda v: (v >= 50) & (v < 80)),
    (">=80%", lambda v: v >= 80),
]


def _source_colors():
    """Paper palette, so these arms read the same here as in every other panel."""
    from paper.model_style import model_colors
    mc = model_colors()
    return {"simroot": "#4d4d4d", "realleaf": mc["Real"],
            "simleaf_esmc": mc["PEINT (ESM-C)"], "simleaf_esm2": mc["PEINT (ESM2)"]}


def _box(ax, data, positions, facecolors, width=0.7):
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=width,
                    showfliers=False, boxprops=dict(linewidth=0.5),
                    whiskerprops=dict(linewidth=0.5), capprops=dict(linewidth=0.5),
                    medianprops=dict(color="black", linewidth=1.0))
    for patch, c in zip(bp["boxes"], facecolors):
        patch.set_facecolor(c)
    return bp


def plot_similarity(d: pd.DataFrame, out_dir: Path) -> None:
    """Closest-match-to-training identity for the root and for each model's generated leaves.

    The root is one sequence per family and the leaves are many, so the two are shown together on
    purpose: the root is the only route by which the held-out set's overlap with training can
    reach a simulation at all, and the leaf boxes say how much of it survives generation.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style

    _set_publication_style()
    # Derived here rather than relying on the caller having added it: a no-hit sequence belongs at
    # 0% identity, and a function that silently needs its caller to have done that breaks the
    # moment the figure is regenerated on its own.
    d = d.assign(pident0=d["pident"].fillna(0.0))
    colors = _source_colors()
    present = [n for n in SOURCE_ORDER if n in set(d["source"])]

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.8))
    for ax, per_family in zip(axes, (False, True)):
        data, ticks = [], []
        for name in present:
            v = d[d["source"] == name]
            x = (v.groupby("family")["pident0"].median().to_numpy() if per_family
                 else v["pident0"].to_numpy())
            data.append(x)
            ticks.append(f"{SHORT_LABELS[name]}\nn={x.size:,}")
        _box(ax, data, list(range(1, len(data) + 1)), [colors[n] for n in present])
        ax.set_xticks(range(1, len(data) + 1))
        ax.set_xticklabels(ticks, fontsize=7.5)
        ax.set_ylim(0, 100)
        ax.set_ylabel("% identity to closest training sequence", fontsize=9)
        ax.set_title("per sequence" if not per_family else "per family (median)", fontsize=9)
        sns.despine(ax=ax)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"simulation_train_similarity.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  wrote {out_dir}/simulation_train_similarity.{{pdf,png}}")


def plot_max_identity(d: pd.DataFrame, out_dir: Path) -> None:
    """Per family, the single closest any of its sequences gets to the training set.

    The max, not the median, is the statistic a leakage argument turns on: one leaf that is
    near-identical to a training sequence matters even if the other five hundred are not. Taken
    over ~60 leaves per family it is the most favourable reading available to the objection.

    The right panel grades families by how close their ROOT sat to training, which is the only
    route by which the held-out set's overlap can reach a simulation. If the leaf boxes stay flat
    as the root grade rises, the root's similarity is not being passed on.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from paper.plot_style import _set_publication_style

    _set_publication_style()
    colors = _source_colors()
    d = d.assign(p=d["pident"].fillna(0.0))
    mx = d.groupby(["source", "family"])["p"].max().unstack("source")
    present = [n for n in SOURCE_ORDER if n in mx.columns]

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9),
                             gridspec_kw={"width_ratios": [1, 1.7]})

    ax = axes[0]
    data = [mx[n].dropna().to_numpy() for n in present]
    _box(ax, data, list(range(1, len(data) + 1)), [colors[n] for n in present])
    ax.set_xticks(range(1, len(data) + 1))
    ax.set_xticklabels([f"{SHORT_LABELS[n]}\nn={len(mx[n].dropna())}" for n in present],
                       fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Max % identity to training\n(over the family's sequences)", fontsize=9)
    sns.despine(ax=ax)

    ax = axes[1]
    leaf_sources = [n for n in ("realleaf", "simleaf_esmc", "simleaf_esm2") if n in mx.columns]
    centres, ticks, data, facecolors, positions = [], [], [], [], []
    pos = 1.0
    for gname, rule in ROOT_GRADES:
        fams = mx.index[rule(mx["simroot"])] if "simroot" in mx.columns else []
        if not len(fams):
            continue
        for name in leaf_sources:
            vals = mx.loc[[f for f in fams if f in mx.index], name].dropna().to_numpy()
            if not vals.size:
                continue
            positions.append(pos)
            data.append(vals)
            facecolors.append(colors[name])
            pos += 1
        centres.append(pos - (len(leaf_sources) + 1) / 2)
        ticks.append(f"{gname}\n{len(fams)} families")
        pos += 1
    if data:
        _box(ax, data, positions, facecolors, width=0.78)
        ax.set_xticks(centres)
        ax.set_xticklabels(ticks, fontsize=8)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Identity of the simulation root to training", fontsize=9)
    ax.set_ylabel("Max % identity to training\n(over the family's leaves)", fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=colors[n]) for n in leaf_sources]
    ax.legend(handles, [LABELS[n] for n in leaf_sources], fontsize=7.5, frameon=False,
              loc="lower right")
    sns.despine(ax=ax)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"simulation_max_identity.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    mx.to_csv(out_dir / "simulation_max_identity_per_family.csv")
    print(f"  wrote {out_dir}/simulation_max_identity.{{pdf,png}}")
    print("\n  per-family MAX % identity to training:")
    for name in present:
        v = mx[name].dropna()
        print(f"    {LABELS[name]:<22} median={v.median():5.1f}  "
              + "  ".join(f">={t}%: {100 * (v >= t).mean():4.1f}%" for t in (80, 90, 95)))


def stage_report(args) -> None:
    out_dir = Path(cfg.SIMILARITY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    d = build_table(force=args.force)
    d["pident0"] = d["pident"].fillna(0.0)

    rows = []
    for name in SOURCES:
        v = d[d["source"] == name]
        if v.empty:
            continue
        per_fam = v.groupby("family")["pident0"].median()
        rows.append({
            "source": LABELS[name], "n_sequences": len(v), "n_families": v["family"].nunique(),
            "pct_no_hit": round(100 * v["pident"].isna().mean(), 2),
            "median_pident": round(per_fam.median(), 2),
            "mean_pident": round(per_fam.mean(), 2),
            "pct_seqs_ge95": round(100 * (v["pident0"] >= 95).mean(), 2),
            "pct_families_any_ge95": round(
                100 * v.groupby("family")["pident0"].max().ge(95).mean(), 2),
        })
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "simulation_train_similarity.csv", index=False)
    print("\nCloseness of the simulation's own sequences to the training set:")
    print(table.to_string(index=False))

    # Does a root close to training hand that closeness to its leaves? Paired within family.
    wide = (d.groupby(["family", "source"])["pident0"].median().unstack("source"))
    if "simroot" in wide.columns:
        wide.to_csv(out_dir / "simulation_train_similarity_per_family.csv")
        print("\nPer-family correlation of root identity with leaf identity:")
        for name in ("simleaf_esm2", "simleaf_esmc", "realleaf"):
            if name not in wide.columns:
                continue
            g = wide[["simroot", name]].dropna()
            if len(g) < 5:
                continue
            print(f"  {LABELS[name]:<22} pearson r={g['simroot'].corr(g[name]):.3f} "
                  f"spearman={g['simroot'].corr(g[name], method='spearman'):.3f} (n={len(g)})")
        near = wide[wide["simroot"] >= 95]
        far = wide[wide["simroot"] < 30]
        print(f"\n  families whose ROOT is >=95% identical to training: {len(near)}")
        print(f"  families whose ROOT is <30%: {len(far)}")
        for name in ("simleaf_esm2", "simleaf_esmc", "realleaf"):
            if name in wide.columns and len(near) and len(far):
                print(f"    {LABELS[name]:<22} leaf identity: root-near={near[name].median():.1f}%  "
                      f"root-far={far[name].median():.1f}%")
    plot_similarity(d, out_dir)
    plot_max_identity(d, out_dir)
    print(f"\nWrote tables + panel to {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="stage", required=True)
    p = sub.add_parser("fasta")
    p.add_argument("--leaves-per-family", type=int, default=DEFAULT_LEAVES_PER_FAMILY)
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=stage_fasta)
    p = sub.add_parser("blastp")
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--sensitivity", default="very-sensitive")
    p.add_argument("--max-target-seqs", type=int, default=100)
    p.add_argument("--evalue", type=float, default=1e-3)
    p.add_argument("--block", type=float, default=2.0)
    p.add_argument("--index-chunks", type=int, default=1)
    p.add_argument("--no-hit-membuf", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=stage_blastp)
    p = sub.add_parser("report")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=stage_report)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
