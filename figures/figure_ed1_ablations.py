"""Extended Data Fig. 1b, 1c and Supplementary Fig. 1 — per-site likelihood across architectural
variants, ablations and decoder depths.

**ED 1b** is mean per-site transition likelihood on held-out cherries against transition time, with
an inset over the short-branch region. **ED 1c** is the same quantity at t ~= 0.1 across training
steps. Both come from one table, a run x step x time-bin sweep over the seven model variants shown::

    figure_data/ed1/master_sweep_553fam.csv     run, step, backbone, t_bin, likelihood, n_sites

**Supp. Fig. 1** is the same pair of views for PEINT with one to five decoder layers (a: t ~= 0.1
across training; b, c: likelihood against transition time at step 60,000 and its difference from
the five-layer model, drawn as one image), from the sweep's own table::

    figure_data/decoder_depth/decoder_depth_553fam.csv

The mean per-site likelihood is ``exp(mean per-site log-likelihood)`` over the alignment columns
where both the aligned source and the aligned target hold a residue, as in Fig. 2a.

The plotting is not reimplemented here. `figure_data/ed1/` ships the sweep's own
`make_report.py`, and this module runs that script against the released table so the panels come
out of the same code that made the published ones. It emits several legend and inset variants;
the published pair is the family-grouped one with the legend inside the axes, which is what gets
copied into `figures/output/`::

    python -m figures.figure_ed1_ablations                          # ED Fig. 1b, 1c
    python -m figures.figure_ed1_ablations --figset decoder_depth   # Supp. Fig. 1

The sweep itself is not rerunnable from this release: it spans fifteen checkpoints per variant,
and those checkpoints are not deposited. The table is the reproducible starting point.
"""

import argparse
import os
import pathlib
import shutil
import subprocess
import sys

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import paper_config as cfg

SCRIPT = "make_report.py"   # always the one in figure_data/ed1/

# Per figure: the table's folder and name, make_report.py's PEINT_FIGSET and output suffix, the
# subfolder of evaluations/plots/ it writes to, and the published panels with the names they are
# given in figures/output.
FIGSETS = {
    "ed1": dict(
        subdir="ed1", table="master_sweep_553fam.csv", figset="ed1", suffix="_ed1", plots="",
        panels={
            "likelihood_curves_latest_camready_legendin_grouped": "ed1b_likelihood_vs_time",
            "likelihood_vs_step_t0.1_camready_legendin_grouped": "ed1c_likelihood_vs_step_t0.1",
        }),
    "decoder_depth": dict(
        subdir="decoder_depth", table="decoder_depth_553fam.csv", figset="decoder_depth",
        suffix="_dd", plots="decoder_depth/wip",
        panels={
            "likelihood_vs_step_t0.1_decoder_depth_legendin": "supp1a_decoder_depth_vs_step_t0.1",
            "likelihood_curves_with_gap_decoder_depth": "supp1bc_decoder_depth_vs_time",
        }),
}
T01 = 0.1036          # the t bin the 1c panel reads, as in the sweep's own script


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figset", choices=sorted(FIGSETS), default="ed1",
                    help="ed1: Extended Data Fig. 1b, 1c (default); "
                         "decoder_depth: Supplementary Fig. 1")
    ap.add_argument("--figure-data", default=None,
                    help="figure_data directory holding ed1/ and decoder_depth/ "
                         "(default: the configured figure_data)")
    args = ap.parse_args()
    fs = FIGSETS[args.figset]

    root = pathlib.Path(args.figure_data) if args.figure_data else cfg.FIGURE_DATA_DIR
    ed1 = root / "ed1"
    table, script = root / fs["subdir"] / fs["table"], ed1 / SCRIPT
    for p in (table, script):
        if not p.exists():
            raise SystemExit(
                f"{p} not found. Fetch the figure_data tier:\n"
                f"  python scripts/fetch_local_data.py --tier figure_data"
            )

    df = pd.read_csv(table)
    steps = sorted(df.step.unique())
    print(f"  reading {table}")
    print(f"  {len(df)} rows: {df.run.nunique()} variants x {len(steps)} steps "
          f"({steps[0]}-{steps[-1]}) x {df.t_bin.nunique()} time bins")

    env = {**os.environ, "PEINT_MASTER": str(table.resolve()), "PEINT_FIGSET": fs["figset"],
           "PEINT_SUFFIX": fs["suffix"], "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, SCRIPT], cwd=ed1, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{SCRIPT} failed (exit {r.returncode})")

    plots = ed1 / "evaluations" / "plots" / fs["plots"]
    out = cfg.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    for stem, name in fs["panels"].items():
        for ext in ("pdf", "png"):
            src = plots / f"{stem}{fs['suffix']}.{ext}"
            if not src.exists():
                raise SystemExit(f"{SCRIPT} did not write {src}")
            shutil.copyfile(src, out / f"{name}.{ext}")
        print(f"  wrote {out / name}.{{pdf,png}}")

    # The ranking the caption describes, at the last checkpoint.
    last = df[(df.step == steps[-1]) & (df.t_bin.sub(T01).abs() < 1e-3)]
    if not last.empty:
        print(f"\n  mean per-site likelihood at t = {T01}, step {steps[-1]}:")
        for row in last.sort_values("likelihood", ascending=False).itertuples():
            print(f"    {row.run:<20s} {row.likelihood:.4f}")


if __name__ == "__main__":
    main()
