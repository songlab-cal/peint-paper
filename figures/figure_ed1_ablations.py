"""Extended Data Fig. 1b, 1c — per-site likelihood across architectural variants and ablations.

**1b** is mean per-site transition likelihood on held-out cherries against transition time, with
an inset over the short-branch region. **1c** is the same quantity at t ~= 0.1 across training
steps. Both come from one table, a run x step x time-bin sweep over ten model variants::

    figure_data/ed1/master_sweep_553fam.csv     run, step, backbone, t_bin, likelihood, n_sites

The plotting is not reimplemented here. `figure_data/ed1/` ships the sweep's own
`make_report.py`, and this module runs that script against the released table so the panels come
out of the same code that made the published ones. It emits several legend and inset variants;
the published pair is the family-grouped one with the legend inside the axes, which is what gets
copied into `figures/output/`::

    python -m figures.figure_ed1_ablations

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

TABLE = "master_sweep_553fam.csv"
SCRIPT = "make_report.py"
SUFFIX = "_ed1"

# The published pair, and the names they are given in figures/output.
PANELS = {
    "likelihood_curves_latest_camready_legendin_grouped": "ed1b_likelihood_vs_time",
    "likelihood_vs_step_t0.1_camready_legendin_grouped": "ed1c_likelihood_vs_step_t0.1",
}
T01 = 0.1036          # the t bin the 1c panel reads, as in the sweep's own script


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ed1-dir", default=None,
                    help="directory holding the sweep table and make_report.py "
                         "(default: <figure_data>/ed1)")
    args = ap.parse_args()

    ed1 = pathlib.Path(args.ed1_dir) if args.ed1_dir else cfg.FIGURE_DATA_DIR / "ed1"
    table, script = ed1 / TABLE, ed1 / SCRIPT
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

    env = {**os.environ, "PEINT_MASTER": str(table.resolve()),
           "PEINT_SUFFIX": SUFFIX, "MPLBACKEND": "Agg"}
    r = subprocess.run([sys.executable, SCRIPT], cwd=ed1, env=env,
                       capture_output=True, text=True)
    if r.returncode:
        sys.stderr.write(r.stderr)
        raise SystemExit(f"{SCRIPT} failed (exit {r.returncode})")

    plots = ed1 / "evaluations" / "plots"
    out = cfg.FIGURES_DIR
    out.mkdir(parents=True, exist_ok=True)
    for stem, name in PANELS.items():
        for ext in ("pdf", "png"):
            src = plots / f"{stem}{SUFFIX}.{ext}"
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
