# peint-paper

Figures and benchmarks for the PEINT paper. The models live in the separate
[`peint`](../peint) repo and are imported here as `protevo`.

**Start with [`installation.md`](installation.md)** — clone, two conda environments, data,
and the three tiers of reproduction (replot / recompute / rerun from a checkpoint).

## Rendering figures

`scripts/render_panels.py` is the whitelist of panels that appear in the paper. Anything not
on it is exploratory work kept for provenance.

```bash
scripts/render_panels.py --list        # what exists, in which env, how long it takes
scripts/render_panels.py --check       # which outputs are missing
scripts/render_panels.py --dry-run     # print the exact commands
scripts/render_panels.py --group main
scripts/render_panels.py --only vep_params
```

Each panel is run with the interpreter its environment needs, so set `PEINT_PAPER_PY_ESMC`
and `PEINT_PAPER_PY_PROTEVO` first. Prerequisites resolve automatically — asking for the
novel-vs-seen OmegaFold panel runs the pLDDT ECDF that writes its input.

## Data

Everything the figures read resolves under `local_data/`:

```
local_data/
  figure_data/   per-panel tables — enough to re-render every panel, no models needed
  vep/           per-run ProteinGym Spearman tables (git-tracked, ships with the code)
  r1/  r2/       revision 1 (classical + PEINT-ESM2) and revision 2 (the ESM-C rerun)
  sim/           trees, root sequences, empirical + simulated MSAs
  peint/         model checkpoints and held-out transitions
  annotations/   Pfam / SCOPe / ECOD labels
  splits/        the 14,498 train / 553 held-out family lists
  derived/       everything the benchmarks WRITE, incl. computation caches
```

Unpacking the deposit into it is the whole setup step. `PEINT_PAPER_LOCAL_DATA` moves the
tree elsewhere; `PEINT_PAPER_LOCAL_DATA_ONLY=1` makes it the only source, which is how to
check the deposit is self-sufficient on a machine that also has the original trees.

`local_data/` is gitignored except `local_data/vep/`, which the VEP panels need and the
deposit does not carry.

`data/MANIFEST.toml` is the inventory — one entry per role, with its path, archive, size and
the panels that need it. Four scripts read it, so it cannot drift:

```bash
scripts/check_local_data.py                      # what is present; what supplies the rest
scripts/link_local_data.sh --apply               # symlink roles, on a machine that has them
scripts/build_archives.sh --root <staged> --apply  # tar them for the deposit
scripts/fetch_local_data.py --tier full          # download and unpack them again
```

`REPRODUCING.md` covers what redraws, what recomputes, and which figures need a full rerun.

### Caching

Expensive steps — simulations, rate-matrix fits, per-site likelihoods, folding — are cached to
disk by `protevo.caching` (and `cherryml.caching` for the classical models). A cached call is
keyed on a hash of its arguments, **including absolute input paths**, so:

- rerunning the same command is instant the second time;
- a cache copied from another machine never hits, which is why none is distributed;
- moving your `local_data/` invalidates everything derived from it.

Caches live under `local_data/derived/`. Deleting one costs time, never correctness.

## Conventions

- **Model colors** come from `paper.model_style` and nowhere else, so a model is the same
  color in every figure. Add a model there, not in a figure module.
- **PDF text is Type42** (`pdf.fonttype = 42`) so labels stay editable in Illustrator.
  Check with `grep -ac FontFile2 file.pdf`.
- Scripts run as modules from the repo root (`python -m figures.figure3_conservation`).
