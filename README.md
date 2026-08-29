# peint-paper

Figures and benchmarks for the PEINT paper. Depends on the model library
[`peint`](../peint) (imported as the `protevo` package) via an editable install.

## Setup

**`installation.md` is the full, tested recipe** — the ordered setup checklist (both repos,
both environments, the data and checkpoints, where to unpack and symlink them), the JAX/PyTorch
install order, the external tools, and how the two environments hand off to each other. The
short version:

```bash
# 1. Install the model library (editable) + this repo's benchmarking extras
pip install -e ../peint
pip install -e .
```

Two environments are used, because the folding/alignment stack and the ESM-C stack do not
coexist cleanly:

| env | panels | needs |
|---|---|---|
| `peint-esmc` | most panels, and **all PEINT-ESM-C generation** | torch + CUDA, `logomaker`, and a `transformers` new enough to know the `esmc` architecture |
| `protevo-env` | the PCP panels, the two Historian indel panels, the 3Di panel, and **all OmegaFold folding** | `cherryml`, `ete3`, `sentencepiece` (ProstT5), MAFFT / IQ-TREE-AliSim / `omegafold` on `PATH` |

Why two environments. The benchmarks span three stacks that do not pin cleanly together:
JAX (AF2Rank via ColabDesign), PyTorch (PEINT, OmegaFold, ESM-C), and the older
`transformers` that ProstT5 was pinned against. Separating the model library from this repo
keeps that mess out of `peint` itself; what is left here is one genuine conflict:

| | `peint-esmc` | `protevo-env` |
|---|---|---|
| transformers | 4.57.6 — knows the `esmc` architecture | 4.45.2 — raises `model type 'esmc' not recognized` |
| OmegaFold | — | installed |
| JAX + PyTorch | torch only | both, side by side |

So ESM-C generation only runs in the first, and folding only in the second. `figure2_simulation`
needs both and is therefore run twice: first in `peint-esmc` with `--skip-structures` to generate
and cache the sequences, then in `protevo-env` without the flag, where those simulations are
cache hits (no ESM-C model is ever constructed there) and only the folding runs.

Point the renderer at both:

```bash
export PEINT_PAPER_PY_ESMC=/path/to/envs/peint-esmc/bin/python
export PEINT_PAPER_PY_PROTEVO=/path/to/envs/protevo-env/bin/python
```

Historian is **not** needed for any released panel — only to regenerate the ancestral
reconstructions themselves (`benchmarks/historian_esmc_progressive.py`), for which
`scripts/build_historian.sh` builds the submodule.

## Data

Everything the figures read resolves under a single root, `local_data/`, laid out so that
unpacking the published dataset into it is the whole setup step:

```
local_data/
  figure_data/   per-panel tables — enough to re-render every panel, no models needed
  vep/           per-run ProteinGym Spearman tables (git-tracked, ships with the code)
  r1/            revision 1: classical baselines, PEINT-ESM2, real
  r2/            revision 2: the ESM-C rerun (its own mafft frame)
  sim/           trees, root sequences, empirical + simulated MSAs
  annotations/   Pfam / SCOPe / ECOD labels for the generalization analysis
  splits/        the 14,498 train / 553 held-out family lists
  derived/       everything the benchmarks WRITE, incl. computation caches
```

### One directory, if you want it

Unpack the deposit into `local_data/` and set one variable:

```bash
export PEINT_PAPER_LOCAL_DATA_ONLY=1        # optional, but makes it deterministic
scripts/render_panels.py
```

By default each input resolves to the authoritative source tree when that exists and to
`local_data/` otherwise, so the same config serves the machine that produced the data and
someone who just downloaded it. On a machine that has both, "otherwise" never fires — which
also means you cannot tell whether the deposit is self-sufficient. `PEINT_PAPER_LOCAL_DATA_ONLY=1`
makes `local_data/` win outright: every input, including the model checkpoints, then comes
from that one directory, and anything missing fails at its deposit path instead of quietly
resolving to a local tree the deposit does not contain. Verified: with it set, no config path
escapes `local_data/` or the repo.

`PEINT_PAPER_LOCAL_DATA` moves the whole tree somewhere else if you would rather not put it
under the checkout.

`local_data/` is git-ignored, with one exception: `local_data/vep/` is 250 KB of per-run
ProteinGym Spearman tables that the three VEP panels cannot render without, so it is tracked
and the deposit neither carries nor overwrites it. Override any location with the
`PEINT_PAPER_*` environment variables listed in `paper_config.py`; `LOCAL_DATA` moves the
whole tree at once.

### The manifest

`data/MANIFEST.toml` is the inventory: one entry per logical role, recording where it lives,
where it must land, which archive carries it, a cheap probe file, measured sizes, and the
panels that need it. Four scripts read it, so the list cannot drift:

```bash
scripts/check_local_data.py                      # what is present; what supplies the rest
scripts/check_local_data.py --panels pcp_panels  # scoped to the figures you want
scripts/link_local_data.sh --apply               # symlink the roles, on a machine that has them
scripts/stage_shared_data.sh --plan              # rsync them to another account
scripts/build_archives.sh --root <staged> --apply  # tar them for the deposit
scripts/fetch_local_data.py --tier full          # download and unpack them again
```

## Reproducing the figures

`REPRODUCING.md` is the full account: what redraws from ~19 MB, what recomputes from ~11 GB,
which figures need a full rerun from a checkpoint, and how each shipped intermediate traces
back to the script that produced it.

## Rendering the figures

`scripts/render_panels.py` is the whitelist of panels that appear in the paper.

```bash
scripts/render_panels.py --list        # what exists, in which env, and how long it takes
scripts/render_panels.py --check       # which outputs are missing
scripts/render_panels.py --dry-run     # print the exact commands
scripts/render_panels.py --group main
scripts/render_panels.py --only vep_params
```

Prerequisites are resolved automatically (the novel-vs-seen OmegaFold panel re-uses the
table the pLDDT ECDF writes, so asking for the former runs the latter first).

Anything not on that list is exploratory work kept for provenance, not part of the paper.

## Conventions

* **Model colors** come from `paper.model_style` and nowhere else. It resolves the seaborn
  `deep` palette explicitly, so a model is the same color regardless of which script drew
  it. Add a model there, not in a figure module.
* **PDF text is Type42** (`pdf.fonttype = 42`) so labels stay editable in Illustrator.
  Check with `grep -ac FontFile2 file.pdf`.
* Scripts are run as modules from the repo root (`python -m figures.figure3_conservation`),
  and none of them depend on the working directory.

## External tools

| Tool | Purpose | How it's provided |
|------|---------|-------------------|
| IQ-TREE 2 / AliSim | classical-model simulation | provided by `peint` (`iqtree2` submodule) |
| MAFFT | alignment | expected on `PATH` |
| Historian | ancestral reconstruction (producers only) | submodule + `scripts/build_historian.sh` |
| OmegaFold | structure prediction (Figure 2 pLDDT + the structure benchmarks) | installed in `protevo-env`; weights cached at `~/.cache/omegafold_ckpt` |
| AF2Rank / ColabDesign | template-based structure scoring (producers only) | see `pyproject.toml`'s `folding` extra |
