# PEINT — figure notebooks

One notebook per manuscript figure, reproducing that figure's panels from the released data
and code. Each cell runs a panel's producer and displays its output.

## Levels

Each cell is marked with the level it runs at.

| level | reads | writes the panel from | cost |
|---|---|---|---|
| 1 | `local_data/figure_data/` | a released summary table | seconds, CPU |
| 2 | `local_data/r1/`, `r2/`, `sim/` | released intermediate results | minutes to an hour |
| 3 | model checkpoints + raw data | regenerated intermediates | hours to days, GPU |

Every notebook runs at level 1 by default. Level-2 and level-3 commands are given in the
cells, commented out.

## Why level 1 is the default, and what the embedded outputs are

These notebooks are shipped **with their outputs embedded**: the panels and printed values you
see are from an actual run, not typed in. Re-running any cell regenerates them.

Level 1 is the default because of size. The data release is already ~26 GB compressed. Some
panels are backed by intermediates far larger than the panel needs — Extended Data Fig. 8b, for
one, reduces a 32 MB Jacobian tensor per structure (25 GB across the set) to a 67 KB matrix.
Depositing the raw intermediates for every panel would multiply the record's size for no gain,
so what ships is the reduced quantity each panel actually plots.

Every cell that redraws also documents how to recompute what it redraws from, and the producers
take the flags to do it. Where a reduction step exists, it is the released implementation:
Fig. 8b's, for instance, was checked against the original and is bitwise identical.

So: level 1 shows the released figures follow from the released numbers, and the recompute path
is there for anyone who wants to go further back.

## Setup

### 1. Repositories

```bash
git clone https://github.com/<org>/peint-paper.git
git clone https://github.com/<org>/peint.git
cd peint-paper
```

Expected layout — keep them as siblings, or set `PEINT_PAPER_PEINT_REPO`:

```
<anywhere>/
├── peint-paper/     notebooks/, figures/, benchmarks/, scripts/, local_data/
└── peint/           the model library, imported as `peint`
```

The notebooks find the repository root by looking for `paper_config.py`. Launch Jupyter from
either directory; no environment variables are needed for level 1.

### 2. Environment

```bash
conda create -p ./envs/peint-paper python=3.10 -y
conda activate ./envs/peint-paper

pip install -e ../peint      # model library
pip install -e .             # this repository, plotting and analysis
pip install jupyterlab
```

This covers every level-1 cell.

Structure panels (Fig. 3f, 3g; ED Fig. 4b, 5d, 5e) also need the folding stack. Two of its
packages are not on PyPI:

```bash
pip install -e ".[folding]"
pip install -q git+https://github.com/sokrypton/ColabDesign.git@v1.1.3 --no-deps
pip install --no-deps -e <OmegaFold checkout>
```

Optional, for display only:

```bash
conda install -c conda-forge poppler   # Fig. 4 panels are PDF; this shows them inline
pip install PyQt5                      # Fig. 4a/4b render a tree through Qt
```

Headless machines need `QT_QPA_PLATFORM=offscreen`; the Figure 4 notebook sets it.

Level 3 uses two further environments — one for PEINT, one for the ESM-C variant — because
JAX and PyTorch need different CUDA builds and ESM-C needs a `transformers` fork. Recipes and
pinned versions are in `../REPRODUCING_TUTORIAL.md`.

### 3. Data

Everything downloads into `local_data/`. Paths in the notebooks are relative to it.

```bash
python scripts/fetch_local_data.py --list                # archives, sizes, contents
python scripts/fetch_local_data.py --tier figure_data    # 20 MB, covers all of level 1
```

Level 2 needs specific archives:

```bash
python scripts/fetch_local_data.py --tier full --archives r1 r2 sim aux
```

| panel | archives | unpacked |
|---|---|---|
| Fig. 3b, 3c, 3f, 3g · ED Fig. 3a–d, 4b, 9 | `r1` `r2` `sim` `aux` | 46 GB |
| Fig. 3d, 3e | `r2` | 25 GB |
| Fig. 2a | `peint_transitions_aligned` `peint_transitions_unaligned` | 26 GB |
| ED Fig. 5d, 5e | `r1_omegafold` `r2` | 71 GB |

`--tier full` fetches all of them: ~22 GB compressed, ~151 GB unpacked. Archives are
checksummed on download; `--verify-only` re-checks an existing copy.

Level 3 also needs `peint_checkpoints` (6.7 GB).

Resulting layout:

```
peint-paper/
├── local_data/
│   ├── figure_data/     summary tables            level 1
│   ├── r1/ r2/ sim/     intermediate results      level 2
│   └── peint/           model checkpoints         level 3
├── figures/output/      panels are written here
└── notebooks/
```

### 4. Verify

```bash
python scripts/check_local_data.py     # lists roles present and missing
```

## Coverage

| notebook | panels | default level | notes |
|---|---|---|---|
| `Figure2.ipynb` | 2a, 2b, 2c | 1 | 2c regenerates at level 3 |
| `Figure3.ipynb` | 3b, 3c, 3d, 3e, 3f, 3g | 1–2 | |
| `Figure4.ipynb` | 4a, 4b, 4c, 4e, 4f | 1 | inputs are in the repository; 4d needs a parser not included here |
| `Figure5.ipynb` | 5a, 5b, 5c, 5d | 1 | |
| `ExtendedData2.ipynb` | 2a–2g | — | ESM-C arm of Figures 2 and 3; produced by those commands |
| `ExtendedData3.ipynb` | 3a, 3b, 3c, 3d | 1–2 | |
| `ExtendedData4.ipynb` | 4b, 4c | 2 | |
| `ExtendedData5.ipynb` | 5d, 5e | 2 | 5b, 5c need a table not in this release |
| `ExtendedData6.ipynb` | 6b–6e | 1 | |
| `ExtendedData7.ipynb` | 7a, 7b, 7c | 1 | |
| `ExtendedData8.ipynb` | 8b, 8c, 8f | 1 | 8a, 8d are cartoons; 8e is not scripted |
| `ExtendedData9.ipynb` | 9a, 9b | 1 | |

Where the manuscript prints a number, the notebook prints the computed value beside it and
marks `MATCH` or `CHECK`.

## Figures without a notebook

- **Figure 1** — schematic.
- **ED Figure 8a, 8d** — cartoons; **8e** is a worked example, not scripted.
- **ED Figure 1b, 1c** — ablation sweep. Numbers ship as `figure_data/ed1/master_sweep_553fam.csv`
  with `make_report.py`, so the panels redraw at level 1. The sweep spans fifteen checkpoints
  and is not rerunnable from this release.
- **Supplementary Figure 1** — dataset and splits, both in the release; no plotting script.

`../REPRODUCING_TUTORIAL.md` has the level-2 and level-3 commands per panel with measured
runtimes.
