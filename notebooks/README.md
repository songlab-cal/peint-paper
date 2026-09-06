# PEINT — figure notebooks

One notebook per manuscript figure. Each is self-contained: open it, run it top to bottom, and
the panels appear inline.

Every notebook **defaults to level 1** — redraw a panel from a table shipped in the Zenodo
deposit. No model, no checkpoint, no GPU; seconds on a laptop. Cells that need more are marked
and left commented out, so nothing in the default path can silently start a six-hour job.

## Setup

```bash
git clone https://github.com/<org>/peint-paper.git && cd peint-paper
git clone https://github.com/<org>/peint.git ../peint

# the plotting environment (see the tutorial for the full recipe)
conda env create -p ./envs/peint-paper -f envs/peint-paper.yml

export PEINT_PAPER_REPO=$PWD
jupyter lab notebooks/
```

Then download the 19 MB summary tier, which is all the level-1 cells need:

```bash
python scripts/fetch_data.py --tier summary
```

## Coverage

| notebook | panels | default level |
|---|---|---|
| `Figure2.ipynb` | 2a, 2b, 2c | 1 (2b, 2c need a GPU) |
| `Figure3.ipynb` | 3b, 3c, 3d, 3e, 3f, 3g | 1–2 |
| `Figure4.ipynb` | 4c, 4e, 4f (4a, 4b, 4d documented) | 1 |
| `Figure5.ipynb` | 5a, 5b, 5c, 5d | 1 |
| `ExtendedData2.ipynb` | 2a–2g | — points at Fig. 2 and 3 |
| `ExtendedData3.ipynb` | 3a, 3b, 3c, 3d | 1–2 |
| `ExtendedData4.ipynb` | 4b, 4c | 2 |
| `ExtendedData5.ipynb` | 5d, 5e | 2 (GPU) |
| `ExtendedData6.ipynb` | 6b–6e | 1 |
| `ExtendedData7.ipynb` | 7a, 7b, 7c | 1 |
| `ExtendedData9.ipynb` | 9a, 9b | 1 |

Where the manuscript prints a number, the notebook prints ours beside it, so you can check the
reproduction rather than take it on trust.

## Figures with no notebook

- **Figure 1** is a schematic — nothing to compute.
- **Extended Data Figure 1** (b, c) is the ablation sweep. The evaluated numbers ship as
  `figure_data/ed1/master_sweep_553fam.csv` with a `make_report.py`; the sweep itself involves
  fifteen checkpoints and is not rerunnable from the deposit.
- **Extended Data Figure 8** (b, c, e, f) needs an interpretability module that is not yet
  released.
- **Supplementary Figure 1** describes the dataset and the splits, both of which are in the
  deposit; there is no plotting script.

`../REPRODUCING_TUTORIAL.md` is the full step-by-step version, including the level-2 and
level-3 paths and the measured runtimes.
