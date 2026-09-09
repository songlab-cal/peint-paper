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

Most cells run at level 1 by default; the coverage table below names the exceptions. Six
panels are outside the replot tier — Fig. 2c, Fig. 3b, Fig. 3e, ED Fig. 3d and ED Fig. 5d/5e —
because they need a larger archive, a GPU, or the folding environment; their level-2/3 commands
are given in the cells. `render_panels.py --from-csv` skips them rather than silently starting a
full recomputation.

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

# Install torch FIRST. Several figure producers import `peint`, which depends on torch, so
# leaving it to the resolver picks the newest release and a CUDA build that may not match the
# driver on the machine. Level-1 replotting never runs a model, so the CPU wheel is enough and
# avoids CUDA entirely:
pip install "torch==2.5.*" --index-url https://download.pytorch.org/whl/cpu

pip install -e ../peint      # model library
pip install -e .             # this repository, plotting and analysis
pip install jupyterlab
```

That covers the level-1 cells, on CPU: no GPU, and flash-attn is not required.

For level 2 and 3 on a GPU, install the CUDA build that matches your driver instead of the CPU
wheel; the tested stack is torch 2.5.0+cu124. `../installation.md` has the details.

Figure 4's producer notebooks need two extras of their own — `ete3` renders the tree through
Qt for 4a-4c, and the stopped-flow fits use scikit-learn for 4e:

```bash
pip install -e ".[figure4]"            # PyQt5 + scikit-learn
conda install -c conda-forge poppler   # Fig. 4 panels are PDF; this shows them inline
```

Fig. 4d additionally needs a `tecantaloupe` clone and its own pandas<2 environment; the Fig. 4d
cell in `Figure4.ipynb` gives the recipe.

Structure panels (Fig. 3f, 3g; ED Fig. 4b, 5d, 5e) also need the folding stack. Two of its
packages are not on PyPI:

```bash
pip install -e ".[folding]"
pip install -q git+https://github.com/sokrypton/ColabDesign.git@v1.1.3 --no-deps
pip install --no-deps -e <OmegaFold checkout>
```

Headless machines need `QT_QPA_PLATFORM=offscreen`; the Figure 4 notebook sets it.

Level 3 uses two further environments — one for PEINT, one for the ESM-C variant — because
JAX and PyTorch need different CUDA builds and ESM-C needs a `transformers` fork. Recipes and
pinned versions are in `../REPRODUCING.md`.

### 3. Data

Everything downloads into `local_data/`. Paths in the notebooks are relative to it.

`fetch_local_data.py` needs to know which Zenodo record to read, either through
`PEINT_PAPER_ZENODO_RECORD` or `--record`, and it unpacks `.tar.zst` archives, so **`zstd`
must be on `PATH`**:

```bash
export PEINT_PAPER_ZENODO_RECORD=<record-id>             # see the top-level README
python scripts/fetch_local_data.py --list                # archives, sizes, contents
python scripts/fetch_local_data.py --tier figure_data    # 20 MB, the level-1 tables
```

While the Zenodo record is unreachable, the same 20 MB archive is mirrored on this
repository's `zenodo-22151902` release; unpack it into `local_data/` by hand:

```bash
tar --use-compress-program=unzstd -xf figure_data.tar.zst -C local_data/
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
| `Figure2.ipynb` | 2a, 2b, 2c | 1 | 2a, 2b redraw at level 1; **2c** is level 3 — a checkpoint, a GPU and Historian, with no replot path |
| `Figure3.ipynb` | 3b, 3c, 3d, 3e, 3f, 3g | 1–2 | 3c, 3d, 3f, 3g redraw at level 1; **3b** needs `sim/` (a per-site logo, so no table can redraw it) and **3e** needs `r2` (per-event lengths are not in the shipped per-family table) |
| `Figure4.ipynb` | 4a, 4b, 4c, 4d, 4e, 4f | 1 | inputs are in the repository; 4d needs a tecantaloupe clone and its own small env |
| `Figure5.ipynb` | 5a, 5b, 5c, 5d | 1 | |
| `ExtendedData1.ipynb` | 1b, 1c | 1 | |
| `ExtendedData2.ipynb` | 2a–2g | — | ESM-C arm of Figures 2 and 3; produced by those commands |
| `ExtendedData3.ipynb` | 3a, 3b, 3c, 3d | 1–2 | 3a–3c redraw at level 1; **3d** recomputes from `r1`/`r2` — its producer has no replot path |
| `ExtendedData4.ipynb` | 4b, 4c | 1 | both redraw from shipped tables |
| `ExtendedData5.ipynb` | 5d, 5e | 2 | needs the folding environment — ESM-IF's GVP-Transformer is imported at module load (`torch_geometric`) even when the scored tables are complete; 5b, 5c need a table not yet recovered |
| `ExtendedData6.ipynb` | 6b–6e | 1 | |
| `ExtendedData7.ipynb` | 7a, 7b, 7c | 1 | |
| `ExtendedData8.ipynb` | 8b, 8c, 8f | 1 | 8a, 8d are diagrams; 8e is not scripted |
| `ExtendedData9.ipynb` | 9a, 9b | 1 | |

Where the manuscript prints a number, the notebook prints the computed value beside it and
marks `MATCH` or `CHECK`.

## Reproduction gaps

These notebooks ship **55 panels with their outputs embedded**. Three entries in the figure list
are not reproduced by a notebook cell. In each case what is missing is a script or an input, not
a result we have reason to doubt; where we are recovering the missing piece, that is said.

| figure | what is missing | status |
|---|---|---|
| **ED Fig. 5b, 5c** | the AlphaFold2-with-MSA prediction table these two panels compare against | produced for the original analysis, not yet recovered from the run that made it |
| **ED Fig. 8e** | a script for the single worked alignment example; its inputs are the same categorical-Jacobian tensors as 8b | not scripted |
| **Supp. Fig. 1** | a plotting script | the dataset and splits it summarises are both in the release |

Figure 1 and ED Figure 8a, 8d are diagrams rather than plots, so there is nothing to reproduce
for them.

`../REPRODUCING.md` has the level-2 and level-3 commands per panel with measured
runtimes.
