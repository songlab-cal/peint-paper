# peint-paper Installation Instructions

This repo holds the figures and benchmarks. The **models** live in the separate
[`peint`](../peint) repo (imported as the `protevo` package) and are installed into these
environments editable — nothing here vendors a model.

Installation is genuinely involved, for one reason: the benchmarks span three stacks that do not
pin together — **JAX** (AF2Rank via ColabDesign), **PyTorch** (PEINT, OmegaFold, ESM-C, ESM-IF),
and the **older `transformers`** that ProstT5 is pinned against. So there are two environments.
The README explains *why* the split exists and which panels land where; this document is the
*how*.

If you only want to redraw the published panels from the shipped tables
(`scripts/render_panels.py --from-csv`), you need **neither** environment in full — see
[Minimal: plotting only](#minimal-plotting-only) at the end.

## Setup, in order

Every step is detailed further down; this is the checklist. Steps 1-4 are the install, 5-6 get
the data, 7 verifies.

```bash
# 1. Both repos, as siblings (the docs and the default paths assume this layout)
git clone <peint repo>        peint
git clone <peint-paper repo>  peint-paper
cd peint-paper
```

2. **Build `peint-esmc`** — follow `../peint/installation.md` (the `peint` core env, then the
   ESM-C clone with the pinned Biohub `transformers` fork), then add this repo's plotting deps.
   See [Environment 1](#environment-1--peint-esmc).

3. **Build `protevo-env`** — the JAX + PyTorch + OmegaFold + ESM-IF stack, in the exact order
   given in [Environment 2](#environment-2--protevo-env). Order matters; do not improvise it.

4. **Install the model library into both** — `pip install -e ../peint` in `peint-esmc`,
   `pip install --no-deps -e ../peint` in `protevo-env`. See
   [Installing the model library](#installing-the-model-library-peint).

5. **External tools** — MAFFT on `PATH`, IQ-TREE 2 built in the `peint` submodule, and (only if
   you regenerate ancestral reconstructions) Historian. See [External tools](#external-tools).

6. **Get the data and the checkpoints** — pick the branch that matches your machine:

```bash
# ---- 6a. From the Zenodo record (the published deposit) -----------------------------
#      See "Downloading the deposit" below for the full recipe and what each archive costs.
ZENODO=https://zenodo.org/records/<record-id>/files
mkdir -p local_data
for a in figure_data aux sim r1 r2 peint_checkpoints; do
    curl -L -O "$ZENODO/$a.tar.zst?download=1"
    tar --use-compress-program=unzstd -xf "$a.tar.zst" -C local_data/
done
# The two rev1 structure archives take a different -C; see the † note below. Optional:
# no panel reads them, and they are 11 GB down / 99 GB on disk.

# ---- 6b. Same thing, scripted (downloads, verifies checksums, unpacks) -------------
export PEINT_PAPER_ZENODO_RECORD=<record-id>
scripts/fetch_local_data.py --tier figure_data   # ~5 MB packed: replot every panel, no models
scripts/fetch_local_data.py --tier full          # ~22 GB packed / ~151 GB unpacked,
                                                 # INCLUDING the model checkpoints

# ---- 6c. You already hold the source trees on this filesystem ----------------------
scripts/link_local_data.sh                       # dry run: show the plan
scripts/link_local_data.sh --apply               # symlinks, nothing copied
```

7. **Wire the two interpreters and verify:**

```bash
export PEINT_PAPER_PY_ESMC=/path/to/envs/peint-esmc/bin/python
export PEINT_PAPER_PY_PROTEVO=/path/to/envs/protevo-env/bin/python

scripts/check_local_data.py       # which data roles are present, and what supplies the rest
scripts/render_panels.py --list   # every panel, its env, and its cost
scripts/render_panels.py --check  # which outputs are still missing
```

### Where the data lands, and how to move it

Every archive unpacks **relative to `local_data/`** — `tar -xf <archive>.tar.zst -C local_data/`
is the whole instruction, and `fetch_local_data.py` does exactly that. The layout:

```
local_data/
  peint/model_checkpoints/    peint.ckpt, vep.ckpt, peint_esmc.ckpt, vep_esmc.ckpt
  figure_data/                per-panel tables — enough to re-render every panel
  vep/                        per-run ProteinGym Spearman tables (git-tracked, ships with the code)
  r1/  r2/                    revision 1 (ESM2 + classical) and revision 2 (the ESM-C rerun)
  sim/                        trees, root sequences, empirical + simulated MSAs
  annotations/  splits/       Pfam/SCOPe/ECOD labels; the 14,498 train / 553 held-out lists
  derived/                    everything the benchmarks WRITE, including the computation caches
```

If you downloaded the archives by hand (e.g. from the Zenodo record rather than through the
script), unpack them yourself into the same place:

```bash
tar --use-compress-program=unzstd -xf peint_checkpoints.tar.zst -C local_data/
```

To keep the bulk off the checkout's filesystem, either point the tree elsewhere or symlink it —
both work, and `local_data/` is gitignored either way:

```bash
export PEINT_PAPER_LOCAL_DATA=/big/disk/peint_paper_data      # move the whole tree
ln -s /big/disk/peint_paper_data local_data                   # or symlink it in place
```

### Downloading the deposit

The deposit is a small set of `.tar.zst` archives plus two metadata files. **Every archive
unpacks relative to `local_data/`**, so there is one instruction regardless of where you got it:

```bash
tar --use-compress-program=unzstd -xf <archive>.tar.zst -C local_data/
```

What the record holds, and what each piece costs (packed sizes are projections from measured
zstd -19 ratios on this data, not yet-built archives):

| Archive | Packed | Unpacked | Files | What it is |
|---|---|---|---|---|
| `figure_data.tar.zst` | ~5 MB | 25 MB | 20 | every table needed to redraw the panels |
| `aux.tar.zst` | ~21 MB | 127 MB | 30,117 | annotations, split lists, per-site rates |
| `sim.tar.zst` | ~0.22 GB | 543 MB | 6,010 | trees, root sequences, empirical + simulated MSAs |
| `r1.tar.zst` | ~2.2 GB | 20 GB | 74,440 | revision 1: classical + PEINT-ESM2 |
| `r2.tar.zst` | ~2.4 GB | 26 GB | 75,904 | revision 2: the ESM-C rerun |
| `peint_checkpoints.tar.zst` | ~6.2 GB | 6.9 GB | 5 | the model checkpoints |
| `r1_af2.tar.zst` | 5.6 GB | 52 GB | 481,316 | raw rev1 AF2Rank structures † |
| `r1_omegafold.tar.zst` | 5.3 GB | 46 GB | 607,696 | raw rev1 OmegaFold structures † |
| **full tier** | **~22 GB** | **~151 GB** | **~1.28 M** | everything above |
| *without the two structure trees* | *~11 GB* | *~53 GB* | *~186 k* | the practical default |

**† These two unpack into `local_data/r1/`, not `local_data/`.** They were streamed straight from
their source directories by `stage_shared_data.sh --as-archive`, so their members are prefixed
`af2/` and `omegafold/` rather than `r1/af2/` and `r1/omegafold/`:

```bash
tar --use-compress-program=unzstd -xf r1_af2.tar.zst       -C local_data/r1/
tar --use-compress-program=unzstd -xf r1_omegafold.tar.zst -C local_data/r1/
```

`fetch_local_data.py` reads this from the archive's `unpack_into` field and gets it right on its
own; it only matters if you unpack by hand. Note that **no panel reads either** — the ECDFs
consume `af2rank_comparisons.csv` and `omegafold_plddt.csv`, 750 KB combined, which ship inside
`r1.tar.zst`. They are deposited so the structures behind those two summaries are citable, not
because anything needs them: skipping both saves 11 GB of download and 99 GB on disk.

Two things are worth knowing before you plan a download. The checkpoints are **more than half the
packed deposit** despite being 13% of the unpacked bytes — model weights are float tensors and
compress at only 1.08x, while the sequence trees compress 8-18x. And if you only want to redraw
figures, `figure_data.tar.zst` alone is ~5 MB and needs no model, no GPU, and none of the rest.

**Verify before unpacking.** The record ships `CHECKSUMS.sha256` covering every archive:

```bash
curl -L -O "$ZENODO/CHECKSUMS.sha256?download=1"
sha256sum -c CHECKSUMS.sha256 --ignore-missing
```

**After unpacking**, confirm the figures can see everything — this reads the manifest, so it
checks the layout rather than just the presence of files:

```bash
scripts/check_local_data.py
```

**On the fetch script.** `scripts/fetch_local_data.py` does the whole loop against the Zenodo
record — download, verify against `CHECKSUMS.sha256`, unpack each archive to the right place
(including the two that need `-C local_data/r1/`). It uses only the standard library, so it
needs nothing installed beyond Python and `tar`. Point it at the record once:

```bash
export PEINT_PAPER_ZENODO_RECORD=<record-id>
scripts/fetch_local_data.py --tier full
scripts/fetch_local_data.py --archives figure_data r1     # or pick archives by name
scripts/fetch_local_data.py --verify-only                 # re-check an existing download
```

The manual `curl` + `tar` loop in step 6a produces a byte-identical tree if you would rather not
use it.

### Where the checkpoints come from

`paper_config` resolves each checkpoint in this order:

1. the `peint` checkout's `model_checkpoints/` — what you get with a sibling clone or step 6b;
2. `local_data/peint/model_checkpoints/` — what the deposit unpacks, so a downloader who never
   clones the model repo still has them;
3. the matching `PEINT_PAPER_*_CHECKPOINT` environment variable, which overrides both.

Set `PEINT_PAPER_LOCAL_DATA_ONLY=1` to force route 2 outright. That is the honest way to check
that the deposit is self-sufficient: on a machine that has both, route 1 always wins and you
cannot otherwise tell whether the deposit would have worked.

## The two environments

| | `peint-esmc` | `protevo-env` |
|---|---|---|
| **Runs** | most panels; **all PEINT-ESM-C generation** | PCP panels, the two Historian indel panels, the 3Di panel; **all OmegaFold folding**, AF2Rank, ESM-IF |
| **Built as** | clone of the `peint` repo's `peint-esmc` env + plotting deps | its own env, older pins, built in a strict order |
| `transformers` | 4.57.6 (Biohub fork — knows the `esmc` architecture) | 4.45.2 (ProstT5's pin; raises `model type 'esmc' not recognized`) |
| `numpy` / `scipy` / `pandas` | 2.2.6 / 1.15.3 / 2.3.3 | 1.26.4 / 1.13.1 / 2.2.2 |
| JAX | — | 0.5.0, alongside torch |
| OmegaFold / ColabDesign / ESM-IF | — | installed |
| `protevo` (the `peint` repo) | editable | editable |

Both are Python **3.10.19** with **torch 2.5.0+cu124** and **flash-attn 2.7.0.post2**. Both have
the `peint` repo installed editable, so `import protevo` resolves to your checkout in either.

`figure2_simulation` needs both and is run twice — first in `peint-esmc` with `--skip-structures`
to generate and cache the sequences, then in `protevo-env` without the flag, where the
simulations are cache hits and only the folding runs. `render_panels.py` handles this for you
once both interpreters are pointed at (see [Wiring](#wiring-the-two-environments)).

## How the two environments hand off

The obvious worry is that the simulations need PEINT (so, the ESM-C stack) while the benchmarks
need the JAX/folding stack, and no environment has both. It works because **what crosses the
boundary is amino-acid sequences in text files, not models or tensors** — so nothing
torch-shaped or JAX-shaped ever has to be version-compatible across the two.

There are two mechanisms, and it is worth knowing which one you are relying on.

### 1. File handoff — the main pipeline

`benchmarks/generate_all_results.py` runs in `peint-esmc` and writes simulated sequences under
its `--out_path`. The structure benchmarks then run in `protevo-env` and *read those files*. No
model is ever constructed in the second environment. `benchmarks/omegafold_peint_esmc.py` and
`af2rank_peint_esmc.py` import only:

```python
from protevo import caching as pc
from protevo.utils import read_msa
from protevo.io import write_msa
```

— no `load_model`, no `protevo.models`. `protevo` is present for its caching decorator and text
I/O, neither of which touches torch. The results tree shows the shape directly: `simulations/`
written by the ESM-C env, then `omegafold/`, `af2/`, `3di/`, `mafft_add/` written beside it by
`protevo-env`.

Running the two halves as separate jobs is not a workaround — it is the designed path.

`omegafold_peint_esmc.py` goes one step further and writes its PDBs to the *same* path
`generate_all_results` uses, so a later `generate_all_results ... --include_plddt` cache-hits
them and only runs the pLDDT scoring and aggregation.

### 2. Cache handoff — `figure2_simulation` only

This one script is run twice, and the join is protevo's computation cache. The trick is that
**the model load sits inside the cached function body**:

```python
@protevo_caching.cached_computation(output_dirs=["output_sequences_dir"],
                                    exclude_args=["device"], ...)
def simulate_evolution_peint(family, starting_sequence, model_checkpoint_path, ...):
    model, vocab = load_model(model_checkpoint_path=..., ...)   # only reached on a cache MISS
```

and the decorator's logic is `if not computed(): func(*args, **kwargs)`
(`protevo/caching/_cached_computation.py`). On a hit it returns the output directories having
never entered the body — so `load_model` is never called, and the Biohub `transformers` fork is
never imported in `protevo-env`. Hence:

```bash
# pass 1, in peint-esmc: generate and cache the sequences
$PEINT_PAPER_PY_ESMC    -m figures.figure2_simulation --skip-structures
# pass 2, in protevo-env: the sims are cache hits; only the folding runs
$PEINT_PAPER_PY_PROTEVO -m figures.figure2_simulation
```

`render_panels.py` encodes this as two panels (`figure2_simulation_generate` and
`figure2_simulation_plddt`, the latter with `depends_on`), so asking for the pLDDT panel runs the
generate pass first, in the right interpreter.

**What the cache key is built from:** every argument except `exclude_args`, `output_dirs`, and
any `exclude_args_if_default` left at its default. `device` is excluded, so cuda/cpu is free —
but `model_checkpoint_path` **is** in the key, as a string. The two passes must pass the
byte-identical path. Keys also hash absolute input paths, so a cache copied from another machine
never hits; this is the same constraint `REPRODUCING.md` flags for `pcp_panels`.

### The failure mode this produces

`figures/figure2_simulation.py` calls `protevo_caching.set_read_only(False)` for both caches in
both passes. So a key mismatch during the folding pass does **not** report a cache miss — it
falls into the function body and tries to build ESM-C inside `protevo-env`, where transformers
4.45.2 raises:

```
ValueError: The checkpoint you are trying to load has model type `esmc` but Transformers
does not recognize this architecture.
```

**If you see that during the folding pass, it means cache miss, not broken install.** The usual
causes are a different checkpoint path string between the two passes, a moved cache directory, or
a changed non-default keyword argument. Switching pass 2 to `set_read_only(True)` would turn that
silent fallthrough into an immediate `CacheUsageError` naming the miss; that is a one-line change
in `figure2_simulation.py`, not an existing flag.

## Prerequisites

- Linux, **Python 3.10**, conda (or mamba)
- An NVIDIA GPU with **compute capability ≥ 8.0** (Ampere+) for Flash Attention, and a driver
  supporting **CUDA 12.4**. Tested on an A100 80GB.
- For building Historian (only if you regenerate ancestral reconstructions): `g++`/`clang++`,
  `make`, GSL (`libgsl-dev`), Boost.regex, zlib.
- **MAFFT** on `PATH`.

Do **not** mix `conda install` and `pip install` for torch/JAX/CUDA packages. Every recipe below
is pip-only inside a bare conda env. This is the single most common way to break the JAX+torch
coexistence.

## Environment 1 — `peint-esmc`

This is the `peint` repo's ESM-C environment plus this repo's plotting dependencies. Build it by
following **`../peint/installation.md`** (the `peint` core env, then the ESM-C clone with the
pinned Biohub `transformers` fork), then add what the figures need:

```bash
conda activate peint-esmc

pip install -e ../peint                 # the model library, as `protevo`
pip install matplotlib seaborn logomaker biotite 'tomli; python_version < "3.11"'
```

The ESM-C backbone is downloaded from the HuggingFace Hub repo `biohub/ESMC-300M`, so set
`HF_HOME` as described in `../peint/installation.md`.

Do not install `transformers` from PyPI into this environment — it will replace the Biohub fork
and ESM-C will stop loading.

## Environment 2 — `protevo-env`

**Order of operations matters here.** JAX and PyTorch each ship their own CUDA/cuDNN wheels, and
installing them in the wrong order (or letting a later resolver move one of them) produces
mismatches that surface as cryptic cuDNN/PJRT errors at runtime rather than at install time.

```bash
conda create -n protevo-env python=3.10 -y
conda activate protevo-env

# --- 1. PyTorch FIRST, pinned, from the cu124 index -------------------------------
pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124

# --- 2. JAX SECOND. Do not upgrade it later. --------------------------------------
pip install -U "jax[cuda12]==0.5.0"

# --- 3. Flash Attention (see ../peint/installation.md if the build fails) ---------
pip install einops==0.8.1
pip install flash-attn==2.7.0.post2 --no-build-isolation

# --- 4. The geometric stack for ESM-IF's GVP-Transformer --------------------------
#     These must match the torch build exactly; take them from the PyG wheel index.
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch_geometric

# --- 5. ProstT5 (3Di states). This pin is what keeps the env off new transformers. --
pip install transformers==4.45.2 sentencepiece

# --- 6. OmegaFold — from a checkout, no deps (its pins are stale) ------------------
git clone https://github.com/HeliXonProtein/OmegaFold /path/to/OmegaFold
pip install --no-deps /path/to/OmegaFold

# --- 7. ColabDesign / AF2Rank — no deps, so it cannot move jax ---------------------
pip install -q git+https://github.com/sokrypton/ColabDesign.git@v1.1.3 --no-deps
pip install dm-haiku==0.0.13 optax==0.2.2 chex==0.1.86 ml_collections dm-tree immutabledict py3Dmol

# --- 8. The model library, WITHOUT deps, so pip cannot disturb the pins above ------
pip install --no-deps -e ../peint
pip install biopython 'cherryml @ git+https://github.com/songlab-cal/CherryML' \
    loguru joblib tqdm

# --- 9. This repo's figure/benchmark dependencies ---------------------------------
pip install matplotlib seaborn logomaker biotite scikit-learn pyhmmer tabulate \
    'tomli; python_version < "3.11"'
```

Step 8's `--no-deps` is deliberate: `../peint/pyproject.toml` declares `lightning>=2.5.2`, and
letting pip resolve it here can pull packages that step on the torch/JAX pins. The environment
that produced the results has `lightning 2.5.0.post0` and works. Install the training extras only
if you actually intend to train from this env — you do not need them to render figures.

Step 7's extra list is ColabDesign's real dependency set, installed explicitly because
`--no-deps` skips it; see
[this ColabDesign issue comment](https://github.com/sokrypton/ColabDesign/issues/195#issuecomment-2957151908).

### Versions this was tested with

Python 3.10.19. The full list is long; these are the ones that matter:

| Package | Version | |
|---|---|---|
| `torch` / `torchvision` / `torchaudio` | 2.5.0+cu124 / 0.20.0+cu124 / 2.5.0+cu124 | |
| `jax` / `jaxlib` | 0.5.0 / 0.5.0 | + `jax-cuda12-plugin`, `jax-cuda12-pjrt` 0.5.0 |
| `nvidia-cudnn-cu12` | 9.1.0.70 | pulled by torch |
| `flash-attn` | 2.7.0.post2 | compute capability ≥ 8.0 |
| `transformers` / `tokenizers` | 4.45.2 / 0.20.3 | ProstT5 |
| `sentencepiece` | 0.2.0 | ProstT5 tokenizer |
| `colabdesign` | 1.1.3 (`a95438f`) | AF2Rank |
| `OmegaFold` | 0.0.0 (local checkout) | |
| `fair-esm` | 2.0.0 | ESM2 + ESM-IF |
| `torch_geometric` | 2.8.0.post1 | |
| `torch_scatter` / `torch_sparse` / `torch_cluster` | 2.1.2 / 0.6.18 / 1.6.3, all `+pt25cu124` | ESM-IF |
| `biotite` | 1.2.0 | see the shim note below |
| `numpy` / `scipy` / `pandas` | 1.26.4 / 1.13.1 / 2.2.2 | older than `peint-esmc` |
| `cherryml` / `ete3` | 0.2.0 / 3.1.3 | |
| `matplotlib` / `seaborn` / `logomaker` | 3.10.3 / 0.13.2 / 0.8.7 | |
| `scikit-learn` / `pyhmmer` | 1.3.2 / 0.12.1 | generalization analysis |

## Installing the model library (`peint`)

Both environments import the models as **`protevo`**, from an editable install of the `peint`
repo. It is deliberately *not* a dependency in `pyproject.toml` — listing it would make pip try
to resolve `protevo` from PyPI on a fresh environment, which fails.

```bash
git clone <peint repo> ../peint      # a sibling checkout is what the docs assume
pip install -e ../peint              # peint-esmc
pip install --no-deps -e ../peint    # protevo-env (protect the torch/JAX pins)
```

Checkpoints are not part of the pip install. `paper_config` looks for them in the `peint`
checkout's `model_checkpoints/`, and falls back to `local_data/peint/model_checkpoints/` — which
is where the published deposit puts them, so unpacking the deposit is sufficient if you do not
have the model repo's copies. Setting `PEINT_PAPER_LOCAL_DATA_ONLY=1` makes `local_data/` the
only source, which is the honest way to check that the deposit is self-sufficient.

## Installing this repo

The tested environments do **not** pip-install `peint-paper` itself — the scripts are run as
modules from the repo root (`python -m figures.figure3_conservation`), and none of them depend on
the working directory beyond that. So you only need the dependencies, which the sections above
install.

`pip install -e .` also works and pulls those dependencies in one step. If you use it in
`protevo-env`, prefer `pip install --no-deps -e .` and install the dependencies explicitly, for
the same pin-protection reason as above.

## Wiring the two environments

`scripts/render_panels.py` subprocesses each panel with the right interpreter, so this is the
only wiring you need:

```bash
export PEINT_PAPER_PY_ESMC=/path/to/envs/peint-esmc/bin/python
export PEINT_PAPER_PY_PROTEVO=/path/to/envs/protevo-env/bin/python
```

Both default to `sys.executable`, so leaving them unset silently runs every panel in whichever
env you launched from — which will fail on the panels that need the other one. Set them.

```bash
scripts/render_panels.py --list       # panels, their env, and what each costs
scripts/render_panels.py --dry-run    # the exact commands, including the interpreter
```

## External tools

| Tool | Needed for | How to get it |
|---|---|---|
| **MAFFT** | alignment (PCP panels, `mafft_add` frames) | expected on `PATH`; [source](https://mafft.cbrc.jp/alignment/software/source.html) |
| **IQ-TREE 2 / AliSim** | classical-model simulation | built in the `peint` repo's `iqtree2` submodule — see `../peint/installation.md` |
| **Historian** | regenerating ancestral reconstructions only (no released panel needs it) | `git submodule update --init --recursive historian` then `scripts/build_historian.sh` (ships a patched Makefile; upstream's does not build on recent Linux) |
| **TM-align** | AF2Rank | `scripts/fetch_af2rank_assets.sh` → `bin/TMalign` |
| **blastp** | the BLAST identity panel | run outside this repo; the results ship as data |

## Downloaded assets and caches

None of these are in the pip install; each has a script or an env override.

| Asset | Fetch with | Lands in |
|---|---|---|
| AlphaFold 2022-12-06 params (~4 GB) + TM-align | `scripts/fetch_af2rank_assets.sh` | `data/af2_params/`, `bin/` |
| ProstT5 encoder + 3Di CNN head | `scripts/fetch_3di_weights.sh` | `data/prostt5/` |
| lm-design projection + n-gram stats | `scripts/fetch_lmdesign_assets.sh` | `data/lm_design/` |
| OmegaFold weights | downloaded on first run | `~/.cache/omegafold_ckpt` |
| ESM-IF `esm_if1_gvp4_t16_142M_UR50` | downloaded on first run by `fair-esm` | torch hub cache |
| ESM-C `biohub/ESMC-300M` | on first run (`peint-esmc` only) | `$HF_HOME` |
| trRosetta training set (a3m + ground-truth PDBs, ~20 GB) | **not scripted** — <https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz> | point `PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR` / `PEINT_PAPER_INPUT_A3M_DIR` at it |
| The figure data / benchmark archives | `scripts/fetch_local_data.py --tier figure_data\|full` | `local_data/` |

Set `HF_HOME` (not the deprecated `TRANSFORMERS_CACHE`, which recent `transformers` warns about
and v5 removes) to somewhere with room — on a cluster, scratch rather than your home quota:

```bash
export HF_HOME=/path/to/hf_cache
```

## Verifying the install

`protevo-env` — the fragile one, because torch and JAX must both see the GPU **in the same
process**:

```bash
conda activate protevo-env
python - <<'EOF'
import torch;  print("torch", torch.__version__, "| cuda:", torch.cuda.is_available())
import jax;    print("jax", jax.__version__, "| devices:", jax.devices())   # [CudaDevice(id=0)]
import protevo; print("protevo:", protevo.__file__)                        # your ../peint checkout
from transformers import T5EncoderModel, T5Tokenizer                       # ProstT5
import esm, torch_geometric, torch_scatter, torch_sparse, torch_cluster    # ESM-IF stack
import omegafold, colabdesign
print("protevo-env OK")
EOF
```

If `jax.devices()` returns `[CpuDevice(id=0)]`, JAX is not seeing CUDA — almost always because
something reinstalled or upgraded `jax`/`jaxlib` after step 2, or because torch and JAX were
installed in the other order. Rebuild the env rather than trying to patch it.

`peint-esmc` — verify per `../peint/installation.md`, then confirm the plotting deps:

```bash
conda activate peint-esmc
python -c "import protevo, matplotlib, seaborn, logomaker, biotite; print('peint-esmc OK')"
```

Then check the wiring end to end:

```bash
scripts/render_panels.py --check      # which outputs are missing
scripts/check_local_data.py           # which data roles are present
```

## Known gotchas

- **`pyproject.toml` says `transformers==4.52.2`; the tested env has 4.45.2.** The env is what
  produced the results. This discrepancy is flagged in the README and should be resolved one way
  or the other before release.
- **ProstT5 is the only thing pinning `protevo-env` to old `transformers`,** and it uses just
  `T5EncoderModel` / `T5Tokenizer`, which are stable well past 4.57. Bumping that pin would
  likely collapse the two environments into one. Worth trying; not yet done.
- **`biotite>=1.0` renamed `filter_backbone`**, which `fair-esm` 2.0.0 imports at module load.
  `paper/esmif.py` shims it before importing `esm.inverse_folding`. If you import
  `esm.inverse_folding` yourself, apply the same shim or pin `biotite<1.0`.
- **protevo's cache keys include absolute input paths** (sha512 over the arguments), so a warm
  cache is only valid at the path where it was built. Copying one elsewhere produces a directory
  that will never be hit. This is why `pcp_panels` is the one genuinely cold panel — see
  `REPRODUCING.md`.
- **Flash Attention wheels** are the usual install failure. The troubleshooting is in
  `../peint/installation.md` (GLIBCXX mismatches, picking the right
  `cxx11abiTRUE`/`cxx11abiFALSE` prebuilt wheel); it is not duplicated here.

## Minimal: plotting only

`REPRODUCING.md` describes a tier that redraws panels from the shipped tables with no models and
no GPU. A light environment covers that, but read the caveat below before assuming it covers the
whole sweep:

```bash
conda create -n peint-paper-plot python=3.10 -y && conda activate peint-paper-plot
pip install numpy pandas scipy matplotlib seaborn tqdm biopython ete3 biotite logomaker \
    'tomli; python_version < "3.11"'
pip install torch --index-url https://download.pytorch.org/whl/cpu   # see caveat
pip install --no-deps -e ../peint                                    # see caveat

scripts/fetch_local_data.py --tier figure_data     # ~40 MB
scripts/render_panels.py --only pcp_panels --from-csv
```

**Caveat — `--from-csv` is not a global cheap path.** Of the 18 panels in the whitelist, exactly
one (`pcp_panels`) declares a `from_csv_argv` (`--replot`). For the other 17, `--from-csv` falls
through to the panel's normal argv, so it runs its full path — including
`figure2_simulation_plddt` (GPU, ~1.5 h) and `figure3_jsd_boxplot` (~30-60 min). Separately,
`figures/figure2_simulation.py` and `figures/figure2_time_estimation.py` import `torch` and
`protevo` at module level, so those two modules will not even import without them, regardless of
flags.

So this environment is the right one for the panels that genuinely read a saved table, reached
with `--only`. A full `scripts/render_panels.py --from-csv` sweep still needs the two real
environments. If the intent is that every panel be redrawable from `figure_data` alone, the
panels need `from_csv_argv` entries — worth resolving before release, since `REPRODUCING.md`
currently promises the broader behavior.
