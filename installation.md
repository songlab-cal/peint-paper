# Installation

Three things to do: clone both repos, build two conda environments, get the data.
Then see [Reproducing figures](#reproducing-figures) for what you can run at each tier.

The models live in the separate [`peint`](../peint) repo and are imported here as `protevo`.
Two environments are needed because ESM-C and the folding stack (OmegaFold, JAX/AF2Rank,
ProstT5) require incompatible `transformers` versions.

## Prerequisites

- Linux, Python 3.10, conda
- NVIDIA GPU, compute capability >= 8.0, CUDA 12.4 driver. Tested on an A100 80GB.
- MAFFT on `PATH`

Use pip inside a bare conda env; do not `conda install` torch, JAX or CUDA packages.

## 1. Clone

```bash
git clone https://github.com/songlab-cal/peint.git        peint
git clone https://github.com/songlab-cal/peint-paper.git  peint-paper
cd peint-paper
```

Sibling directories — the defaults assume it.

## 2. Environments

### `peint-esmc` — ESM-C generation, most panels

Build it per [`../peint/installation.md`](../peint/installation.md), then add:

```bash
conda activate peint-esmc
pip install -e ../peint
pip install matplotlib seaborn logomaker biotite 'tomli; python_version < "3.11"'
```

Do not `pip install transformers` here — it replaces the Biohub fork and ESM-C stops loading.

### `protevo-env` — folding, alignment, ESM-IF

**Install in this order.** JAX and PyTorch each ship CUDA wheels; the wrong order gives cuDNN
errors at runtime, not install time.

```bash
conda create -n protevo-env python=3.10 -y && conda activate protevo-env

pip install torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu124
pip install -U "jax[cuda12]==0.5.0"

pip install einops==0.8.1
pip install flash-attn==2.7.0.post2 --no-build-isolation

# ESM-IF's GVP-Transformer; must match the torch build
pip install torch_scatter torch_sparse torch_cluster \
    -f https://data.pyg.org/whl/torch-2.5.0+cu124.html
pip install torch_geometric

pip install transformers==4.45.2 sentencepiece          # ProstT5 / 3Di

git clone https://github.com/HeliXonProtein/OmegaFold /path/to/OmegaFold
pip install --no-deps /path/to/OmegaFold

pip install -q git+https://github.com/sokrypton/ColabDesign.git@v1.1.3 --no-deps
pip install dm-haiku==0.0.13 optax==0.2.2 chex==0.1.86 ml_collections dm-tree \
    immutabledict py3Dmol

pip install --no-deps -e ../peint                       # --no-deps protects the pins above
pip install biopython 'cherryml @ git+https://github.com/songlab-cal/CherryML' \
    loguru joblib tqdm
pip install matplotlib seaborn logomaker biotite scikit-learn pyhmmer tabulate \
    'tomli; python_version < "3.11"'
```

Verify — torch and JAX must both see the GPU in one process:

```bash
python - <<'EOF'
import torch;  print("torch", torch.__version__, torch.cuda.is_available())
import jax;    print("jax", jax.__version__, jax.devices())   # expect [CudaDevice(id=0)]
import protevo, omegafold, colabdesign, esm, torch_geometric
from transformers import T5EncoderModel
print("ok")
EOF
```

`[CpuDevice(id=0)]` means something moved `jax`/`jaxlib` after the install. Rebuild the env.

### Point the renderer at both

```bash
export PEINT_PAPER_PY_ESMC=/path/to/envs/peint-esmc/bin/python
export PEINT_PAPER_PY_PROTEVO=/path/to/envs/protevo-env/bin/python
export HF_HOME=/path/to/hf_cache          # ESM-C and ProstT5 weights land here
```

`render_panels.py` runs each panel in the right one. Unset, everything runs in the current
env and the cross-env panels fail.

## 3. Data

Everything resolves under `local_data/`. Pick a tier:

```bash
export PEINT_PAPER_ZENODO_RECORD=<record-id>

scripts/fetch_local_data.py --tier figure_data   # 19 MB   plots only
scripts/fetch_local_data.py --tier full          # ~22 GB  recompute + rerun
```

This downloads, verifies against `CHECKSUMS.sha256`, and unpacks. Then:

```bash
scripts/check_local_data.py                      # what is present, what is missing
```

| Archive | Download | Unpacked | Contents |
|---|---|---|---|
| `figure_data` | 19 MB | 21 MB | the table behind every panel |
| `aux` | 23 MB | 127 MB | annotations, split lists, site rates |
| `sim` | 201 MB | 543 MB | trees, root sequences, empirical MSAs |
| `r1` | 2.8 GB | 20 GB | revision 1: classical + PEINT-ESM2 |
| `r2` | 2.4 GB | 26 GB | revision 2: the ESM-C rerun |
| `peint_checkpoints` | 5.0 GB | 6.9 GB | the model checkpoints |
| `peint_transitions_{,un}aligned` | 4.5 GB | 26 GB | held-out transitions for the likelihood panel |
| `r1_af2`, `r1_omegafold` | 11 GB | 99 GB | raw rev1 structures; no panel reads them |

To unpack by hand: `tar --use-compress-program=unzstd -xf <archive>.tar.zst -C local_data/`.
The two structure archives are the exception — they take `-C local_data/r1/`.

`PEINT_PAPER_LOCAL_DATA=/big/disk/...` puts the tree elsewhere.

## Reproducing figures

```bash
scripts/render_panels.py --list     # every panel, its env, its cost
scripts/render_panels.py --group main
scripts/render_panels.py --only figure3_jsd_boxplot
```

### Tier 1 — replot from the shipped tables

`figure_data` (19 MB) holds the values behind every published panel. No models, no GPU.
`pcp_panels --from-csv` redraws in ~20 s; the other panels recompute their statistic even
in this mode, so this tier is mainly for checking plotted numbers against the paper.

### Tier 2 — recompute the metrics from the deposited simulations

With the `full` tier unpacked, most panels derive their statistic fresh from shipped
sequences and structures rather than reading a stored number: the conservation JSD panels,
the 3Di panel, both pLDDT ECDFs, the generalization panels, BLAST identity, ESM-IF.
No model is loaded — these read files.

### Tier 3 — rerun from a checkpoint

Regenerating the simulations themselves needs a checkpoint (`peint_checkpoints`) and a GPU.
This is where the two environments hand off.

**The handoff.** Simulation needs ESM-C; folding needs the JAX/OmegaFold stack. What crosses
between them is **sequences in text files**, so nothing has to be version-compatible:

```bash
# peint-esmc: load the checkpoint, simulate, write sequences
$PEINT_PAPER_PY_ESMC -m benchmarks.generate_all_results --out_path <dir>

# protevo-env: read those sequences off disk, fold and score them
$PEINT_PAPER_PY_PROTEVO -m benchmarks.omegafold_peint_esmc --out_path <dir>
$PEINT_PAPER_PY_PROTEVO -m benchmarks.threedi_jsd_all_models
```

`omegafold_peint_esmc.py` imports only `read_msa` / `write_msa` / the caching decorator — it
never constructs a model. Running the halves as separate jobs is the designed path.

**What that buys you.** From one set of simulated sequences:

| Output | From | Panel |
|---|---|---|
| held-out likelihood vs. time | scoring the deposited transitions with the checkpoint | `figure2_likelihood_eval` |
| pLDDT vs. time; pLDDT ECDFs | OmegaFold / AF2Rank over the simulated leaves | `figure2_simulation_plddt`, `figure3_*_ecdf` |
| conservation JSD; 3Di JSD | column entropies of simulated vs. real alignments | `figure3_jsd_boxplot`, `threedi_jsd_boxplot` |
| indel length distributions | Historian ancestral reconstruction | `historian_indel_*` |
| ESM-IF self-consistency | inverse-folding the predicted structures | `esmif_validation` |

`figure2_simulation` is the one script that needs both environments. Run it twice —
`--skip-structures` in `peint-esmc` to generate and cache the sequences, then again in
`protevo-env` where those are cache hits and only the folding runs. `render_panels.py` does
this automatically via `depends_on`.

Note the cache key includes the checkpoint path as a string, so both passes must use the
identical path. A mismatch in the folding pass surfaces as `model type 'esmc' not
recognized` — that means cache miss, not a broken install.

## External tools

| Tool | For | Source |
|---|---|---|
| MAFFT | alignment | `PATH`; [source](https://mafft.cbrc.jp/alignment/software/source.html) |
| IQ-TREE 2 / AliSim | classical simulation | `peint`'s `iqtree2` submodule |
| TM-align + AF2 weights | AF2Rank | `scripts/fetch_af2rank_assets.sh` |
| ProstT5 weights | 3Di states | `scripts/fetch_3di_weights.sh` |
| Historian | ancestral reconstruction (regeneration only) | `scripts/build_historian.sh` |
| OmegaFold / ESM-IF / ESM-C weights | — | downloaded on first use |

## Known issues

- `biotite>=1.0` renamed `filter_backbone`, which `fair-esm` 2.0.0 imports at load.
  `paper/esmif.py` shims it; do the same if you import `esm.inverse_folding` yourself.
- protevo cache keys hash absolute input paths, so a cache copied from another machine
  never hits. This is why `pcp_panels` is cold unless you built its cache in place.
- Flash Attention wheel failures: see `../peint/installation.md`.
