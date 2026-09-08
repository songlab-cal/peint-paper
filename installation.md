# Installation

Three things to do: clone both repos, build two conda environments, get the data.
Then see [Reproducing figures](#reproducing-figures) for what you can run at each tier.

The models live in the separate [`peint`](../peint) repo and are imported here as `peint`.
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

### `peint-paper` — folding, alignment, ESM-IF

**Install in this order.** JAX and PyTorch each ship CUDA wheels; the wrong order gives cuDNN
errors at runtime, not install time.

```bash
conda create -n peint-paper python=3.10 -y && conda activate peint-paper

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
import peint, omegafold, colabdesign, esm, torch_geometric
from transformers import T5EncoderModel
print("ok")
EOF
```

`[CpuDevice(id=0)]` means something moved `jax`/`jaxlib` after the install. Rebuild the env.

### Point the renderer at both

```bash
export PEINT_PAPER_PY_ESMC=/path/to/envs/peint-esmc/bin/python
export PEINT_PAPER_PY_PEINT=/path/to/envs/peint-paper/bin/python
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
| `transitions_192l` | 319 MB | 1.3 GB | reduced 192-leaf transition set (see below) |
| `r1_af2`, `r1_omegafold` | 11 GB | 99 GB | raw rev1 structures; no panel reads them |

To unpack by hand: `tar --use-compress-program=unzstd -xf <archive>.tar.zst -C local_data/`.
The two structure archives are the exception — they take `-C local_data/r1/`.

`PEINT_PAPER_LOCAL_DATA=/big/disk/...` puts the tree elsewhere.

### A smaller input set

The published figures use a 512-leaf-per-family simulation. `transitions_192l` is the same
kind of data at 192 leaves per family — 319 MB instead of tens of GB — and is worth using if
you want to exercise the transition and time-estimation machinery without the full download or
runtime:

```bash
scripts/fetch_local_data.py --archives transitions_192l
export PEINT_PAPER_TRANSITIONS_DIR=local_data/transitions_192l   # already the default
```

It backs `figures/figure2_time_estimation.py`, which is not one of the released panels. It is
not a drop-in for the held-out likelihood panel, which reads its own transition archives.

### trRosetta training set (optional)

Not in the deposit as it is provided by third-party and ~30 GB. 
Needed only by the panels that compare against experimental structures 
or real alignments: ESM-IF self-consistency, AF2Rank, and the
conservation JSD panels.

**Also worth using if one wants to use this data to train a PEINT model, 
using the dataset machinery in the peint repository**

```bash
curl -O https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz
tar -xzf training_set.tar.gz -C <root>/input_data/    # gives a3m/, npy/, pdb/
export PEINT_PAPER_DATA_ROOT=<root>
```

`paper_config` then finds `input_data/a3m` and `input_data/pdb` on its own; only those two are
read. Point `PEINT_PAPER_INPUT_A3M_DIR` / `PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR` at them
directly if you keep them somewhere else.

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

Regenerating the simulations needs a checkpoint and a GPU. Simulation runs in `peint-esmc`,
folding in `peint-paper`, and what crosses between them is **sequences in text files** — so
nothing has to be version-compatible across the two. Running the halves as separate jobs is
the designed path, not a workaround. Full recipe in
[Running from a checkpoint](#running-from-a-checkpoint-end-to-end) below.

**What that buys you.** From one set of simulated sequences:

| Output | From | Panel |
|---|---|---|
| held-out likelihood vs. time | scoring the deposited transitions with the checkpoint | `figure2_likelihood_eval` |
| pLDDT vs. time; pLDDT ECDFs | OmegaFold / AF2Rank over the simulated leaves | `figure2_simulation_plddt`, `figure3_*_ecdf` |
| conservation JSD; 3Di JSD | column entropies of simulated vs. real alignments | `figure3_jsd_boxplot`, `threedi_jsd_boxplot` |
| indel length distributions | Historian ancestral reconstruction | `historian_indel_*` |
| ESM-IF self-consistency | inverse-folding the predicted structures | `esmif_validation` |

### Running from a checkpoint, end to end

Everything below was run against the deposit alone (`PEINT_PAPER_LOCAL_DATA_ONLY=1`), on five
held-out families. You need three archives — about 12 GB — and no `r1`/`r2`:

```bash
scripts/fetch_local_data.py --archives peint_checkpoints sim aux \
    peint_transitions_aligned peint_transitions_unaligned
```

Have MAFFT, `iqtree2` and the environments' `bin/` on `PATH` (OmegaFold is a console script
in `peint-paper/bin`, so calling the interpreter directly is not enough).

**0. Pick families.** Both entry points take the same JSON: `in_family` are families seen in
training, `held_out_family` are not. This list of families allows you to subset.

`data/example_families.json` is a fixed set of **five held-out families**, ready to use.
It is five rather than a larger sample because every family in it must have all three
simulation inputs in `sim/` (tree, root sequence, empirical MSA), and only the 545
simulation families do — training families have transitions but were never given them:

```json
{"in_family": [], "held_out_family": ["1a2t_1_A", "1acf_1_A", "1amx_1_A", "1aq6_1_A", "1bai_1_A"]}
```

**1. Simulate** (`peint-esmc`, GPU). Loads the checkpoint, evolves each
root sequence down its tree, runs the WAG/LG baselines alongside, and scores amino-acid
conservation JSD against the real alignments:

```bash
$PEINT_PAPER_PY_ESMC -m benchmarks.generate_all_results \
    --families_path data/example_families.json \
    --peint_checkpoint_path local_data/peint/model_checkpoints/peint_esmc.ckpt \
    --tree_dir local_data/sim/trees_newick \
    --root_sequences_dir local_data/sim/root_sequences \
    --real_sequences_dir local_data/sim/empirical_msas \
    --out_path runs/example --include_conservation
```

Use `sim/trees_newick`, not `sim/trees` — the latter is a different node-list format and fails
with `NewickError`.

**2. Structures and 3Di** (`peint-paper`, GPU). The same command, different interpreter and
flags. The simulations are cache hits, so no model is built here; only folding and ProstT5
run. This is the whole handoff — what crosses between the environments is sequences in files:

```bash
$PEINT_PAPER_PY_PEINT -m benchmarks.generate_all_results \
    --families_path data/example_families.json \
    --tree_dir local_data/sim/trees_newick \
    --root_sequences_dir local_data/sim/root_sequences \
    --real_sequences_dir local_data/sim/empirical_msas \
    --out_path runs/example \
    --subsample_msa_size 3 --include_plddt --include_3di
```

`--subsample_msa_size` is how many leaves per family to fold; the paper uses 30, `-1` folds
everything. Add `--use_af2` for AF2Rank instead of OmegaFold.

**3. Held-out per-site likelihood** (`peint-esmc`, GPU). Scores the deposited transitions and
fits the WAG / LG rate matrices from the deposited training transitions:

```bash
$PEINT_PAPER_PY_ESMC -m figures.figure2_ll_eval_esmc \
    --esmc-checkpoint local_data/peint/model_checkpoints/peint_esmc.ckpt \
    --families-path data/example_families.json \
    --num-processes 1 --out-dir out
```

`--num-processes` becomes `mpirun -np` inside cherryml's counting step. Ask for more than your
machine or Slurm allocation has slots for and Open MPI refuses to launch silently — so the
script probes first and steps down, printing what it settled on. Omit `--families-path` for the
published 14,498 / 553 split.

**4. Time estimation** (`peint-esmc`, GPU). Uses the reduced transition set:

```bash
$PEINT_PAPER_PY_ESMC -m figures.figure2_time_estimation \
    --checkpoint local_data/peint/model_checkpoints/peint_esmc.ckpt \
    --num-families 5 --output-dir out
```

**What you get.** From one checkpoint:

| Output | Where |
|---|---|
| simulated MSAs, per model | `runs/example/simulations/{peint_progressive,peint_single_shot,wag,lg}_*` |
| predicted structures | `runs/example/omegafold/<family>/<model>/structures/*.pdb` |
| amino-acid conservation JSD | `runs/example/results/<family>/jsd_vs_real_aa.csv` |
| 3Di (structural alphabet) JSD | `runs/example/results/<family>/jsd_vs_real_foldseek.csv` |
| pLDDT, per sequence and aggregated | `runs/example/results/<family>/plddt.csv`, `results/omegafold_plddt/all_plddt.csv` |
| held-out likelihood vs. time | `out/figure2_likelihood_eval_{test,train_held_out}_esmc.pdf` |
| time-estimation panels | `out/` |

Swap the checkpoint to compare models; swap the family list to change the split.

`figure2_simulation` is the one script that needs both environments. Run it twice —
`--skip-structures` in `peint-esmc` to generate and cache the sequences, then again in
`peint-paper` where those are cache hits and only the folding runs. `render_panels.py` does
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
| trRosetta training set | ESM-IF, AF2Rank, conservation JSD | see [above](#trrosetta-training-set-optional) |
| Historian | ancestral reconstruction (regeneration only) | `scripts/build_historian.sh` |
| OmegaFold / ESM-IF / ESM-C weights | — | downloaded on first use |

## Known issues

- `biotite>=1.0` renamed `filter_backbone`, which `fair-esm` 2.0.0 imports at load.
  `paper/esmif.py` shims it; do the same if you import `esm.inverse_folding` yourself.
- peint cache keys hash absolute input paths, so a cache copied from another machine
  never hits. This is why `pcp_panels` is cold unless you built its cache in place.
- Flash Attention wheel failures: see `../peint/installation.md`.
- The classical baselines are fit through cherryml, which shells out to `mpirun`. Asking for
  more processes than there are MPI slots fails with no output; the likelihood script probes
  and steps down, but other entry points do not.
