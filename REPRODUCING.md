# Reproducing the figures

Three questions, answered separately, because conflating them is how reproducibility
claims become untrue:

1. **Can you redraw the published panels?** Yes, all of them, from ~40 MB.
2. **Can you recompute the numbers behind them?** Most of them, from ~53 GB.
3. **Can you regenerate that 53 GB from raw inputs?** Partly, and the parts you cannot
   are named below rather than glossed.

Everything here is driven by `data/MANIFEST.toml`. If a command and this document
disagree, the command is right — ask the manifest directly:

```bash
python -m paper.manifest --chain      # what produced each role, and which panels use it
scripts/check_local_data.py           # what you have, and what supplies the rest
scripts/render_panels.py --list       # the panels, their env, and what each costs
```

---

## 1. Redraw everything — `figure_data`, ~40 MB

Every published panel has a table that fully determines it. No models, no GPU, no
structures, no cluster.

```bash
scripts/fetch_local_data.py --tier figure_data
scripts/render_panels.py --from-csv
```

This is the tier to hand someone who wants to check a number, restyle a figure, or
confirm that the plotted values match the paper.

### Everything from one directory

Both tiers resolve out of `local_data/`. Set `PEINT_PAPER_LOCAL_DATA_ONLY=1` to make that
binding rather than best-effort — see the README. Model checkpoints are included: they land
at `local_data/peint/model_checkpoints/`, which is where `paper_config` looks when the peint
repo does not have them.

## 2. Recompute the metrics — the archives, ~22 GB packed / ~151 GB unpacked

(~11 GB / ~53 GB if you skip the two rev1 structure trees, which no panel reads.)

```bash
scripts/fetch_local_data.py --tier full     # or the curl+tar loop in installation.md
scripts/render_panels.py                    # for a Zenodo record, which speaks no Hub
```

`installation.md` has the per-archive breakdown and the Zenodo recipe. Note that the model
checkpoints are more than half the packed download (~6.2 of ~11 GB) because weights compress at
1.08x while the sequence trees compress 8-18x — if you do not need to run a model, skip that one
archive and everything except the simulating panels still recomputes.

or, on a machine that already holds the source trees:

```bash
scripts/link_local_data.sh --apply    # symlinks, no copying
scripts/render_panels.py
```

What this actually recomputes: the conservation JSD, the 3Di JSD, both pLDDT ECDFs, the
generalization panels, the BLAST identity panel, the indel panels, and the ESM-IF panels.
Each reads shipped simulation output or shipped structures and derives its statistic fresh.
What it does **not** recompute is the simulation and the folding themselves — those are the
shipped artifacts, and re-running them is section 3.

### The one panel that is genuinely cold

`pcp_panels` re-runs AliSim over ~500 families × 3 models unless it finds a warm cache.
The caches are deliberately **not** distributed: protevo keys a cached computation on a
sha512 of its arguments *including absolute input paths*, so a cache is only valid at the
path where it was built. Copying one elsewhere produces a directory that will never be hit.

So you have two honest options, and `--replot` is the one most people want:

```bash
scripts/render_panels.py --only pcp_panels --from-csv   # ~20 s, from the shipped tables
scripts/render_panels.py --only pcp_panels              # 30-45 min warm, hours cold
```

A cold run needs `iqtree2` and MAFFT on `PATH`. It writes its output to `DERIVED_DIR`
rather than back into the input tree, so pointing at a read-only shared copy is fine.

## 3. Regenerate the shipped data — what produced what

`python -m paper.manifest --chain` prints this per role, kept next to the data rather than
in prose so it cannot drift. The shape of it:

| shipped role | produced by |
|---|---|
| `r1/simulations`, `r1/mafft_add` | `benchmarks/generate_all_results.py --out_path <r1>` |
| `r1/af2`, `r1/omegafold` | the same driver with `--use_af2` / `--include_plddt` |
| `r2/simulations` | the same driver with `--out_path <r2> --alisim_models WAG LG` |
| `r2/simulations/historian_progressive` | `benchmarks/historian_esmc_progressive.py` |
| `r2/omegafold` | `benchmarks/omegafold_peint_esmc.py` |
| `r2/af2` | `benchmarks/af2rank_peint_esmc.py` |
| `r1/3di`, `r2/3di` | `benchmarks/threedi_jsd_all_models.py` without `--skip-3di-generation` |
| `blast_sequences` | `blastp` against nr, run outside this repo |

The exact rev2 invocation, recovered from its own slurm log, is in the `r2_simulations`
role's `producer` field — including the checkpoint path and the shard list.

### Not regenerable from this deposit

- **PEINT checkpoints** now ship in the `full` tier (`peint_checkpoints`, 3.3 GB, four
  files), so a step that simulates or scores is covered by the deposit alone. What is still
  not regenerable is the training that produced them.
- **Ground-truth structures and a3m alignments** come from the trRosetta training set
  (~20 GB, third party). Only steps that fold or thread against experimental structures
  need them; every panel reads shipped output instead.
- **The rev1 structure trees** (`r1/af2`, `r1/omegafold`) are 1,089,012 files and ~99 GB.
  They now **ship**, as `r1_af2.tar.zst` and `r1_omegafold.tar.zst` (5.6 and 5.3 GB packed),
  and unpack into `local_data/r1/`. No panel reads them — the two summary CSVs they produce,
  750 KB combined, are what the pLDDT ECDFs actually consume — so they are there for
  inspection and citation, and skipping both leaves every figure reproducible.
- **The Figure 2 likelihood panel** needs a checkpoint and a GPU. Its inputs — the held-out
  transitions, aligned and unaligned — are deposited. The model repo's `_cache_peint` is not,
  because it has the same absolute-path keying problem as above and is valid only in place;
  it is a warm start, not an input, so the panel recomputes without it.
- **The ESM-MCMC spectrum** summarises a multi-day GPU study; its driver ships for
  provenance, its five result CSVs ship as data, and re-running it is not expected.

---

## Associating an intermediate with its chain

For any shipped role, `--chain` gives you the producer, the panels it feeds, and its size:

```bash
python -m paper.manifest --chain | grep -A4 r2_omegafold
```

Going the other way — from a figure back to its inputs:

```bash
scripts/check_local_data.py --panels threedi_jsd_boxplot
```

That is the association worth preserving: a predicted structure in `r2/omegafold` was
written by `omegafold_peint_esmc.py` from the leaves in `r2/mafft_add`, which
`generate_all_results.py` produced from the trees and root sequences in `sim/`, and it is
read by the OmegaFold pLDDT ECDF and by ESM-IF self-consistency. Every link in that
sentence is a field in the manifest.

## Two environments

The benchmarks span stacks that do not pin together: JAX (AF2Rank), PyTorch (PEINT,
OmegaFold, ESM-C), and the older `transformers` ProstT5 was pinned against.
`render_panels.py` subprocesses each panel with the right interpreter, so you only need to
set `PEINT_PAPER_PY_ESMC` and `PEINT_PAPER_PY_PROTEVO`.

The simulations need the ESM-C stack and the folding needs the JAX stack, and no environment
has both — which works because what crosses the boundary is **sequences in text files, not
models**. The folding benchmarks never construct a PEINT model; they read the simulated
sequences off disk. The one exception is `figure2_simulation`, which is the same script run
twice and joined by protevo's computation cache: the model load sits *inside* the cached
function, so on a cache hit it is never reached. `installation.md` has the full account,
including the one failure mode worth recognising — a cache miss in the folding pass surfaces
as `model type 'esmc' not recognized`, which means *miss*, not *broken install*.
