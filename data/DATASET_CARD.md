<!--
Template for the data deposit that accompanies the PEINT paper. `build_archives.sh` copies
this file into the upload directory as README.md, so edit it HERE, not there.

Fill in every <FILL:...> before publishing, and mint the DOI LAST — a DOI locks the record,
after which renaming, deleting, or changing visibility all require a support request.

Zenodo takes its own metadata (title, authors, license, DOI) from the deposit form, not from
this file. This is the landing-page prose only.
-->

# PEINT paper — figure data and simulation results

Everything needed to reproduce the figures in <FILL: paper title / preprint link>.

Code: <FILL: link to the peint-paper repo> (commit `<FILL>`)
Model library: <FILL: link to the peint repo> (commit `<FILL>`)

## What this is

The paper evaluates PEINT, a learned protein-evolution simulator, against classical
substitution models (WAG, LG, LG4X, LG+C60, LG+S256) and against real sequence data, on
553 held-out protein families. This deposit holds the simulation outputs and the derived
per-family metrics behind every published panel.

It is organised in two tiers, because most people want the first one.

| tier | download | unpacked | what you can do |
|---|---|---|---|
| `figure_data.tar.zst` | ~5 MB | 21 MB | re-render **every** panel exactly. No models, no GPU, nothing else. |
| \+ `r1`, `r2`, `sim`, `aux`, `peint_checkpoints` | ~11 GB | ~53 GB | recompute the metrics from the simulated alignments themselves. |
| \+ `r1_af2`, `r1_omegafold` | ~22 GB | ~151 GB | inspect the raw revision-1 structure predictions. No panel reads them. |

## Quick start

```bash
git clone <FILL: peint-paper repo>
cd peint-paper
```

Then fetch the data. From this record, each archive unpacks relative to `local_data/`:

```bash
ZENODO=<FILL: https://zenodo.org/records/NNNNNNN/files>
mkdir -p local_data
curl -L -O "$ZENODO/figure_data.tar.zst?download=1"
tar --use-compress-program=unzstd -xf figure_data.tar.zst -C local_data/
```

Add `aux`, `sim`, `r1`, `r2`, `peint_checkpoints` the same way for the recompute tier. The two
structure archives take `-C local_data/r1/` instead — see the note under the file table below.
Verify first if you like: `curl -L -O "$ZENODO/CHECKSUMS.sha256?download=1"` then
`sha256sum -c CHECKSUMS.sha256 --ignore-missing`.

The repo automates all of that — download, checksum, unpack — given the record id:

```bash
python scripts/fetch_local_data.py --tier figure_data --record <FILL: record id>
python scripts/fetch_local_data.py --tier full        --record <FILL: record id>
```

Then:

```bash
python scripts/check_local_data.py          # what is present; what supplies the rest
python scripts/render_panels.py --list
python scripts/render_panels.py --group main
```

Installing the code is its own job — two conda environments, because the folding stack and the
ESM-C stack do not coexist. The repo's `installation.md` is the tested recipe.

## Contents

| file | unpacked | files | what it is |
|---|---|---|---|
| `figure_data/` | 25 MB | 20 | per-panel tables — the values actually plotted |
| `r1.tar.zst` | 19.9 GB | 74,440 | revision 1: classical baselines, PEINT-ESM2, real |
| `r2.tar.zst` | 25.2 GB | 75,904 | revision 2: the ESM-C rerun, its own mafft frame |
| `sim.tar.zst` | 0.5 GB | 6,010 | trees, root sequences, empirical + simulated MSAs |
| `aux.tar.zst` | 0.1 GB | 30,117 | Pfam/SCOPe/ECOD labels, per-site rates, split lists |
| `peint_checkpoints.tar.zst` | 6.9 GB | 5 | the PEINT / VEP model checkpoints |
| `r1_af2.tar.zst` | 52 GB | 481,316 | raw rev1 AF2Rank structures (unpacks into `local_data/r1/`) |
| `r1_omegafold.tar.zst` | 46 GB | 607,696 | raw rev1 OmegaFold structures (unpacks into `local_data/r1/`) |

```
r1/  mafft_add/     realigned real + PEINT leaves (shared column frame)
     simulations/   per-model simulated MSAs + the three Historian reconstructions
     3di/           ProstT5 structural-alphabet sequences per model
     *.csv          per-family AF2Rank / OmegaFold pLDDT summaries
r2/  mafft_add/, 3di/, af2/, omegafold/
     simulations/historian_progressive/  ESM-C Historian, refine and norefine
aux  annotations/   derived Pfam domain + family labels
     splits/        the 14,498 train / 553 held-out family lists
     peint/local_data/output_site_rates_dir/   4-category per-site rates (beside the held-out transitions)
```

Each archive unpacks relative to `local_data/`, so `tar -xf X.tar.zst -C local_data/` is
the whole instruction. `fetch_local_data.py` does this for you, and verifies every file
against `CHECKSUMS.sha256` before unpacking.

**Two exceptions:** `r1_af2.tar.zst` and `r1_omegafold.tar.zst` were tarred from their source
directories, so their members begin at `af2/` and `omegafold/` and they unpack into
`local_data/r1/` instead. Each archive's `unpack_into` field in `MANIFEST.toml` records this,
and `fetch_local_data.py` reads it; it only matters if you unpack by hand:

```
tar -xf r1_af2.tar.zst -C local_data/r1/
```

`MANIFEST.toml` is the machine-readable inventory: one entry per role, with its path, the
archive that carries it, a probe file, measured sizes, and the panels that need it. Roles
marked `in_repo` ship with the code instead and are deliberately absent here.

Bulk roles are tarred rather than stored as loose files on purpose: several hold tens of
thousands of small per-family text files. A Zenodo record is also a flat list of files with a
100-file limit, so a directory cannot be deposited as such: everything is an archive.

## What is deliberately **not** here

These are third-party or too large to redistribute; the code fetches them:

- **trRosetta training set** — the input a3m alignments and ground-truth PDBs
  (`https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz`, ~20 GB).
- **Pfam-A HMMs, ECOD, CATH, SCOPe** — fetched on demand by `paper.generalization`. The
  *derived* labels are in `aux.tar.zst`, so the generalization panels work offline.
- **AlphaFold weights, TM-align, ProstT5, ProteinGym** — see the repo's `scripts/fetch_*.sh`.
- **The held-out transition trees and the `peint` evaluation cache** — declared in
  `MANIFEST.toml` as `tier = "on_request"`. The cache is keyed on absolute paths, so a copy is
  only valid where it was built; see the repo's `REPRODUCING.md`.

Model checkpoints and the raw revision-1 structure trees **are** included — see the archive
table above. No panel reads the structure trees (the two per-family pLDDT summaries they
produce are what the figures consume, and those live in `r1.tar.zst`); they are deposited so
the structures behind those summaries can be inspected and cited. Skipping both saves 11 GB of
download and 99 GB on disk, and leaves every figure reproducible.

## Reproducibility, honestly

Not every panel can be recomputed from this deposit, and it is better to say which:

- **All panels re-render** from `figure_data/`.
- **Most recompute** from the archives: the conservation JSD panels, the 3Di panel, the
  pLDDT ECDFs, the generalization panels, and two of the three arms of the indel panels.
- **These do not**, and the reasons are structural rather than fixable:
  - The Figure 2 per-site likelihood panel needs a separate transition dataset and
    evaluation cache from the model repo, plus a GPU.
  - The parent-child-pair panels need a cold AliSim re-simulation (iqtree2 + MAFFT).
  - The ESM-MCMC spectrum figure summarises a multi-day GPU study; its driver script is in
    the repo for provenance.
  - The variant-effect panels rest on ProteinGym zero-shot scores for ~13 external models
    that we do not redistribute; only the resulting per-assay Spearman tables are here.
  - The ESM-C arm of the indel panels is shipped as Historian event tables — regenerating
    it is a 64-process MAFFT + Historian job.

## Citation

<FILL: BibTeX for the paper, then the dataset DOI once minted>

## Licence

<FILL: confirm>. Code in the companion repo is licensed separately; the vendored
`paper/vendor/lm_design/` is MIT (see its `NOTICE.md`).
