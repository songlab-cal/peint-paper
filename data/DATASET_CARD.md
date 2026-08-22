---
license: cc-by-nc-4.0
pretty_name: PEINT paper — figure data and simulation results
tags:
  - protein-evolution
  - phylogenetics
  - protein-language-model
  - reproducibility
configs: []
---

<!--
Template for the Hugging Face dataset repo that accompanies the PEINT paper.
Copy this to the dataset repo as README.md, fill in every <FILL:...>, and mint the DOI
LAST — a DOI locks the repo, after which renaming, deleting, or changing visibility all
require a support request.
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

| tier | size | what you can do |
|---|---|---|
| `figure_data/` | ~10 MB | re-render **every** panel exactly. No models, no GPU, no other downloads. |
| the `*.tar.zst` archives | ~4 GB | recompute the metrics from the simulated alignments themselves. |

## Quick start

```bash
git clone <FILL: peint-paper repo>
cd peint-paper && pip install -e .

python scripts/fetch_local_data.py --tier figure_data --repo <FILL: org/repo>
python scripts/render_panels.py --list
python scripts/render_panels.py --group main
```

For the recompute tier, `--tier full` instead, and see the repo README for the two conda
environments involved.

## Contents

```
figure_data/     per-panel tables — the values actually plotted
r1.tar.zst       revision 1: classical baselines, PEINT-ESM2, real
                   mafft_add/     realigned real + PEINT leaves (shared column frame)
                   simulations/   per-model simulated MSAs + Historian reconstructions
                   3di/           ProstT5 structural-alphabet sequences per model
                   *_plddt.csv    per-family AF2Rank / OmegaFold pLDDT
r2.tar.zst       revision 2: the ESM-C rerun (its own mafft frame)
                   historian_esmc/{refine,norefine}/{events,reconstructions}
sim.tar.zst      trees, root sequences, empirical + simulated MSAs (1,093 families)
aux.tar.zst      Pfam/SCOPe/ECOD labels, per-site rates, family split lists
omegafold_structures.tar.zst   optional: raw ESM-C OmegaFold PDBs
```

Each archive unpacks relative to `local_data/`, so `tar -xf X.tar.zst -C local_data/` is
the whole instruction. `fetch_local_data.py` does this for you.

Bulk roles are tarred rather than stored as loose files on purpose: several hold tens of
thousands of small per-family text files, which would exceed the Hub's 10,000-entries-per-folder
limit and turn a download into thousands of requests. `figure_data/` stays loose so it can
be browsed here.

## What is deliberately **not** here

These are third-party or too large to redistribute; the code fetches them:

- **trRosetta training set** — the input a3m alignments and ground-truth PDBs
  (`https://files.ipd.uw.edu/pub/trRosetta/training_set.tar.gz`, ~20 GB).
- **Pfam-A HMMs, ECOD, CATH, SCOPe** — fetched on demand by `paper.generalization`. The
  *derived* labels are in `aux.tar.zst`, so the generalization panels work offline.
- **AlphaFold weights, TM-align, ProstT5, ProteinGym** — see the repo's `scripts/fetch_*.sh`.
- **Model checkpoints** — <FILL: where PEINT checkpoints live, if released>.
- **Raw AF2Rank/OmegaFold structure predictions for revision 1** — ~100 GB across ~835,000
  files. Only the per-family pLDDT summaries they produce are included.

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
