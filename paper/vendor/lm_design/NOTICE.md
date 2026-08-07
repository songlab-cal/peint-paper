# Vendored from facebookresearch/esm — lm-design (MIT)

These modules are copied (with minor edits) from the ESM repository's `examples/lm-design/utils`,
used to reproduce the **literal lm-design energy** as an accept/reject criterion in the ESM-MCMC
proposer experiment (`benchmarks/esm_mcmc_proposer.py`, `paper/lmdesign_energy.py`).

Source: https://github.com/facebookresearch/esm/tree/main/examples/lm-design
License: MIT (Meta Platforms, Inc.) — original copyright headers are preserved in each file.

Files and edits:
- `linear_projection.py` — `LinearProjectionDistogramModel`: a single learned affine projection
  (~11,898 params, frozen ESM2) from ESM2's 660 attention maps to a per-residue-pair distogram
  (18 Cβ-Cβ distance bins). Edit: removed an unused `omegaconf` import so it loads without the
  lm-design config stack.
- `loss.py` — `get_cce_loss` (masked categorical cross-entropy). Unmodified.
- `pdb_loader.py` — `get_coords6d` (+ `get_dihedrals`/`get_angles`) recomputes Cβ from N/CA/C and
  builds the target Cβ-Cβ distance matrix. Unmodified; only `get_coords6d` is used here.

Not vendored: the lm-design Designer/config/plotting stack, and `utils/ngram.py` (its n-gram KL is
reimplemented `nltk`-free in `paper/lmdesign_energy.py`). Pretrained weights and the n-gram
background stats are fetched by `scripts/fetch_lmdesign_assets.sh` into `data/lm_design/`.
