#!/usr/bin/env bash
# Reproduce the ESM-MCMC-vs-PEINT reviewer-response study end to end:
# situating PEINT against protein-LM MCMC simulators (Bitbol Phylogeny-ESM2 and ESM lm-design),
# on quality (JSD / ESM-IF / OmegaFold), cost, and the evolution<->design energy spectrum, and
# producing the 3-panel summary figure (figures/output/figure_esm_mcmc_spectrum.png).
#
# PREREQUISITES
#   * A CUDA GPU, and the project conda env (fair-esm, torch + torch_geometric stack, biotite,
#     pandas/seaborn) with `omegafold` on PATH.
#   * The external data the figures read, reachable via the PEINT_PAPER_* paths in paper_config.py:
#       - trees / root_sequences / empirical_msas  (SIM_ROOT)
#       - ground-truth PDBs                         (GROUND_TRUTH_STRUCTURE_DIR)
#       - mafft_add/ + simulations/ + omegafold/    (RESULTS_DIR; the other models' MSAs + cached folds)
#       - the PEINT checkpoint                       (PEINT_PAPER_CHECKPOINT; needed for the proposer)
#   * ESM2 / ESM-IF / OmegaFold weights auto-download on first use; lm-design assets are fetched below.
#
# RUNTIME: many hours of GPU time (Bitbol sim ~6h incl. the 681-aa family; ESM-IF ~2h; proposer ~1h;
#   OmegaFold ~1h; lm-design sim+sweep ~2h). Set LIMIT=2 for a fast smoke test. Steps are ordered by
#   dependency; comment out any you don't need. Outputs land under figures/output/ (gitignored).
#
# Usage:  bash scripts/reproduce_esm_mcmc.sh            # full study
#         LIMIT=2 bash scripts/reproduce_esm_mcmc.sh    # 2-family smoke test
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LIMIT="${LIMIT:-8}"                                   # held-out families for the Bitbol/ESM-IF arms
PROPOSER_NFAM="${PROPOSER_NFAM:-30}"                  # hash-seeded families for the proposer experiment
# lm-design simulator families (the LIMIT=8 set minus the 681-aa 1bf2, whose distogram forward is slow):
LMDESIGN_FAMS="${LMDESIGN_FAMS:-1a2t_1_A 1acf_1_A 1amx_1_A 1aq6_1_A 1bai_1_A 1bja_1_A 1byu_1_B}"
if [ "$LIMIT" -lt 8 ]; then LMDESIGN_FAMS="1bja_1_A 1bai_1_A"; fi   # smoke-test subset

step() { echo; echo "===== [$(date +%H:%M:%S)] $* ====="; }

step "0/10  Fetch lm-design assets (distogram weights + n-gram stats)"
bash scripts/fetch_lmdesign_assets.sh

step "1/10  ESM-IF Approach-1 likelihoods for all models (cached CSV the evals below read)"
python -m benchmarks.esmif_validation --approach 1 --limit "$LIMIT"

step "2/10  Bitbol (ESM2-MCMC) simulator: evolve $LIMIT families down their trees (hamming mode)"
python -m benchmarks.esm_mcmc_simulation --target-mode hamming --limit "$LIMIT"

step "3/10  Evaluate the Bitbol leaves: JSD + ESM-IF vs PEINT / classical / Real"
python -m benchmarks.esm_mcmc_eval

step "4/10  Template-free structural check: OmegaFold pLDDT on the Bitbol leaves (shared leaf set)"
python -m benchmarks.esm_mcmc_omegafold

step "5/10  Proposer experiment: PEINT vs uniform under the lm-design energy (needs PEINT checkpoint)"
python -m benchmarks.esm_mcmc_proposer --n-families "$PROPOSER_NFAM" --n 16

step "6/10  Single-site cost: faithful lm-design single-site (acceptance + forwards/branch)"
python -m benchmarks.esm_mcmc_singlesite_cost

step "7/10  lm-design simulator arm: evolve with the full lm-design energy as accept/reject"
python -m benchmarks.esm_mcmc_lmdesign_simulate --families $LMDESIGN_FAMS

step "8/10  Evaluate the lm-design leaves: JSD + ESM-IF"
python -m benchmarks.esm_mcmc_lmdesign_eval

step "9/10  Map the evolution<->design spectrum: sweep the LM weight lambda"
python -m benchmarks.esm_mcmc_lmdesign_sweep

step "10/10 Summary figure (spectrum / endpoint conservation / cost)"
# Panels (a) and (b) are regenerated from the runs above; panel (c) uses the recorded per-method
# generation times (Bitbol/lm-design timings are in their sim CSVs' 'seconds' columns; the PEINT
# per-family timing is a separate measurement, documented in figures/figure_esm_mcmc_spectrum.py).
python -m figures.figure_esm_mcmc_spectrum

echo; echo "Done -> figures/output/figure_esm_mcmc_spectrum.png"
