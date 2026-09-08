#!/usr/bin/env bash
# Run the full from-checkpoint pipeline on a family list: simulate, fold, score.
#
#   export PEINT_PAPER_PY_ESMC=/path/to/envs/peint-esmc/bin/python
#   export PEINT_PAPER_PY_PEINT=/path/to/envs/peint-paper/bin/python
#   export HF_HOME=/path/to/hf_cache
#   scripts/run_from_checkpoint.sh --checkpoint local_data/peint/model_checkpoints/peint_esmc.ckpt
#
# Needs the peint_checkpoints, sim, aux and peint_transitions_* archives unpacked under
# local_data/, plus mafft and iqtree2 on PATH. Four steps, two environments:
#
#   1. simulate + conservation JSD        peint-esmc    GPU
#   2. fold + 3Di + pLDDT                 peint-paper   GPU   (sims are cache hits here)
#   3. held-out per-site likelihood       peint-esmc    GPU
#   4. time estimation                    peint-esmc    GPU
#
# Each step is cached, so re-running skips whatever is already done.
set -euo pipefail

CKPT="local_data/peint/model_checkpoints/peint_esmc.ckpt"
FAMILIES="data/example_families.json"
OUT="runs/example"
FIGS="out"
SUBSAMPLE=3
STEPS="1 2 3 4"

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CKPT="${2:?}"; shift ;;
    --families)   FAMILIES="${2:?}"; shift ;;
    --out)        OUT="${2:?}"; shift ;;
    --figures)    FIGS="${2:?}"; shift ;;
    --subsample)  SUBSAMPLE="${2:?}"; shift ;;
    --steps)      STEPS="${2:?}"; shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

: "${PEINT_PAPER_PY_ESMC:?set it to the peint-esmc interpreter}"
: "${PEINT_PAPER_PY_PEINT:?set it to the peint-paper interpreter}"
[[ -f "$CKPT" ]]     || { echo "no checkpoint at $CKPT" >&2; exit 1; }
[[ -f "$FAMILIES" ]] || { echo "no family list at $FAMILIES" >&2; exit 1; }
command -v mafft >/dev/null || { echo "mafft is not on PATH" >&2; exit 1; }
command -v iqtree2 >/dev/null || echo "WARNING: iqtree2 is not on PATH; the WAG/LG arms will fail" >&2

# omegafold is a console script in peint-paper/bin, which is not on PATH when the interpreter
# is invoked by absolute path. Put that bin first for step 2.
PROTEVO_BIN="$(dirname "$PEINT_PAPER_PY_PEINT")"

DATA=(--tree_dir       local_data/sim/trees_newick      # NOT sim/trees: different format
      --root_sequences_dir local_data/sim/root_sequences
      --real_sequences_dir local_data/sim/empirical_msas)

step() { grep -qw "$1" <<<"$STEPS"; }
run()  { echo; echo "=== $* ==="; }

if step 1; then
  run "1/4 simulate + conservation JSD  (peint-esmc)"
  "$PEINT_PAPER_PY_ESMC" -m benchmarks.generate_all_results \
      --families_path "$FAMILIES" --peint_checkpoint_path "$CKPT" \
      "${DATA[@]}" --out_path "$OUT" --include_conservation
fi

if step 2; then
  run "2/4 structures + 3Di  (peint-paper; simulations are cache hits)"
  PATH="$PROTEVO_BIN:$PATH" "$PEINT_PAPER_PY_PEINT" -m benchmarks.generate_all_results \
      --families_path "$FAMILIES" "${DATA[@]}" --out_path "$OUT" \
      --subsample_msa_size "$SUBSAMPLE" --include_plddt --include_3di
fi

if step 3; then
  run "3/4 held-out per-site likelihood  (peint-esmc)"
  # --num-processes is mpirun -np inside cherryml; the script probes and steps down if the
  # machine has fewer slots, so asking for 8 is safe.
  "$PEINT_PAPER_PY_ESMC" -m figures.figure2_ll_eval_esmc \
      --esmc-checkpoint "$CKPT" --families-path "$FAMILIES" \
      --num-processes 8 --out-dir "$FIGS"
fi

if step 4; then
  run "4/4 time estimation  (peint-esmc)"
  "$PEINT_PAPER_PY_ESMC" -m figures.figure2_time_estimation \
      --checkpoint "$CKPT" --num-families 5 --output-dir "$FIGS"
fi

echo
echo "Done."
echo "  simulations / structures / per-family metrics : $OUT"
echo "  likelihood + time-estimation panels           : $FIGS"
