#!/usr/bin/env bash
# Populate local_data/ with symlinks to the authoritative data trees on this machine,
# reproducing the layout a Zenodo user gets by unpacking the tarballs into local_data/.
#
# Dry-run by default: pass --apply to actually create links. This script ONLY ever
# creates symlinks. It never moves, copies over, or deletes anything, and it refuses to
# touch a path that already exists and is not a symlink we own.
#
# Usage:
#   scripts/link_local_data.sh              # show what would be done
#   scripts/link_local_data.sh --apply      # create the links
#   scripts/link_local_data.sh --check      # only verify existing links resolve
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATA="${PEINT_PAPER_LOCAL_DATA:-$REPO_ROOT/local_data}"
DATA_ROOT="${PEINT_PAPER_DATA_ROOT:-/scratch/users/akoehl/protein-evolution}"
SIM_ROOT="${PEINT_PAPER_SIM_ROOT:-/scratch/users/akoehl/old/protein-evolution/local_data/simulation/final_simulation_512_leaves/ratio_0-1_nucleus_1-0}"
GT_PDB="${PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR:-/scratch/users/matthew_liu/input_data/pdb}"

MODE=dry
case "${1:-}" in
  --apply) MODE=apply ;;
  --check) MODE=check ;;
  ""|--dry-run) MODE=dry ;;
  *) echo "usage: $0 [--apply|--check|--dry-run]" >&2; exit 2 ;;
esac

# link <relative path under LOCAL_DATA> <absolute target>
declare -a LINKS=(
  "r1|$DATA_ROOT/local_data/results_revision1"
  "r2|$DATA_ROOT/local_data/results_revision2_esmc"
  "sim|$SIM_ROOT"
  "a3m|$DATA_ROOT/input_data/a3m"
  "pdb|$GT_PDB"
  "annotations|$DATA_ROOT/local_data/generalization/annotations"
  # The benchmarks WRITE into these two, so they point at the warm caches this machine
  # has already built. Everything above is read-only input.
  "derived/_cache_protevo|$DATA_ROOT/_cache_protevo"
  "derived/_cache_benchmarking|$DATA_ROOT/_cache_benchmarking"
)

rc=0
for entry in "${LINKS[@]}"; do
  rel="${entry%%|*}"; target="${entry#*|}"
  link="$LOCAL_DATA/$rel"

  if [[ ! -e "$target" ]]; then
    printf '  MISSING TARGET  %-34s -> %s\n' "$rel" "$target"; rc=1; continue
  fi

  if [[ -L "$link" ]]; then
    current="$(readlink "$link")"
    if [[ "$current" == "$target" ]]; then
      [[ -e "$link" ]] && printf '  ok              %-34s -> %s\n' "$rel" "$target" \
                       || { printf '  BROKEN LINK     %-34s -> %s\n' "$rel" "$current"; rc=1; }
    else
      printf '  POINTS ELSEWHERE %-33s -> %s (want %s)\n' "$rel" "$current" "$target"; rc=1
    fi
    continue
  fi

  if [[ -e "$link" ]]; then
    # A real file or directory. Never clobber it.
    printf '  REAL PATH, SKIP %-34s (not a symlink; leaving it alone)\n' "$rel"; rc=1; continue
  fi

  case "$MODE" in
    apply)
      mkdir -p "$(dirname "$link")"
      ln -s "$target" "$link"          # no -f: refuses to overwrite, and never nests
      printf '  linked          %-34s -> %s\n' "$rel" "$target" ;;
    check)
      printf '  ABSENT          %-34s (run with --apply)\n' "$rel"; rc=1 ;;
    *)
      printf '  would link      %-34s -> %s\n' "$rel" "$target" ;;
  esac
done

echo
if [[ $rc -eq 0 ]]; then echo "local_data OK ($LOCAL_DATA)"; else echo "local_data has issues (see above)"; fi
exit $rc
