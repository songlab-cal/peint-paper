#!/usr/bin/env bash
# Populate local_data/ with symlinks to the authoritative data trees on this machine,
# reproducing the layout a downloader gets by unpacking the archives into local_data/.
#
# The role list comes from data/MANIFEST.toml, so this cannot drift from what is staged,
# archived and fetched. That makes the links finer-grained than they used to be: one per
# role (local_data/r1/simulations, local_data/r1/3di, ...) rather than one per revision.
# The point of the change is truthfulness -- a single local_data/r1 link exposed r1/af2 and
# r1/omegafold, 1.09 M files that the deposit deliberately excludes, so --check passed on
# data no downloader will ever have.
#
# Dry-run by default: pass --apply to actually create links. This script ONLY ever creates
# symlinks. It never moves, copies over, or deletes anything, and it refuses to touch a path
# that already exists and is not a symlink pointing where the manifest says.
#
# Usage:
#   scripts/link_local_data.sh              # show what would be done
#   scripts/link_local_data.sh --apply      # create the links
#   scripts/link_local_data.sh --check      # only verify existing links resolve
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCAL_DATA="${PEINT_PAPER_LOCAL_DATA:-$REPO_ROOT/local_data}"
PY="${PEINT_PAPER_PY_ESMC:-${PEINT_PAPER_PY:-python3}}"

MODE=dry
case "${1:-}" in
  --apply) MODE=apply ;;
  --check) MODE=check ;;
  ""|--dry-run) MODE=dry ;;
  *) echo "usage: $0 [--apply|--check|--dry-run]" >&2; exit 2 ;;
esac

# rel|target, one per directory-backed role in the shipped tiers.
LINKS="$("$PY" -m paper.manifest --links)"
[[ -n "$LINKS" ]] || { echo "empty link plan from paper.manifest" >&2; exit 1; }

# The benchmarks WRITE into these two, so they point at the warm caches this machine already
# has. They are not deposit roles -- protevo cache keys hash absolute input paths, so a cache
# is worthless anywhere but where it was built -- which is why they are appended here rather
# than declared in the manifest.
DATA_ROOT="${PEINT_PAPER_DATA_ROOT:-/scratch/users/akoehl/protein-evolution}"
LINKS+=$'\n'"derived/_cache_protevo|$DATA_ROOT/_cache_protevo"
LINKS+=$'\n'"derived/_cache_benchmarking|$DATA_ROOT/_cache_benchmarking"

rc=0
superseded=()

while IFS='|' read -r rel target; do
  [[ -n "${rel:-}" ]] || continue
  link="$LOCAL_DATA/$rel"

  if [[ ! -e "$target" ]]; then
    printf '  MISSING TARGET  %-34s -> %s\n' "$rel" "$target"; rc=1; continue
  fi

  # A coarse link from the previous layout (local_data/r1 -> the whole rev1 tree) still
  # resolves every role beneath it, so this is not an error -- but it also exposes r1/af2 and
  # r1/omegafold, 1.09 M files the deposit excludes, which means a --check against local_data
  # passes on data no downloader will have. Worth saying; never worth fixing behind your back.
  via=""
  parent="$(dirname "$rel")"
  while [[ "$parent" != "." && "$parent" != "/" ]]; do
    if [[ -L "$LOCAL_DATA/$parent" ]]; then
      superseded+=("$LOCAL_DATA/$parent"); via=" (via $parent)"; break
    fi
    parent="$(dirname "$parent")"
  done

  # Some roles are sourced from inside the repo (figure_data, vep, output_site_rates_dir);
  # there is nothing to link, the data is already where the figures look for it.
  if [[ -e "$link" ]] && [[ "$(readlink -f "$link")" == "$(readlink -f "$target")" ]]; then
    printf '  in place        %-34s%s\n' "$rel" "$via"; continue
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
done <<<"$LINKS"

echo
if [[ ${#superseded[@]} -gt 0 ]]; then
  # Deduplicate, then print an exact command. Removing a symlink cannot touch its target, but
  # this script still will not do it for you.
  printf '%s\n' "${superseded[@]}" | sort -u | {
    echo "These coarse links predate the per-role layout. Everything still resolves, but"
    echo "they expose subtrees the deposit excludes. To match it exactly, remove the LINKS"
    echo "(never their targets) and re-run with --apply:"
    while read -r p; do echo "    rm '$p'"; done
  }
  echo
fi
if [[ $rc -eq 0 ]]; then echo "local_data OK ($LOCAL_DATA)"; else echo "local_data has issues (see above)"; fi
exit $rc
