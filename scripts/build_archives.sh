#!/usr/bin/env bash
# Build the data-deposit archives from a staged local_data/ tree.
#
# Run this WHERE THE STAGED TREE LIVES, not on the machine that produced the data: on our
# cluster the producing account has no quota headroom for a second copy, and tarring from the
# shared account puts the archives on the shared account's quota where they belong.
#
#   scripts/build_archives.sh --root /scratch/users/spa-evolution-yss/peint_paper_data
#   scripts/build_archives.sh --root <staged> --out <upload-dir> --apply
#   scripts/build_archives.sh --root <staged> --only r2 --apply    # rebuild one archive
#
# Produces, in --out (default <root>/_upload):
#   figure_data/            loose, so the Hub can browse it
#   r1.tar.zst r2.tar.zst sim.tar.zst aux.tar.zst
#   MANIFEST.toml  README.md  CHECKSUMS.sha256
#
# Each archive unpacks relative to local_data/, so `tar -xf r1.tar.zst -C local_data/` is the
# whole instruction. Dry-run by default; never deletes the staged tree.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PEINT_PAPER_PY_ESMC:-${PEINT_PAPER_PY:-python3}}"
ROOT=""
OUT=""
APPLY=0
ONLY=""
LEVEL="${PEINT_PAPER_ZSTD_LEVEL:-19}"
THREADS="${PEINT_PAPER_ZSTD_THREADS:-8}"

# The header comment block, minus the shebang, is the help text.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --root)  ROOT="${2:?}"; shift ;;
    --out)   OUT="${2:?}"; shift ;;
    --apply) APPLY=1 ;;
    --only)  ONLY="${ONLY:+$ONLY }${2:?--only needs an archive name, or 'loose'}"; shift ;;
    --level) LEVEL="${2:?}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

[[ -n "$ROOT" ]] || { echo "--root is required (the staged local_data tree)" >&2; exit 2; }
[[ -d "$ROOT" ]] || { echo "no such directory: $ROOT" >&2; exit 2; }
ROOT="$(cd "$ROOT" && pwd)"
OUT="${OUT:-$ROOT/_upload}"

# pzstd parallelises; plain zstd is the fallback.
if command -v pzstd >/dev/null; then COMPRESS="pzstd -$LEVEL -p $THREADS"
elif command -v zstd >/dev/null; then COMPRESS="zstd -$LEVEL -T$THREADS"
else echo "need pzstd or zstd on PATH" >&2; exit 1; fi

PLAN="$("$PY" -m paper.manifest --archive-plan)"
[[ -n "$PLAN" ]] || { echo "empty archive plan from paper.manifest" >&2; exit 1; }

# --only narrows to named archives, or to "loose" for the untarred figure_data tier. Useful
# for rebuilding one archive after fixing a role, without re-tarring 46 GB.
if [[ -n "$ONLY" ]]; then
  PLAN="$(awk -F'\t' -v want=" $ONLY " '{ k=($1==""?"loose":$1); if (index(want, " " k " ")) print }' <<<"$PLAN")"
  [[ -n "$PLAN" ]] || { echo "no archive matched --only '$ONLY'" >&2; exit 2; }
fi

echo "staged tree  $ROOT"
echo "upload dir   $OUT"
echo "compressor   tar | $COMPRESS"
echo

run() { if [[ $APPLY -eq 1 ]]; then "$@"; else printf '    would: %s\n' "$*"; fi; }

# ---------------------------------------------------------------- loose roles
# Anything with no archive ships as plain files.
LOOSE="$(awk -F'\t' '$1=="" {print $3}' <<<"$PLAN")"
for rel in $LOOSE; do
  if [[ ! -e "$ROOT/$rel" ]]; then echo "  MISSING  $rel" >&2; exit 1; fi
  echo "  loose    $rel"
  run mkdir -p "$OUT/$(dirname "$rel")"
  run cp -r "$ROOT/$rel" "$OUT/$(dirname "$rel")/"
done

# ---------------------------------------------------------------- archives
for arc in $(awk -F'\t' '$1!="" {print $1}' <<<"$PLAN" | awk '!seen[$0]++'); do
  file="$(awk -F'\t' -v a="$arc" '$1==a {print $2; exit}' <<<"$PLAN")"
  mapfile -t paths < <(awk -F'\t' -v a="$arc" '$1==a {print $3}' <<<"$PLAN")

  for rel in "${paths[@]}"; do
    [[ -e "$ROOT/$rel" ]] || { echo "  MISSING  $rel (needed by $file)" >&2; exit 1; }
  done

  echo "  archive  $file  <- ${#paths[@]} path(s)"
  run mkdir -p "$OUT"
  # -C "$ROOT" so every member path is relative to local_data/.
  run tar --use-compress-program="$COMPRESS" -cf "$OUT/$file" -C "$ROOT" "${paths[@]}"
done

# ---------------------------------------------------------------- metadata
if [[ -n "$ONLY" ]]; then
  echo
  echo "Partial build (--only $ONLY): metadata and checksums not regenerated."
  exit 0
fi

echo "  metadata MANIFEST.toml, README.md"
run cp "$REPO_ROOT/data/MANIFEST.toml" "$OUT/MANIFEST.toml"
run cp "$REPO_ROOT/data/DATASET_CARD.md" "$OUT/README.md"

echo "  checksums CHECKSUMS.sha256"
if [[ $APPLY -eq 1 ]]; then
  ( cd "$OUT" && find . -type f ! -name CHECKSUMS.sha256 -printf '%P\n' \
      | sort | xargs -r sha256sum > CHECKSUMS.sha256 )
  echo
  ls -la "$OUT"
  echo
  echo "Verify with:  cd $OUT && sha256sum -c CHECKSUMS.sha256"
else
  printf '    would: sha256sum every file in %s\n' "$OUT"
fi

echo
if [[ $APPLY -eq 1 ]]; then
  echo "Built. The staged tree at $ROOT was not modified."
else
  echo "Nothing was written. Re-run with --apply."
fi
