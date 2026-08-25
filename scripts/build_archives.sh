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
# Like stage_shared_data.sh, --src-host lets the account holding the staged tree run this
# without a checkout: the archive plan and the two metadata files are fetched over ssh.
#   scripts/build_archives.sh --root <staged> --src-host akoehl@beren --apply
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
# Exported, not set per call: `python -m paper.manifest` otherwise only resolves when the
# working directory happens to be the repo root, and this script is meant to be run from
# anywhere -- including copied to another account on its own.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PEINT_PAPER_PY_ESMC:-${PEINT_PAPER_PY:-python3}}"
# Same remote-source options as stage_shared_data.sh, so the account holding the staged tree
# needs no checkout of its own: the plan and the two metadata files come over ssh.
SRC_HOST="${PEINT_PAPER_SRC_HOST:-}"
# The ssh hop switches *user*, not machine: /scratch is one NFS mount, so the source data is
# the same bytes from any node. Default to whichever host you are on, and accept a bare
# username for --src-host so nothing here is pinned to one login node.
THIS_HOST="$(hostname)"
SRC_REPO="${PEINT_PAPER_SRC_REPO:-/scratch/users/akoehl/peint-paper}"
SRC_PYTHON="${PEINT_PAPER_SRC_PYTHON:-/scratch/users/akoehl/conda/envs/peint-esmc/bin/python}"
SSH_CM=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=8h)

on_src() { if [[ -n "$SRC_HOST" ]]; then ssh -n "${SSH_CM[@]}" "$SRC_HOST" "$@"; else bash -c "$*" </dev/null; fi; }
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
    --src-host)   SRC_HOST="${2:?--src-host needs user@host}"; shift ;;
    --src-repo)   SRC_REPO="${2:?--src-repo needs a path}"; shift ;;
    --src-python) SRC_PYTHON="${2:?--src-python needs a path}"; shift ;;
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

manifest() {
  if [[ -n "$SRC_HOST" ]]; then on_src "cd '$SRC_REPO' && '$SRC_PYTHON' -m paper.manifest $*"
  else "$PY" -m paper.manifest "$@"; fi
}

# This file is a copy when --src-host is used; the manifest it reads is live. Warn on drift.
if [[ -n "$SRC_HOST" ]]; then
  _mine="$(sha256sum "${BASH_SOURCE[0]}" 2>/dev/null | cut -d' ' -f1)"
  _theirs="$(on_src "sha256sum '$SRC_REPO/scripts/build_archives.sh' 2>/dev/null | cut -d' ' -f1" || true)"
  if [[ -n "$_mine" && -n "$_theirs" && "$_mine" != "$_theirs" ]]; then
    echo "NOTE: this copy of build_archives.sh differs from $SRC_REPO's. Refresh with:" >&2
    echo "          ssh $SRC_HOST 'cat $SRC_REPO/scripts/build_archives.sh' > \"${BASH_SOURCE[0]}\"" >&2
    echo >&2
  fi
fi

# A bare username means "same host, other account" -- the only thing this hop is ever for.
[[ -n "$SRC_HOST" && "$SRC_HOST" != *@* ]] && SRC_HOST="$SRC_HOST@$THIS_HOST"

if ! PLAN="$(manifest --archive-plan 2>&1)"; then
  echo "Could not generate the archive plan${SRC_HOST:+ on $SRC_HOST}:" >&2
  echo "$PLAN" >&2
  exit 5
fi
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
if [[ -n "$SRC_HOST" ]]; then
  # `cat` over the existing ssh master; no checkout needed on this side.
  run_pipe() { if [[ $APPLY -eq 1 ]]; then on_src "cat '$1'" > "$2"; else printf '    would: ssh %s cat %s > %s\n' "$SRC_HOST" "$1" "$2"; fi; }
  run mkdir -p "$OUT"
  run_pipe "$SRC_REPO/data/MANIFEST.toml"    "$OUT/MANIFEST.toml"
  run_pipe "$SRC_REPO/data/DATASET_CARD.md"  "$OUT/README.md"
else
  run cp "$REPO_ROOT/data/MANIFEST.toml" "$OUT/MANIFEST.toml"
  run cp "$REPO_ROOT/data/DATASET_CARD.md" "$OUT/README.md"
fi

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
