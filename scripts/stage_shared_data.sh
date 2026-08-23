#!/usr/bin/env bash
# Copy the paper's data roles to a shared cluster account, or anywhere else.
#
# Why this exists in this shape: on our cluster the producing account is mode 700, so nothing
# can pull out of it, and the shared account's directory is not group-writable, so nothing can
# push into it as us. The one arrangement that works is rsync over ssh -- the local side reads
# as us, the remote side writes as the shared account, which is also what makes the copy count
# against the shared account's quota instead of ours.
#
#   scripts/stage_shared_data.sh                 # dry run: show every file that would move
#   scripts/stage_shared_data.sh --plan          # just the role table, no filesystem access
#   scripts/stage_shared_data.sh --apply
#   scripts/stage_shared_data.sh --apply --only r2_omegafold
#   scripts/stage_shared_data.sh --verify        # confirm the destination is complete
#
# SAFETY. This script never writes to, moves, or deletes anything on the source side; source
# paths appear only as rsync sources. --delete and --remove-source-files are rejected by an
# explicit guard, the destination is checked against every source root before anything runs,
# and a dry run is what you get if you forget --apply.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PEINT_PAPER_PY_ESMC:-${PEINT_PAPER_PY:-python3}}"
DEST="${PEINT_PAPER_SHARED_DEST:-spa-evolution-yss@beren:/scratch/users/spa-evolution-yss/peint_paper_data}"

MODE=dry
ONLY=""
RESUME=0

# The header comment block, minus the shebang, is the help text.
usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)   MODE=apply ;;
    --dry-run) MODE=dry ;;
    --verify)  MODE=verify ;;
    --plan)    MODE=plan ;;
    --resume)  RESUME=1 ;;
    --dest)    DEST="${2:?--dest needs a value}"; shift ;;
    --only)    ONLY="${2:?--only needs a role name}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- destination
# "host:/path" or a plain local "/path".
if [[ "$DEST" == *:* && "$DEST" != /* ]]; then
  DEST_HOST="${DEST%%:*}"; DEST_PATH="${DEST#*:}"
else
  DEST_HOST=""; DEST_PATH="$DEST"
fi
[[ "$DEST_PATH" == /* ]] || { echo "destination path must be absolute: $DEST_PATH" >&2; exit 2; }
DEST_PATH="${DEST_PATH%/}"

remote() {  # run a command on whichever side the destination lives
  if [[ -n "$DEST_HOST" ]]; then ssh -o BatchMode=yes "$DEST_HOST" "$@"; else bash -c "$*"; fi
}

# ---------------------------------------------------------------- the plan
# Tab-separated: role, kind(dir|file), destination-relative path, source, comma-separated
# excludes. One source of truth -- data/MANIFEST.toml -- for this, the symlink farm, the
# archive builder and the downloader.
PLAN="$("$PY" -m paper.manifest --rsync-plan)"
[[ -n "$PLAN" ]] || { echo "empty transfer plan from paper.manifest" >&2; exit 1; }
if [[ -n "$ONLY" ]]; then
  PLAN="$(awk -F'\t' -v r="$ONLY" '$1==r' <<<"$PLAN")"
  [[ -n "$PLAN" ]] || { echo "no role named '$ONLY' in the shipped tiers" >&2; exit 2; }
fi

# Refuse a destination that is inside a source, or a source inside the destination. This is
# the guard that matters: everything else this script does is a read.
while IFS=$'\t' read -r role kind rel src excl; do
  [[ -n "${src:-}" ]] || continue
  srcdir="$src"; [[ "$kind" == file ]] && srcdir="$(dirname "$src")"
  if [[ -z "$DEST_HOST" ]]; then
    case "$DEST_PATH/" in "$srcdir"/*) echo "REFUSING: destination $DEST_PATH is inside source $srcdir" >&2; exit 3 ;; esac
    case "$srcdir/" in "$DEST_PATH"/*) echo "REFUSING: source $srcdir is inside destination $DEST_PATH" >&2; exit 3 ;; esac
  fi
done <<<"$PLAN"

if [[ "$MODE" == plan ]]; then
  "$PY" -m paper.manifest --list --tier figure_data
  echo
  "$PY" -m paper.manifest --list --tier full
  echo
  echo "destination: ${DEST_HOST:+$DEST_HOST:}$DEST_PATH"
  exit 0
fi

# ---------------------------------------------------------------- rsync flags
# --no-o --no-g is the whole point: the receiving rsync stamps its own ownership, so the bytes
# land against the shared account's quota rather than ours.
# --mkpath creates intermediate destination directories (rsync >= 3.2.3).
# --partial-dir makes an interrupted role resume instead of restarting.
BASE=(-a --no-o --no-g --chmod=D755,F644 --mkpath
      --partial --partial-dir=.rsync-partial --human-readable)

for f in "${BASE[@]}"; do
  case "$f" in
    --delete*|--remove-source-files|--force)
      echo "REFUSING: destructive rsync flag in BASE: $f" >&2; exit 3 ;;
  esac
done

case "$MODE" in
  apply)  FLAGS=("${BASE[@]}" --info=progress2) ;;
  dry)    FLAGS=("${BASE[@]}" -n --itemize-changes) ;;
  verify) FLAGS=("${BASE[@]}" -n --itemize-changes) ;;
esac

rc=0
n_roles=0

# ---------------------------------------------------------------- transfer
transfer() {
  local role="$1" kind="$2" rel="$3" src="$4" excl="$5"
  local -a exargs=()
  if [[ -n "$excl" ]]; then
    local IFS=','
    for e in $excl; do exargs+=(--exclude="$e"); done
  fi

  if [[ ! -e "$src" ]]; then
    printf '  MISSING SOURCE  %-22s %s\n' "$role" "$src"; rc=1; return
  fi

  local target="${DEST_HOST:+$DEST_HOST:}$DEST_PATH/$rel/"
  local -a srcargs
  if [[ "$kind" == dir ]]; then srcargs=("$src/"); else srcargs=("$src"); fi

  # A destination that already holds this role, and that we are not explicitly resuming into,
  # is worth stopping for -- it usually means a second run against a changed manifest.
  # The test has to be per-kind: a directory role owns its whole directory, but a file role
  # owns only its own file and shares the directory with sibling roles (r1_structure_tables
  # lands in r1/, right next to r1/simulations).
  if [[ "$MODE" == apply && $RESUME -eq 0 ]]; then
    local occupied=""
    if [[ "$kind" == dir ]]; then
      remote "test -d '$DEST_PATH/$rel' && find '$DEST_PATH/$rel' -mindepth 1 -print -quit | grep -q ." \
        >/dev/null 2>&1 && occupied="$rel/"
    else
      remote "test -e '$DEST_PATH/$rel/$(basename "$src")'" \
        >/dev/null 2>&1 && occupied="$rel/$(basename "$src")"
    fi
    if [[ -n "$occupied" ]]; then
      printf '  ALREADY THERE   %-22s %s (pass --resume to continue into it)\n' "$role" "$occupied"
      rc=1; return
    fi
  fi

  printf '  %-8s %-22s %s\n' "$MODE" "$role" "$rel"
  # --mkpath makes rsync announce "created directory" even under -n, which reads alarmingly
  # in a dry run that writes nothing. Drop those lines; keep every real itemized change.
  if [[ "$MODE" == dry ]]; then
    rsync "${FLAGS[@]}" "${exargs[@]}" "${srcargs[@]}" "$target" | grep -v '^created ' || true
  else
    rsync "${FLAGS[@]}" "${exargs[@]}" "${srcargs[@]}" "$target"
  fi
}

# ---------------------------------------------------------------- verify
verify_role() {
  local role="$1" kind="$2" rel="$3" src="$4" excl="$5"
  local -a exargs=()
  if [[ -n "$excl" ]]; then
    local IFS=','
    for e in $excl; do exargs+=(--exclude="$e"); done
  fi
  local target="${DEST_HOST:+$DEST_HOST:}$DEST_PATH/$rel/"
  local -a srcargs
  if [[ "$kind" == dir ]]; then srcargs=("$src/"); else srcargs=("$src"); fi

  # 1. Itemized dry run must be silent: every file present, same size, same mtime.
  local diff
  diff="$(rsync "${FLAGS[@]}" "${exargs[@]}" "${srcargs[@]}" "$target" \
          | grep -v '^\.rsync-partial' | grep -v '^$' || true)"
  if [[ -n "$diff" ]]; then
    printf '  DIFFERS   %-22s\n' "$role"
    sed 's/^/      /' <<<"$diff" | head -20
    rc=1
    return
  fi

  # 2. File counts, as an independent cross-check that rsync and the filesystem agree.
  # Skipped for roles with excludes, where the two sides are not meant to match.
  if [[ "$kind" == dir && -z "$excl" ]]; then
    local a b
    a="$(find "$src" -type f 2>/dev/null | wc -l)"
    b="$(remote "find '$DEST_PATH/$rel' -type f 2>/dev/null | wc -l")"
    if [[ "$a" != "$b" ]]; then
      printf '  COUNT     %-22s src=%s dest=%s\n' "$role" "$a" "$b"; rc=1; return
    fi
  fi
  printf '  ok        %-22s\n' "$role"
}

echo "source plan from data/MANIFEST.toml"
echo "destination ${DEST_HOST:+$DEST_HOST:}$DEST_PATH"
echo
while IFS=$'\t' read -r role kind rel src excl; do
  [[ -n "${role:-}" ]] || continue
  n_roles=$((n_roles + 1))
  if [[ "$MODE" == verify ]]; then verify_role "$role" "$kind" "$rel" "$src" "${excl:-}"
  else transfer "$role" "$kind" "$rel" "$src" "${excl:-}"; fi
done <<<"$PLAN"

echo
if [[ $rc -eq 0 ]]; then
  echo "$n_roles entries OK ($MODE)"
  [[ "$MODE" == dry ]] && echo "Nothing was written. Re-run with --apply, then --verify."
else
  echo "$MODE finished with problems (see above)"
fi
exit $rc
