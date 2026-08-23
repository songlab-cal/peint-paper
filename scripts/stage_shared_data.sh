#!/usr/bin/env bash
# Copy the paper's data roles to a shared cluster account, or anywhere else.
#
# Two directions, because which one you can use depends on which account has a password.
# Both exist for the same reason: the source tree is mode 700 so the shared account cannot
# read it directly, and the shared directory is mode 755 owned by that account so we cannot
# write it directly. Whichever way round, rsync-over-ssh is what puts a reader on the source
# side and a writer on the destination side -- and the writer's identity is what decides
# whose quota the bytes land on.
#
#   PULL (run as the shared account; the source account has the password)
#     scripts/stage_shared_data.sh --src-host akoehl@beren #         --dest /scratch/users/spa-evolution-yss/peint_paper_data --apply
#     The plan is fetched over ssh from the source repo, so this side needs no python,
#     no conda and no checkout -- just this one file. --src-python names the interpreter
#     used over there (an ssh login shell will not have your conda env on PATH).
#
#   PUSH (run as yourself; the shared account accepts your key)
#     scripts/stage_shared_data.sh --apply
#
#   Either way:
#     scripts/stage_shared_data.sh --plan          # role table, touches nothing
#     scripts/stage_shared_data.sh                 # dry run: every file that would move
#     scripts/stage_shared_data.sh --verify        # confirm the destination is complete
#     scripts/stage_shared_data.sh --only r2_omegafold --apply
#
# SAFETY. This script never writes to, moves, or deletes anything on the source side; source
# paths appear only as rsync sources. --delete and --remove-source-files are rejected by an
# explicit guard, the destination is checked against every source root before anything runs,
# and a dry run is what you get if you forget --apply.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PEINT_PAPER_PY_ESMC:-${PEINT_PAPER_PY:-python3}}"
SHARED_DIR="/scratch/users/spa-evolution-yss/peint_paper_data"
DEST="${PEINT_PAPER_SHARED_DEST:-}"

MODE=dry
ONLY=""
RESUME=0
SRC_HOST="${PEINT_PAPER_SRC_HOST:-}"
SRC_REPO="${PEINT_PAPER_SRC_REPO:-/scratch/users/akoehl/peint-paper}"
# An ssh command gets a login shell, not your interactive environment, so the default python
# there is whatever is first on PATH -- here a 3.8 that predates tomllib and has no tomli.
# Name the interpreter explicitly rather than hoping.
SRC_PYTHON="${PEINT_PAPER_SRC_PYTHON:-/scratch/users/akoehl/conda/envs/peint-esmc/bin/python}"

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
    --src-host) SRC_HOST="${2:?--src-host needs user@host}"; shift ;;
    --src-repo) SRC_REPO="${2:?--src-repo needs a path}"; shift ;;
    --src-python) SRC_PYTHON="${2:?--src-python needs a path}"; shift ;;
    --only)    ONLY="${2:?--only needs a role name}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

# ---------------------------------------------------------------- destination
# Pull mode writes locally (we are already the receiving account); push mode has to name the
# receiving account explicitly. Either way the bytes end up owned by whoever runs the writing
# half of the rsync, which is the whole point.
if [[ -z "$DEST" ]]; then
  if [[ -n "$SRC_HOST" ]]; then DEST="$SHARED_DIR"
  else DEST="spa-evolution-yss@beren:$SHARED_DIR"; fi
fi

# "host:/path" or a plain local "/path".
if [[ "$DEST" == *:* && "$DEST" != /* ]]; then
  DEST_HOST="${DEST%%:*}"; DEST_PATH="${DEST#*:}"
else
  DEST_HOST=""; DEST_PATH="$DEST"
fi
[[ "$DEST_PATH" == /* ]] || { echo "destination path must be absolute: $DEST_PATH" >&2; exit 2; }
DEST_PATH="${DEST_PATH%/}"

# One authenticated ssh connection, reused by every rsync and every remote check. Without
# multiplexing, password auth would prompt once per role, which for 18 roles plus their
# occupancy checks is unusable. ControlPersist keeps the master alive between invocations,
# so the password is typed once for the whole transfer.
SSH_CM=(-o ControlMaster=auto -o "ControlPath=$HOME/.ssh/cm-%r@%h:%p" -o ControlPersist=8h)
SSH="ssh ${SSH_CM[*]}"

[[ -n "$SRC_HOST" && -n "$DEST_HOST" ]] && {
  echo "remote source and remote destination at once is not supported" >&2; exit 2; }

# -n is load-bearing, not hygiene: these run inside a loop reading the plan from stdin, and
# an ssh without it swallows the remaining lines. That silently transferred 1 of 5 files in
# the first pull-mode run.
on_src()  { if [[ -n "$SRC_HOST" ]];  then ssh -n "${SSH_CM[@]}" "$SRC_HOST"  "$@"; else bash -c "$*" </dev/null; fi; }
on_dest() { if [[ -n "$DEST_HOST" ]]; then ssh -n "${SSH_CM[@]}" "$DEST_HOST" "$@"; else bash -c "$*" </dev/null; fi; }

# Whichever side is remote, that is the one needing an open connection.
REMOTE_SIDE="${SRC_HOST:-$DEST_HOST}"

# ---------------------------------------------------------------- the plan
# Tab-separated: role, kind(dir|file), destination-relative path, source, comma-separated
# excludes. One source of truth -- data/MANIFEST.toml -- for this, the symlink farm, the
# archive builder and the downloader.
# In pull mode the manifest lives on the source side, behind a mode-700 directory, so the
# plan is generated there and streamed back. That is what lets the pulling account run this
# script with no checkout, no conda and no python of its own.
if [[ -n "$SRC_HOST" ]]; then
  if ! PLAN="$(on_src "cd '$SRC_REPO' && '$SRC_PYTHON' -m paper.manifest --rsync-plan" 2>&1)"; then
    cat >&2 <<EOF
Could not generate the transfer plan on $SRC_HOST:

$PLAN

Tried: cd $SRC_REPO && $SRC_PYTHON -m paper.manifest --rsync-plan
Point --src-python at an interpreter there that can read TOML (3.11+, or 3.10 with tomli),
and --src-repo at the peint-paper checkout if it is not where I looked.
EOF
    exit 5
  fi
else
  PLAN="$("$PY" -m paper.manifest --rsync-plan)"
fi
[[ -n "$PLAN" ]] || { echo "empty transfer plan from paper.manifest" >&2; exit 1; }
if [[ -n "$ONLY" ]]; then
  PLAN="$(awk -F'\t' -v r="$ONLY" '$1==r' <<<"$PLAN")"
  [[ -n "$PLAN" ]] || { echo "no role named '$ONLY' in the shipped tiers" >&2; exit 2; }
fi

# Refuse a destination that is inside a source, or a source inside the destination. This is
# the guard that matters: everything else this script does is a read.
while IFS=$'\t' read -r -u 3 role kind rel src excl; do
  [[ -n "${src:-}" ]] || continue
  srcdir="$src"; [[ "$kind" == file ]] && srcdir="$(dirname "$src")"
  if [[ -z "$DEST_HOST" ]]; then
    case "$DEST_PATH/" in "$srcdir"/*) echo "REFUSING: destination $DEST_PATH is inside source $srcdir" >&2; exit 3 ;; esac
    case "$srcdir/" in "$DEST_PATH"/*) echo "REFUSING: source $srcdir is inside destination $DEST_PATH" >&2; exit 3 ;; esac
  fi
done 3<<<"$PLAN"

if [[ -n "$REMOTE_SIDE" ]]; then
  if ! ssh "${SSH_CM[@]}" -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_SIDE" true 2>/dev/null; then
    cat >&2 <<EOF
Cannot reach $REMOTE_SIDE without a prompt. Open one master connection first, and every
step below -- including --plan -- will reuse it:

    ssh ${SSH_CM[*]} -fN $REMOTE_SIDE

The password is asked once and the connection stays open for 8 hours. (Or install a public
key in that account's ~/.ssh/authorized_keys and skip this entirely.)
EOF
    exit 4
  fi
fi

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
[[ -n "$REMOTE_SIDE" ]] && BASE+=(-e "$SSH")

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

  if ! on_src "test -e '$src'"; then
    printf '  MISSING SOURCE  %-22s %s\n' "$role" "$src"; rc=1; return
  fi

  local target="${DEST_HOST:+$DEST_HOST:}$DEST_PATH/$rel/"
  local -a srcargs
  if [[ "$kind" == dir ]]; then srcargs=("${SRC_HOST:+$SRC_HOST:}$src/")
  else srcargs=("${SRC_HOST:+$SRC_HOST:}$src"); fi

  # A destination that already holds this role, and that we are not explicitly resuming into,
  # is worth stopping for -- it usually means a second run against a changed manifest.
  # The test has to be per-kind: a directory role owns its whole directory, but a file role
  # owns only its own file and shares the directory with sibling roles (r1_structure_tables
  # lands in r1/, right next to r1/simulations).
  if [[ "$MODE" == apply && $RESUME -eq 0 ]]; then
    local occupied=""
    if [[ "$kind" == dir ]]; then
      on_dest "test -d '$DEST_PATH/$rel' && find '$DEST_PATH/$rel' -mindepth 1 -print -quit | grep -q ." \
        >/dev/null 2>&1 && occupied="$rel/"
    else
      on_dest "test -e '$DEST_PATH/$rel/$(basename "$src")'" \
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
    rsync "${FLAGS[@]}" "${exargs[@]}" "${srcargs[@]}" "$target" </dev/null \
      | grep -v '^created ' || true
  else
    rsync "${FLAGS[@]}" "${exargs[@]}" "${srcargs[@]}" "$target" </dev/null
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
  if [[ "$kind" == dir ]]; then srcargs=("${SRC_HOST:+$SRC_HOST:}$src/")
  else srcargs=("${SRC_HOST:+$SRC_HOST:}$src"); fi

  # 1. Itemized dry run must be silent: every file present, same size, same mtime.
  local diff
  diff="$(rsync "${FLAGS[@]}" "${exargs[@]}" "${srcargs[@]}" "$target" </dev/null \
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
    a="$(on_src "find '$src' -type f 2>/dev/null | wc -l")"
    b="$(on_dest "find '$DEST_PATH/$rel' -type f 2>/dev/null | wc -l")"
    if [[ "$a" != "$b" ]]; then
      printf '  COUNT     %-22s src=%s dest=%s\n' "$role" "$a" "$b"; rc=1; return
    fi
  fi
  printf '  ok        %-22s\n' "$role"
}

echo "mode        ${SRC_HOST:+pull from $SRC_HOST}${DEST_HOST:+push to $DEST_HOST}"
echo "plan        ${SRC_HOST:+$SRC_HOST:}$SRC_REPO/data/MANIFEST.toml"
echo "destination ${DEST_HOST:+$DEST_HOST:}$DEST_PATH"
echo
while IFS=$'\t' read -r -u 3 role kind rel src excl; do
  [[ -n "${role:-}" ]] || continue
  n_roles=$((n_roles + 1))
  if [[ "$MODE" == verify ]]; then verify_role "$role" "$kind" "$rel" "$src" "${excl:-}"
  else transfer "$role" "$kind" "$rel" "$src" "${excl:-}"; fi
done 3<<<"$PLAN"

echo
if [[ $rc -eq 0 ]]; then
  echo "$n_roles entries OK ($MODE)"
  [[ "$MODE" == dry ]] && echo "Nothing was written. Re-run with --apply, then --verify."
else
  echo "$MODE finished with problems (see above)"
fi
exit $rc
