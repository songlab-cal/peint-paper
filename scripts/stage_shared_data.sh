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
#   BULK ARCHIVES — for roles too file-heavy to rsync (the rev1 structure trees are
#   481,316 and 607,696 files), --as-archive streams the role as one tar.zst straight to
#   the destination. Nothing intermediate is written, so it costs no scratch on either side.
#     scripts/stage_shared_data.sh --src-host akoehl --only r1_af2 --as-archive --apply
#     scripts/stage_shared_data.sh --src-host akoehl --only r1_af2 --as-archive --verify
#   --only reaches on_request roles by name; --level sets zstd effort (default 10).
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
# Exported, not set per call: `python -m paper.manifest` otherwise only resolves when the
# working directory happens to be the repo root, and this script is meant to be run from
# anywhere -- including copied to another account on its own.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PEINT_PAPER_PY_ESMC:-${PEINT_PAPER_PY:-python3}}"
SHARED_DIR="/scratch/users/spa-evolution-yss/peint_paper_data"
SHARED_USER="${PEINT_PAPER_SHARED_USER:-spa-evolution-yss}"
DEST="${PEINT_PAPER_SHARED_DEST:-}"

MODE=dry
ONLY=""
RESUME=0
AS_ARCHIVE=0
ZLEVEL="${PEINT_PAPER_ZSTD_LEVEL:-10}"
ZTHREADS="${PEINT_PAPER_ZSTD_THREADS:-8}"
SRC_HOST="${PEINT_PAPER_SRC_HOST:-}"
# The ssh hop switches *user*, not machine: /scratch is one NFS mount, so the source data is
# the same bytes from any node. Default to whichever host you are on, and accept a bare
# username for --src-host so nothing here is pinned to one login node.
THIS_HOST="$(hostname)"
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
    --as-archive) AS_ARCHIVE=1 ;;
    --level)   ZLEVEL="${2:?--level needs a number}"; shift ;;
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
  else DEST="$SHARED_USER@$THIS_HOST:$SHARED_DIR"; fi
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

[[ -n "$SRC_HOST" && "$SRC_HOST" != *@* ]] && SRC_HOST="$SRC_HOST@$THIS_HOST"
[[ -n "$SRC_HOST" && -n "$DEST_HOST" ]] && {
  echo "remote source and remote destination at once is not supported" >&2; exit 2; }

# -n is load-bearing, not hygiene: these run inside a loop reading the plan from stdin, and
# an ssh without it swallows the remaining lines. That silently transferred 1 of 5 files in
# the first pull-mode run.
on_src()  { if [[ -n "$SRC_HOST" ]];  then ssh -n "${SSH_CM[@]}" "$SRC_HOST"  "$@"; else bash -c "$*" </dev/null; fi; }
on_dest() { if [[ -n "$DEST_HOST" ]]; then ssh -n "${SSH_CM[@]}" "$DEST_HOST" "$@"; else bash -c "$*" </dev/null; fi; }
# Same, but keeps stdin open -- only for the archive stream, which pipes into it.
on_dest_pipe() { if [[ -n "$DEST_HOST" ]]; then ssh "${SSH_CM[@]}" "$DEST_HOST" "$@"; else bash -c "$*"; fi; }

# Whichever side is remote, that is the one needing an open connection.
REMOTE_SIDE="${SRC_HOST:-$DEST_HOST}"

# --plan needs the connection only in pull mode, where the manifest itself lives over there.
# In push mode --plan reads a local manifest and touches nothing remote.
NEED_SSH=""
[[ -n "$SRC_HOST" ]] && NEED_SSH="$SRC_HOST"
[[ -n "$DEST_HOST" && "$MODE" != plan ]] && NEED_SSH="$DEST_HOST"

if [[ -n "$NEED_SSH" ]]; then
  if ! ssh "${SSH_CM[@]}" -o BatchMode=yes -o ConnectTimeout=5 "$NEED_SSH" true 2>/dev/null; then
    cat >&2 <<EOF
Cannot reach $NEED_SSH without a prompt. Open one master connection first, and every
step below -- including --plan -- will reuse it:

    ssh ${SSH_CM[*]} -fN $NEED_SSH

The password is asked once and the connection stays open for 8 hours. (Or install a public
key in that account's ~/.ssh/authorized_keys and skip this entirely.)
EOF
    exit 4
  fi
fi

# ---------------------------------------------------------------- staleness
# In pull mode this file is a COPY, taken from the source repo at some point in the past,
# while the plan it runs on comes from that repo live. That split is the whole reason the
# receiving account needs no checkout -- and it is also a drift hazard, because a stale copy
# will happily run against a newer manifest. Compare the two and say so; never act on it.
if [[ -n "$SRC_HOST" && "$MODE" != plan ]]; then
  _mine="$(sha256sum "${BASH_SOURCE[0]}" 2>/dev/null | cut -d' ' -f1)"
  _theirs="$(on_src "sha256sum '$SRC_REPO/scripts/stage_shared_data.sh' 2>/dev/null | cut -d' ' -f1" || true)"
  if [[ -n "$_mine" && -n "$_theirs" && "$_mine" != "$_theirs" ]]; then
    cat >&2 <<EOF
NOTE: this copy of stage_shared_data.sh differs from the one in $SRC_REPO.
      Refresh it with:
          ssh $SRC_HOST 'cat $SRC_REPO/scripts/stage_shared_data.sh' > "${BASH_SOURCE[0]}"
      Continuing with the copy you have.

EOF
  fi
fi

# ---------------------------------------------------------------- the plan
# Tab-separated: role, kind(dir|file), destination-relative path, source, comma-separated
# excludes. One source of truth -- data/MANIFEST.toml -- for this, the symlink farm, the
# archive builder and the downloader.
# In pull mode the manifest lives on the source side, behind a mode-700 directory, so the
# plan is generated there and streamed back. That is what lets the pulling account run this
# script with no checkout, no conda and no python of its own.
# Run paper.manifest on whichever side actually holds the repo. In pull mode that is the
# source, which is the whole reason the pulling account needs no checkout -- but it has to be
# every call, not just the plan: --plan asks for the role tables too, and a version of this
# that only redirected the plan appeared to work purely because the shell happened to be
# sitting in the repo.
manifest() {
  if [[ -n "$SRC_HOST" ]]; then on_src "cd '$SRC_REPO' && '$SRC_PYTHON' -m paper.manifest $*"
  else "$PY" -m paper.manifest "$@"; fi
}

if ! PLAN="$(manifest --rsync-plan 2>&1)"; then
  cat >&2 <<EOF
Could not generate the transfer plan${SRC_HOST:+ on $SRC_HOST}:

$PLAN

Tried: ${SRC_HOST:+cd $SRC_REPO && $SRC_PYTHON}${SRC_HOST:-$PY} -m paper.manifest --rsync-plan
The interpreter needs to read TOML (3.11+, or 3.10 with tomli) and import the repo.
In pull mode, point --src-python and --src-repo at the source account's interpreter and
checkout; locally, run from the repo or set PYTHONPATH to it.
EOF
  exit 5
fi
[[ -n "$PLAN" ]] || { echo "empty transfer plan from paper.manifest" >&2; exit 1; }
# Default to what the deposit ships. Naming a role explicitly overrides that, including the
# on_request ones -- asking for r1_af2 by name is a deliberate act, and making you pass a tier
# flag too would be friction rather than a safeguard.
if [[ -n "$ONLY" ]]; then
  PLAN="$(awk -F'\t' -v r="$ONLY" '$1==r' <<<"$PLAN")"
  [[ -n "$PLAN" ]] || { echo "no role named '$ONLY' in data/MANIFEST.toml" >&2; exit 2; }
else
  PLAN="$(awk -F'\t' '$2!="on_request"' <<<"$PLAN")"
fi

# Refuse a destination that is inside a source, or a source inside the destination. This is
# the guard that matters: everything else this script does is a read.
while IFS=$'\t' read -r -u 3 role tier kind rel src excl; do
  [[ -n "${src:-}" ]] || continue
  srcdir="$src"; [[ "$kind" == file ]] && srcdir="$(dirname "$src")"
  if [[ -z "$DEST_HOST" ]]; then
    case "$DEST_PATH/" in "$srcdir"/*) echo "REFUSING: destination $DEST_PATH is inside source $srcdir" >&2; exit 3 ;; esac
    case "$srcdir/" in "$DEST_PATH"/*) echo "REFUSING: source $srcdir is inside destination $DEST_PATH" >&2; exit 3 ;; esac
  fi
done 3<<<"$PLAN"

if [[ "$MODE" == plan ]]; then
  manifest --list --tier figure_data
  echo
  manifest --list --tier full
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

# ---------------------------------------------------------------- archive stream
# Move a role as ONE compressed stream instead of file-by-file. For the rev1 structure trees
# -- 481,316 and 607,696 files -- per-file rsync is the wrong tool by an order of magnitude,
# and tarring to disk first would need ~25 GB of scratch on a quota that has ~52 GB free.
# Nothing intermediate is ever written: tar and zstd run on the source side, the bytes land
# straight in the destination file, and it is owned by whoever runs the receiving end.
#
# The archive name matches the role's `archive` file in MANIFEST.toml, so deciding later to
# ship one of these is a tier change in the manifest, not a rename here.
stream_archive() {
  local role="$1" src="$2"
  local parent base arc
  parent="$(dirname "$src")"; base="$(basename "$src")"
  arc="$DEST_PATH/${role}.tar.zst"

  if [[ "$MODE" != apply ]]; then
    printf '  %-8s %-22s %s  (tar | zstd -%s -T%s, streamed)\n' \
      "$MODE" "$role" "${role}.tar.zst" "$ZLEVEL" "$ZTHREADS"
    return
  fi

  if [[ $RESUME -eq 0 ]] && on_dest "test -e '$arc'" >/dev/null 2>&1; then
    printf '  ALREADY THERE   %-22s %s (pass --resume to overwrite)\n' "$role" "${role}.tar.zst"
    rc=1; return
  fi

  printf '  %-8s %-22s %s\n' "$MODE" "$role" "${role}.tar.zst"
  # pipefail matters here: without it a tar that dies mid-way still leaves a valid-looking
  # zstd file, which is precisely the silent-truncation failure this script exists to avoid.
  local tarcmd="set -o pipefail; tar -C '$parent' -cf - '$base' | zstd -q -T$ZTHREADS -$ZLEVEL"
  if [[ -n "$SRC_HOST" ]]; then
    on_src "$tarcmd" > "$arc" || { printf '  FAILED    %-22s\n' "$role"; rc=1; return; }
  elif [[ -n "$DEST_HOST" ]]; then
    bash -o pipefail -c "$tarcmd" | on_dest_pipe "cat > '$arc'" \
      || { printf '  FAILED    %-22s\n' "$role"; rc=1; return; }
  else
    bash -o pipefail -c "$tarcmd" > "$arc" \
      || { printf '  FAILED    %-22s\n' "$role"; rc=1; return; }
  fi
}

verify_archive() {
  local role="$1" src="$2"
  local arc="$DEST_PATH/${role}.tar.zst"
  if ! on_dest "test -e '$arc'"; then
    printf '  MISSING   %-22s %s\n' "$role" "${role}.tar.zst"; rc=1; return
  fi
  # zstd frames carry a content checksum, so -t proves the stream decompresses to what was
  # compressed. It cannot prove tar was *given* everything, hence the member count too.
  if ! on_dest "zstd -t '$arc'" >/dev/null 2>&1; then
    printf '  CORRUPT   %-22s %s\n' "$role" "${role}.tar.zst"; rc=1; return
  fi
  local want have
  want="$(on_src "find '$src' -type f | wc -l")"
  have="$(on_dest "tar --use-compress-program=unzstd -tf '$arc' | grep -vc '/\$'")"
  if [[ "$want" != "$have" ]]; then
    printf '  COUNT     %-22s src=%s archive=%s\n' "$role" "$want" "$have"; rc=1; return
  fi
  printf '  ok        %-22s %s (%s members)\n' "$role" "${role}.tar.zst" "$have"
}

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
while IFS=$'\t' read -r -u 3 role tier kind rel src excl; do
  [[ -n "${role:-}" ]] || continue
  n_roles=$((n_roles + 1))
  if [[ $AS_ARCHIVE -eq 1 ]]; then
    if [[ "$kind" != dir ]]; then
      echo "  --as-archive only applies to directory roles; $role is a file list" >&2
      rc=1; continue
    fi
    if [[ "$MODE" == verify ]]; then verify_archive "$role" "$src"
    else stream_archive "$role" "$src"; fi
  elif [[ "$MODE" == verify ]]; then verify_role "$role" "$kind" "$rel" "$src" "${excl:-}"
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
