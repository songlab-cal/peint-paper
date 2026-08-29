#!/usr/bin/env bash
# Upload a built deposit directory to a Zenodo draft, from the machine that holds it.
#
# Uploading 26 GB through a laptop is a waste of a round trip, and Zenodo's browser
# uploader is unreliable for the 5-6 GB archives here. This uses the bucket API, which is
# the supported path for large files, and skips anything already uploaded at the same size
# so an interrupted run can simply be repeated.
#
#   export ZENODO_TOKEN=...            # never passed as an argument: see below
#   scripts/upload_to_zenodo.sh --deposition 22151902            # dry run
#   scripts/upload_to_zenodo.sh --deposition 22151902 --apply
#
# The token is read from the environment on purpose. Passing a credential as a command-line
# flag puts it in `ps` output and your shell history; an env var does neither. Create one at
# zenodo.org/account/settings/applications/tokens/new/ with the `deposit:write` scope, and
# keep it out of any file you commit.
#
# The draft must already exist (create it in the web form and Reserve DOI). This script only
# adds files -- it never publishes, and never deletes anything already in the draft.
set -euo pipefail

API="${ZENODO_API:-https://zenodo.org/api}"
DIR="_upload"
DEP=""
APPLY=0

usage() { awk 'NR>1 && /^#/ {sub(/^# ?/, ""); print; next} NR>1 {exit}' "${BASH_SOURCE[0]}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --deposition) DEP="${2:?--deposition needs the draft id}"; shift ;;
    --dir)        DIR="${2:?--dir needs a path}"; shift ;;
    --apply)      APPLY=1 ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

DEP="${DEP:-${ZENODO_DEPOSITION:-}}"
[[ -n "$DEP" ]] || { echo "--deposition is required (or set ZENODO_DEPOSITION)" >&2; exit 2; }
[[ -n "${ZENODO_TOKEN:-}" ]] || { echo "set ZENODO_TOKEN in the environment (see --help)" >&2; exit 2; }
[[ -d "$DIR" ]] || { echo "no such directory: $DIR" >&2; exit 2; }
command -v jq >/dev/null || { echo "need jq" >&2; exit 1; }

AUTH=(-H "Authorization: Bearer $ZENODO_TOKEN")

meta="$(curl -sf "${AUTH[@]}" "$API/deposit/depositions/$DEP")" || {
  echo "could not read draft $DEP -- check the id, the token, and that it is not published" >&2
  exit 5
}
BUCKET="$(jq -r '.links.bucket // empty' <<<"$meta")"
[[ -n "$BUCKET" ]] || { echo "draft $DEP has no file bucket (already published?)" >&2; exit 5; }
echo "draft    $(jq -r '.title // "(untitled)"' <<<"$meta")  [$DEP]"
echo "state    $(jq -r '.state' <<<"$meta")   submitted=$(jq -r '.submitted' <<<"$meta")"
echo "from     $DIR"
echo

# name -> size already in the draft, so a repeated run resumes instead of re-sending.
declare -A HAVE=()
while IFS=$'\t' read -r n s; do [[ -n "$n" ]] && HAVE["$n"]="$s"; done < <(
  jq -r '.files[]? | [.filename, (.filesize|tostring)] | @tsv' <<<"$meta")

rc=0
for f in "$DIR"/*; do
  [[ -f "$f" ]] || continue
  name="$(basename "$f")"
  size="$(stat -c%s "$f")"
  human="$(numfmt --to=iec --suffix=B "$size" 2>/dev/null || echo "$size")"
  if [[ "${HAVE[$name]:-}" == "$size" ]]; then
    printf '  have     %-38s %8s\n' "$name" "$human"; continue
  fi
  if [[ -n "${HAVE[$name]:-}" ]]; then
    printf '  DIFFERS  %-38s %8s (draft has %s bytes; delete it in the web form first)\n' \
           "$name" "$human" "${HAVE[$name]}" >&2
    rc=1; continue
  fi
  if [[ $APPLY -eq 0 ]]; then
    printf '  would    %-38s %8s\n' "$name" "$human"; continue
  fi
  printf '  upload   %-38s %8s\n' "$name" "$human"
  curl -f --progress-bar "${AUTH[@]}" --upload-file "$f" "$BUCKET/$name" -o /dev/null || {
    echo "  FAILED   $name -- re-run to resume" >&2; rc=1; }
done

echo
if [[ $APPLY -eq 0 ]]; then
  echo "Nothing uploaded. Re-run with --apply."
else
  echo "Done. Review the draft in the web form, then Publish there -- this script never publishes."
fi
exit $rc
