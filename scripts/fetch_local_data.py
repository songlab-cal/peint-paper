#!/usr/bin/env python
"""Download the paper's data from its Zenodo record into local_data/.

Two tiers, because most people want the first one:

  figure_data  ~5 MB    every table needed to re-render all the panels. No model
                        weights, no GPU, no other download.
  full         ~22 GB   the inputs behind those tables (~151 GB unpacked), so the
                        metrics can be recomputed rather than replotted.

    scripts/fetch_local_data.py --tier figure_data
    scripts/fetch_local_data.py --tier full --record 1234567
    scripts/fetch_local_data.py --tier full --archives r1 sim    # just these
    scripts/fetch_local_data.py --verify-only                    # re-check checksums

Every role is a `.tar.zst`: a Zenodo record is a flat list of files with no directories,
and several roles hold tens of thousands of small per-family files.

What to fetch is read from data/MANIFEST.toml, the same inventory the staging and archive
scripts use, so this cannot drift from what was actually deposited.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paper import manifest  # noqa: E402

DEFAULT_RECORD = os.environ.get("PEINT_PAPER_ZENODO_RECORD", "")
ZENODO_BASE = os.environ.get("PEINT_PAPER_ZENODO_BASE", "https://zenodo.org")
CHECKSUMS = "CHECKSUMS.sha256"
METADATA = ["MANIFEST.toml", "README.md", CHECKSUMS]


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _verify(local, names=None):
    """Check downloaded files against CHECKSUMS.sha256. Returns the number of failures."""
    sums = local / CHECKSUMS
    if not sums.exists():
        print(f"  no {CHECKSUMS} alongside the download; skipping verification")
        return 0
    bad = checked = 0
    for line in sums.read_text().splitlines():
        if not line.strip():
            continue
        digest, _, rel = line.partition("  ")
        rel = rel.strip()
        if names and rel not in names:
            continue
        target = local / rel
        if not target.exists():          # not part of this tier's download
            continue
        checked += 1
        if _sha256(target) != digest:
            print(f"  CHECKSUM FAILED  {rel}")
            bad += 1
    print(f"  verified {checked - bad}/{checked} file(s)")
    return bad


def _download(record, name, dest, required=True):
    """Fetch one file from the record. Zenodo serves files at a stable per-record path."""
    url = f"{ZENODO_BASE}/records/{record}/files/{name}?download=1"
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"  downloading {name}")
    try:
        with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
            shutil.copyfileobj(r, fh, 1 << 20)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code == 404 and not required:
            print(f"    not in this record; skipping {name}")
            return False
        sys.exit(f"could not fetch {name} from record {record}: HTTP {e.code}")
    except urllib.error.URLError as e:
        tmp.unlink(missing_ok=True)
        sys.exit(f"could not reach {ZENODO_BASE}: {e.reason}")
    tmp.replace(dest)
    return True


def _unpack(archive, dest, into=""):
    """Unpack one archive under `dest`, honouring the archive's `unpack_into` prefix.

    Almost every archive's members are already destination-relative, so `into` is empty and
    they land directly in local_data/. The two prebuilt rev1 structure archives are the
    exception: they were tarred from their source directories, so their members start at
    `af2/` / `omegafold/` and must be extracted into local_data/r1/ instead.
    """
    target = dest / into if into else dest
    target.mkdir(parents=True, exist_ok=True)
    if shutil.which("tar") is None:
        sys.exit("need `tar` (with zstd support) to unpack")
    print(f"  unpacking {archive.name} -> {target}")
    subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(archive),
                    "-C", str(target)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("figure_data", "full"), default="figure_data")
    ap.add_argument("--archives", nargs="+",
                    help="Override the tier's archive list (implies --tier full).")
    ap.add_argument("--record", default=DEFAULT_RECORD,
                    help="Zenodo record id (or $PEINT_PAPER_ZENODO_RECORD).")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--keep-archives", action="store_true",
                    help="Do not delete the .tar.zst files after unpacking.")
    ap.add_argument("--no-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true",
                    help="Re-check an existing download against CHECKSUMS.sha256 and exit.")
    ap.add_argument("--list", action="store_true", help="Describe the archives and exit.")
    args = ap.parse_args(argv)

    arcs = manifest.archives()
    local = Path(args.local_dir) if args.local_dir else manifest.local_data()

    if args.list:
        for name, arc in arcs.items():
            rs = manifest.roles(archive=name)
            t = manifest.totals(rs)
            print(f"  {name:12s} {arc['file']:28s} {t['apparent_mb'] / 1024:5.1f} G unpacked  "
                  f"{t['files']:>9,} files")
            print(f"      {arc.get('note', '')}")
            print(f"      roles: {', '.join(r.name for r in rs) or '(none shipped)'}")
        return 0

    if args.verify_only:
        return 1 if _verify(local) else 0

    if not args.record:
        sys.exit("Set --record (or $PEINT_PAPER_ZENODO_RECORD) to the published record id.")

    if args.archives:
        unknown = [a for a in args.archives if a not in arcs]
        if unknown:
            sys.exit(f"unknown archive(s) {unknown}; known: {sorted(arcs)}")
        wanted = list(args.archives)
    else:
        # The archives this tier actually deposits, in manifest order. on_request roles
        # declare archives too; those are staged on demand and are not in the record.
        shipped = {r.get("archive") for r in manifest.roles(tier="full")}
        full = [a for a in arcs if a in shipped]
        figure = [a for a in arcs
                  if a in {r.get("archive") for r in manifest.roles(tier="figure_data")}]
        # figure_data always comes along: it is small and it is what makes every panel
        # replottable regardless of which bulk archives were chosen.
        wanted = full if args.tier == "full" else figure
    for a in [x for x in arcs if x in {r.get("archive")
                                       for r in manifest.roles(tier="figure_data")}]:
        if a not in wanted:
            wanted.insert(0, a)

    local.mkdir(parents=True, exist_ok=True)
    print(f"record  {ZENODO_BASE}/records/{args.record}")
    print(f"into    {local}")
    print(f"tier    {args.tier}  archives={','.join(wanted)}")

    for name in METADATA:
        _download(args.record, name, local / name, required=(name != "README.md"))

    got = []
    for a in wanted:
        f = arcs[a]["file"]
        if _download(args.record, f, local / f):
            got.append(f)

    if not args.no_verify:
        if _verify(local, names=set(got) | set(METADATA)):
            sys.exit("checksum verification failed; not unpacking. Re-run the download.")

    for a in wanted:
        arc = local / arcs[a]["file"]
        if not arc.exists():
            continue
        _unpack(arc, local, arcs[a].get("unpack_into", ""))
        if not args.keep_archives:
            arc.unlink()

    print("\nDone. Check what the figures can see with:")
    print("  scripts/check_local_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
