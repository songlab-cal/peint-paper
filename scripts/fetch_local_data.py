#!/usr/bin/env python
"""Download the paper's data from the Hugging Face Hub into local_data/.

Two tiers, because most people want the first one:

  figure_data  ~26 MB   every table needed to re-render all the panels. No model
                        weights, no GPU, no network beyond this download.
  full         ~14 GB   the inputs behind those tables (~46 GB unpacked), so the
                        metrics can be recomputed rather than replotted.

    scripts/fetch_local_data.py --tier figure_data
    scripts/fetch_local_data.py --tier full --repo <org>/peint-paper-data
    scripts/fetch_local_data.py --tier full --archives r1 sim    # just these
    scripts/fetch_local_data.py --verify-only                    # re-check checksums

`figure_data` is stored as loose files so it can be browsed on the Hub; the bulk roles are
one `.tar.zst` each. Tarring them is deliberate -- the Hub caps a folder at 10,000 entries
and `output_site_rates_dir` alone holds 30,104, which would also make the download tens of
thousands of separate requests.

What to fetch is read from data/MANIFEST.toml, the same inventory the staging and archive
scripts use, so this cannot drift from what was actually deposited.
"""

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paper import manifest  # noqa: E402

DEFAULT_REPO = os.environ.get("PEINT_PAPER_HF_REPO", "TODO-org/peint-paper-data")
CHECKSUMS = "CHECKSUMS.sha256"


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


def _unpack(archive, dest):
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("tar") is None:
        sys.exit("need `tar` (with zstd support) to unpack")
    print(f"  unpacking {archive.name} -> {dest}")
    subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(archive),
                    "-C", str(dest)], check=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("figure_data", "full"), default="figure_data")
    ap.add_argument("--archives", nargs="+",
                    help="Override the tier's archive list (implies --tier full).")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF dataset repo id (default {DEFAULT_REPO}, "
                         f"or $PEINT_PAPER_HF_REPO).")
    ap.add_argument("--revision", default=None,
                    help="Pin a git revision or tag -- use the one the paper cites.")
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
            print(f"  {name:12s} {arc['file']:16s} {t['apparent_mb'] / 1024:5.1f} G unpacked  "
                  f"{t['files']:>7,} files")
            print(f"      {arc.get('note', '')}")
            print(f"      roles: {', '.join(r.name for r in rs)}")
        loose = [r for r in manifest.roles(tier="figure_data")]
        print(f"  {'(loose)':12s} {'figure_data/':16s} "
              f"{manifest.totals(loose)['apparent_mb']} M unpacked  "
              f"{manifest.totals(loose)['files']:>7,} files")
        return 0

    if args.verify_only:
        return 1 if _verify(local) else 0

    if args.repo.startswith("TODO"):
        sys.exit("Set --repo (or $PEINT_PAPER_HF_REPO) to the published dataset repo id.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")

    if args.archives:
        unknown = [a for a in args.archives if a not in arcs]
        if unknown:
            sys.exit(f"unknown archive(s) {unknown}; known: {sorted(arcs)}")
        wanted = list(args.archives)
    elif args.tier == "full":
        wanted = list(arcs)
    else:
        wanted = []

    # The loose figure_data tier always comes along: it is small and it is what makes every
    # panel replottable regardless of which bulk archives were chosen.
    patterns = [f"{r['path']}/**" for r in manifest.roles(tier="figure_data")]
    patterns += ["README.md", "MANIFEST.toml", CHECKSUMS]
    patterns += [arcs[a]["file"] for a in wanted]

    local.mkdir(parents=True, exist_ok=True)
    print(f"repo    {args.repo}" + (f" @ {args.revision}" if args.revision else ""))
    print(f"into    {local}")
    print(f"tier    {args.tier}" + (f"  archives={','.join(wanted)}" if wanted else ""))
    snapshot_download(repo_id=args.repo, repo_type="dataset", revision=args.revision,
                      local_dir=str(local), allow_patterns=patterns)

    if not args.no_verify:
        if _verify(local):
            sys.exit("checksum verification failed; not unpacking. Re-run the download.")

    for a in wanted:
        arc = local / arcs[a]["file"]
        if not arc.exists():
            print(f"  WARNING {arcs[a]['file']} was not downloaded; skipping")
            continue
        _unpack(arc, local)              # every archive is relative to local_data/
        if not args.keep_archives:
            arc.unlink()

    print("\nDone. Check what the figures can see with:")
    print("  scripts/check_local_data.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
