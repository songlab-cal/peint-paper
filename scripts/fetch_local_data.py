#!/usr/bin/env python
"""Download the paper's data from the Hugging Face Hub into local_data/.

Two tiers, because most people want the first one:

  figure_data  ~10 MB   every table needed to re-render all the panels. No model
                        weights, no GPU, no network beyond this download.
  full         ~4 GB    the inputs behind those tables, so the metrics can be
                        recomputed rather than replotted.

    scripts/fetch_local_data.py --tier figure_data
    scripts/fetch_local_data.py --tier full --repo <org>/peint-paper-data
    scripts/fetch_local_data.py --tier full --roles r1 sim      # just these

`figure_data` is stored as loose files so it can be browsed on the Hub; the bulk roles are
one `.tar.zst` each. Tarring them is deliberate — the Hub caps a folder at 10,000 entries
and several of these trees hold tens of thousands of small per-family files, which would
also make the download thousands of separate requests.
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = os.environ.get("PEINT_PAPER_HF_REPO", "TODO-org/peint-paper-data")

# role -> (archive name in the HF repo, directory it unpacks to, what it is for)
ROLES = {
    "r1": ("r1.tar.zst", "r1",
           "revision 1: classical baselines, PEINT-ESM2, real (mafft_add, simulations, 3Di)"),
    "r2": ("r2.tar.zst", "r2",
           "revision 2: the ESM-C rerun, incl. the Historian event tables"),
    "sim": ("sim.tar.zst", "sim",
            "trees, root sequences, empirical + simulated MSAs"),
    "aux": ("aux.tar.zst", None,
            "annotation labels, per-site rates, family split lists"),
    "omegafold_structures": ("omegafold_structures.tar.zst", "r2/omegafold",
                             "optional: the raw ESM-C OmegaFold PDBs (only to re-derive pLDDT)"),
}
FULL_ROLES = ["sim", "aux", "r1", "r2"]     # omegafold_structures is opt-in


def _unpack(archive: Path, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    if shutil.which("zstd") is None and shutil.which("tar") is None:
        sys.exit("need `tar` (and zstd support) to unpack")
    print(f"  unpacking {archive.name} -> {dest}")
    subprocess.run(["tar", "--use-compress-program=unzstd", "-xf", str(archive), "-C", str(dest)],
                   check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=("figure_data", "full"), default="figure_data")
    ap.add_argument("--roles", nargs="+", choices=sorted(ROLES),
                    help="Override the tier's role list (implies --tier full).")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"HF dataset repo id (default {DEFAULT_REPO}, "
                         f"or $PEINT_PAPER_HF_REPO).")
    ap.add_argument("--revision", default=None,
                    help="Pin a git revision or tag — use the one the paper cites.")
    ap.add_argument("--local-dir", default=os.environ.get(
        "PEINT_PAPER_LOCAL_DATA", str(REPO_ROOT / "local_data")))
    ap.add_argument("--keep-archives", action="store_true",
                    help="Do not delete the .tar.zst files after unpacking.")
    ap.add_argument("--list", action="store_true", help="Describe the roles and exit.")
    args = ap.parse_args()

    if args.list:
        for name, (arc, dest, why) in ROLES.items():
            star = "" if name in FULL_ROLES else "   (opt-in)"
            print(f"  {name:22s} {arc:32s}{star}\n      {why}")
        return 0

    if args.repo.startswith("TODO"):
        sys.exit("Set --repo (or $PEINT_PAPER_HF_REPO) to the published dataset repo id.")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("pip install huggingface_hub")

    local = Path(args.local_dir)
    local.mkdir(parents=True, exist_ok=True)

    if args.roles:
        roles = args.roles
    elif args.tier == "full":
        roles = FULL_ROLES
    else:
        roles = []

    patterns = ["figure_data/**", "README.md", "MANIFEST.toml"]
    patterns += [ROLES[r][0] for r in roles]

    print(f"repo    {args.repo}" + (f" @ {args.revision}" if args.revision else ""))
    print(f"into    {local}")
    print(f"tier    {args.tier}" + (f"  roles={','.join(roles)}" if roles else ""))
    snapshot_download(repo_id=args.repo, repo_type="dataset", revision=args.revision,
                      local_dir=str(local), allow_patterns=patterns)

    for r in roles:
        arc_name, dest_rel, _ = ROLES[r]
        arc = local / arc_name
        if not arc.exists():
            print(f"  WARNING {arc_name} was not downloaded; skipping")
            continue
        # dest None means the archive carries its own top-level directories.
        _unpack(arc, local if dest_rel is None else local / dest_rel)
        if not args.keep_archives:
            arc.unlink()

    print("\nDone. Check what the figures can see with:")
    print("  scripts/render_panels.py --check")
    return 0


if __name__ == "__main__":
    sys.exit(main())
