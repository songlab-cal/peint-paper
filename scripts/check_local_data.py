#!/usr/bin/env python
"""Report which data roles are present, and name the archive that supplies each missing one.

    scripts/check_local_data.py                     # everything the deposit ships
    scripts/check_local_data.py --tier figure_data  # just the replot tier
    scripts/check_local_data.py --panels pcp_panels blast_similarity
    scripts/check_local_data.py --sources           # can THIS machine stage the deposit?

The role inventory lives in data/MANIFEST.toml; this is the human-facing view of it.
`--panels` takes panel names from scripts/render_panels.py and answers the question you
actually have: "I want these figures -- what am I missing, and where does it come from?"
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from paper import manifest  # noqa: E402


def _gb(mb):
    return f"{mb / 1024:.1f} G" if mb >= 1024 else f"{mb} M"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=manifest.TIERS)
    ap.add_argument("--roles", nargs="+", help="Check only these role names.")
    ap.add_argument("--panels", nargs="+",
                    help="Check only what these panels need (names from render_panels.py).")
    ap.add_argument("--sources", action="store_true",
                    help="Probe the source side instead: can this machine stage the deposit?")
    args = ap.parse_args(argv)

    if args.panels:
        sel, seen = [], set()
        for panel in args.panels:
            for r in manifest.roles(panel=panel):
                if r.name not in seen:
                    seen.add(r.name)
                    sel.append(r)
        if not sel:
            print(f"No roles are declared for {args.panels}. Panel names come from "
                  f"`scripts/render_panels.py --list`.", file=sys.stderr)
            return 2
    elif args.roles:
        sel = manifest.roles(name=args.roles)
    elif args.tier:
        sel = manifest.roles(tier=args.tier)
    else:
        sel = manifest.roles(shipped=True)

    if args.sources:
        return manifest.main(["--sources"])

    present, missing = manifest.check(sel)
    arcs = manifest.archives()

    for r in present:
        print(f"  ok        {r.name:22s} {r['path']}")

    needed = {}
    for r in missing:
        if r.get("tier") == "on_request":
            first_line = (r.get("note") or "not deposited").split("\n")[0]
            print(f"  --        {r.name:22s} {r['path']}\n"
                  f"                                  absent by design: {first_line}")
            continue
        arc = arcs.get(r.get("archive"))
        supplier = arc["file"] if arc else "loose files in the dataset repo"
        needed.setdefault(supplier, []).append(r)
        print(f"  MISSING   {r.name:22s} {r['path']}   <- {supplier}")

    t = manifest.totals([r for r in sel if r.get("tier") != "on_request"])
    print(f"\n{len(present)}/{len(sel)} roles present under {manifest.local_data()}")
    print(f"selection is {_gb(t['apparent_mb'])} unpacked across {t['files']:,} files")

    if needed:
        print("\nTo get what is missing:")
        for supplier, rs in sorted(needed.items()):
            mb = sum(r.get("apparent_mb", 0) for r in rs)
            print(f"  {supplier:20s} {_gb(mb):>8s} unpacked   ({', '.join(r.name for r in rs)})")
        print("\n  scripts/fetch_local_data.py --tier full")

    return 1 if needed else 0


if __name__ == "__main__":
    sys.exit(main())
