"""Reader for ``data/MANIFEST.toml``, the inventory of the paper's data roles.

The role list used to exist three times -- hardcoded in ``scripts/link_local_data.sh``, in a
``ROLES`` dict in ``scripts/fetch_local_data.py``, and implicitly in ``paper_config`` -- and
had already drifted. Everything now reads it from here.

    from paper import manifest
    manifest.roles(tier="full")               # what ships in the archives
    manifest.roles(panel="pcp_panels")        # what one panel needs
    manifest.check(manifest.roles())          # (present, missing), by probing

As a CLI it emits the machine-readable forms the shell scripts consume::

    python -m paper.manifest --list
    python -m paper.manifest --links          # rel|target, for link_local_data.sh
    python -m paper.manifest --rsync-plan     # role/kind/dest/source/excludes, tab-separated
    python -m paper.manifest --chain          # producer -> role -> panels, for provenance
"""

import argparse
import os
import sys
from pathlib import Path

try:                                    # 3.11+
    import tomllib
except ModuleNotFoundError:             # 3.10 and older; both project envs are 3.10
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = Path(os.environ.get(
    "PEINT_PAPER_MANIFEST", str(REPO_ROOT / "data" / "MANIFEST.toml")))

TIERS = ("figure_data", "full", "on_request")
SHIPPED_TIERS = ("figure_data", "full")

# Placeholder -> the paper_config symbol that overrides the manifest's [defaults], and the
# environment variable behind it. paper_config is the authority whenever it imports; the env
# vars are the fallback so the staging script runs from a bare interpreter.
_ROOT_KEYS = {
    "data_root": ("DATA_ROOT", "PEINT_PAPER_DATA_ROOT"),
    "sim_root": ("SIM_ROOT", "PEINT_PAPER_SIM_ROOT"),
    "gt_pdb_root": ("GROUND_TRUTH_STRUCTURE_DIR", "PEINT_PAPER_GROUND_TRUTH_STRUCTURE_DIR"),
    "peint_repo": ("PEINT_REPO", "PEINT_PAPER_PEINT_REPO"),
}

_cache = {}


def _raw():
    if "raw" not in _cache:
        with open(MANIFEST_PATH, "rb") as fh:
            _cache["raw"] = tomllib.load(fh)
    return _cache["raw"]


def roots():
    """Absolute source roots, with ``{placeholder}`` names as keys."""
    if "roots" in _cache:
        return _cache["roots"]
    defaults = _raw().get("defaults", {})
    out = {"repo_root": str(REPO_ROOT)}
    try:
        import paper_config as cfg
        for key, (attr, _env) in _ROOT_KEYS.items():
            out[key] = str(getattr(cfg, attr))
    except Exception:
        for key, (_attr, env) in _ROOT_KEYS.items():
            out[key] = os.environ.get(env) or defaults.get(key, "")
    _cache["roots"] = out
    return out


def local_data():
    """Root the roles resolve under -- where a downloader unpacks the archives."""
    env = os.environ.get("PEINT_PAPER_LOCAL_DATA")
    if env:
        return Path(env)
    try:
        import paper_config as cfg
        return Path(cfg.LOCAL_DATA)
    except Exception:
        return REPO_ROOT / "local_data"


def _expand(template):
    return template.format(**roots())


class Role(dict):
    """One manifest entry. A dict, so unknown keys stay accessible as the schema grows."""

    @property
    def name(self):
        return self["name"]

    @property
    def dest(self):
        """Absolute path this role must occupy for the figures to find it."""
        return local_data() / self["path"]

    @property
    def source(self):
        """Absolute source directory on the producing machine, or None for a file list."""
        return Path(_expand(self["source"])) if "source" in self else None

    @property
    def source_files(self):
        return [Path(_expand(f)) for f in self.get("source_files", [])]

    @property
    def in_repo(self):
        """True if the role is git-tracked and ships with the code, not the deposit."""
        return bool(self.get("in_repo"))

    @property
    def excludes(self):
        return list(self.get("exclude", []))

    def probe_path(self):
        """The cheap path whose existence stands for the whole role."""
        probe = self.get("probe")
        if probe is None:                       # file-list roles probe their first file
            files = self.get("source_files")
            probe = Path(files[0]).name if files else "."
        return self.dest / probe

    def present(self):
        return self.probe_path().exists()


def load():
    return _raw()


def archives():
    return {a["name"]: a for a in _raw().get("archive", [])}


def roles(tier=None, archive=None, panel=None, name=None, shipped=False):
    """Roles matching every filter given. ``shipped=True`` means figure_data + full."""
    out = [Role(r) for r in _raw().get("role", [])]
    if tier:
        out = [r for r in out if r.get("tier") == tier]
    if shipped:
        out = [r for r in out if r.get("tier") in SHIPPED_TIERS]
    if archive:
        out = [r for r in out if r.get("archive") == archive]
    if panel:
        out = [r for r in out if panel in r.get("panels", []) or "all" in r.get("panels", [])]
    if name:
        wanted = {name} if isinstance(name, str) else set(name)
        out = [r for r in out if r["name"] in wanted]
    return out


def check(role_list):
    """Split roles into (present, missing) by probing. Never touches the sources."""
    present, missing = [], []
    for r in role_list:
        (present if r.present() else missing).append(r)
    return present, missing


def totals(role_list):
    return {
        "blocks_mb": sum(r.get("blocks_mb", 0) for r in role_list),
        "apparent_mb": sum(r.get("apparent_mb", 0) for r in role_list),
        "files": sum(r.get("files", 0) for r in role_list),
    }


# --------------------------------------------------------------------------- CLI

def _cmd_list(args):
    sel = roles(tier=args.tier, archive=args.archive, panel=args.panel)
    print(f"{'role':22s} {'tier':11s} {'arch':5s} {'blocks':>8s} {'apparent':>9s} "
          f"{'files':>8s}  path")
    for r in sel:
        print(f"{r.name:22s} {r.get('tier',''):11s} {r.get('archive','-'):5s} "
              f"{r.get('blocks_mb',0):7d}M {r.get('apparent_mb',0):8d}M "
              f"{r.get('files',0):8d}  {r['path']}")
    t = totals(sel)
    print(f"\n{len(sel)} roles   {t['blocks_mb']/1024:.1f} G blocks   "
          f"{t['apparent_mb']/1024:.1f} G apparent   {t['files']:,} files")
    return 0


def _cmd_links(args):
    """rel|target lines for the developer symlink farm. Directory roles only.

    A role's `exclude` list is deliberately ignored here: it is a staging and archiving
    concern (what the deposit ships), not a reason to leave the role unlinked locally.

    This is a declaration of what should exist, not a diff against what does -- roles whose
    source is already at the destination are filtered by the consumer, so the plan does not
    change shape depending on which links happen to be in place when it runs.
    """
    for r in roles(shipped=True):
        if r.source is not None:
            print(f"{r['path']}|{r.source}")
        for f in r.source_files:        # one link per file, so the layout matches the deposit
            print(f"{r['path']}/{f.name}|{f}")
    return 0


def _cmd_rsync_plan(args):
    """Tab-separated: role, tier, kind, dest-relative-path, source, excludes.

    Emits every tier, on_request included. The consumer decides what to act on -- naming an
    on_request role explicitly is a legitimate request for it, and making the caller pass a
    tier flag as well would be friction without safety.
    """
    for r in roles():
        exc = ",".join(r.excludes)
        tier = r.get("tier", "")
        if r.source is not None:
            print(f"{r.name}\t{tier}\tdir\t{r['path']}\t{r.source}\t{exc}")
        for f in r.source_files:
            print(f"{r.name}\t{tier}\tfile\t{r['path']}\t{f}\t")
    return 0


def _cmd_archive_plan(args):
    """Tab-separated: archive name, filename, destination-relative path, prebuilt flag.

    figure_data-tier roles have no archive and are emitted with an empty archive name -- they
    ship loose so they stay browsable on the Hub.

    The fourth field is "prebuilt" when the archive already exists as a finished .tar.zst in
    the staged tree (streamed there by stage_shared_data.sh --as-archive) and must be linked
    rather than rebuilt. Its third field is then not a path to tar, so consumers must skip the
    existence check for it. Appending rather than inserting keeps older $1/$2/$3 readers valid.
    """
    arcs = archives()
    for r in roles(shipped=True):
        if r.in_repo:                   # ships with the code; not part of the deposit
            continue
        arc = r.get("archive")
        meta = arcs.get(arc, {})
        file = meta.get("file", "")
        prebuilt = "prebuilt" if meta.get("prebuilt") else ""
        # A file-list role must contribute its individual files, never its directory: several
        # roles share r1/, and tarring the directory would sweep in the on_request ones.
        paths = ([f"{r['path']}/{f.name}" for f in r.source_files]
                 if r.get("source_files") else [r["path"]])
        for path in paths:
            print(f"{arc or ''}\t{file}\t{path}\t{prebuilt}")
    return 0


def _cmd_chain(args):
    """Provenance: what produced each role, and which panels consume it.

    The producer side is the half nothing else records -- render_panels knows how to draw a
    panel, the manifest knows where its inputs live, but neither says how those inputs came
    to exist. For a shipped structure tree that is the difference between data you can cite
    and data you have to take on faith.
    """
    sel = roles(tier=args.tier) if args.tier else roles()
    for r in sel:
        panels = ", ".join(r.get("panels", [])) or "-"
        print(f"{r.name}   [{r.get('tier','')}]")
        print(f"    path      {r['path']}")
        print(f"    produced  {r.get('producer', 'unrecorded')}")
        print(f"    feeds     {panels}")
        if r.get("files"):
            print(f"    size      {r.get('apparent_mb',0)} MB apparent, {r['files']:,} files")
        print()
    return 0


def _cmd_sources(args):
    """Probe the SOURCE side: can this machine stage the deposit? Read-only."""
    missing = 0
    for r in roles(shipped=True):
        for src in ([r.source] if r.source is not None else []) + r.source_files:
            ok = src.exists()
            missing += not ok
            print(f"  {'ok      ' if ok else 'MISSING '}  {r.name:22s} {src}")
    print(f"\n{'all sources present' if not missing else str(missing) + ' source(s) missing'}")
    return 1 if missing else 0


def _cmd_check(args):
    sel = roles(tier=args.tier, panel=args.panel) if (args.tier or args.panel) \
        else roles(shipped=True)
    present, missing = check(sel)
    arcs = archives()
    for r in present:
        print(f"  ok        {r.name:22s} {r['path']}")
    for r in missing:
        supplier = arcs.get(r.get("archive"), {}).get("file")
        where = f"in {supplier}" if supplier else "loose in the dataset repo"
        if r.get("tier") == "on_request":
            where = "absent by design -- " + (r.get("note", "").split("\n")[0])
        print(f"  MISSING   {r.name:22s} {r['path']}  ({where})")
    print(f"\n{len(present)}/{len(sel)} roles present under {local_data()}")
    return 1 if missing else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", choices=TIERS)
    ap.add_argument("--archive")
    ap.add_argument("--panel")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--list", action="store_true")
    g.add_argument("--links", action="store_true")
    g.add_argument("--rsync-plan", action="store_true")
    g.add_argument("--archive-plan", action="store_true")
    g.add_argument("--chain", action="store_true",
                   help="provenance: producer -> role -> panels")
    g.add_argument("--check", action="store_true",
                   help="probe the destination: can a downloader render?")
    g.add_argument("--sources", action="store_true",
                   help="probe the source side: can this machine stage?")
    args = ap.parse_args(argv)

    if args.links:
        return _cmd_links(args)
    if args.rsync_plan:
        return _cmd_rsync_plan(args)
    if args.archive_plan:
        return _cmd_archive_plan(args)
    if args.chain:
        return _cmd_chain(args)
    if args.check:
        return _cmd_check(args)
    if args.sources:
        return _cmd_sources(args)
    return _cmd_list(args)


if __name__ == "__main__":
    sys.exit(main())
