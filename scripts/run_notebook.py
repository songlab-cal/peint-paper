#!/usr/bin/env python
"""Execute a .ipynb's code cells as a plain script — no jupyter, no nbconvert.

The Fig 4 producers ship as notebooks, but their imports are ordinary
(pandas/matplotlib/seaborn/numpy/sklearn/Bio/biotite/ete3), so a notebook runtime is not
actually required to reproduce the panels. This reads the cells in order and execs them in one
namespace with cwd set to the notebook's directory, which is what relative data paths assume.

    python scripts/run_notebook.py <notebook.ipynb> [--dry-run]

IPython magics (%...), shell escapes (!...) and `display(...)` are skipped/stubbed, since none
of them affect the figures. Matplotlib is forced to Agg.
"""
import argparse, json, os, re, sys, pathlib

def cells(nb):
    j = json.loads(pathlib.Path(nb).read_text())
    for i, c in enumerate(j.get("cells", []), 1):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source", []))
        # drop magics and shell escapes; keep everything else verbatim
        src = "\n".join(l for l in src.split("\n")
                        if not re.match(r'^\s*[%!]', l))
        if src.strip():
            yield i, src

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("notebook")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--stub", default="", help="comma-separated module:Name to stub before "
                    "running, e.g. ete3:TreeStyle. Use for GUI-only symbols that a "
                    "figure-producing cell never actually calls.")
    ap.add_argument("--keep-going", action="store_true",
                    help="continue after a failing cell (report at the end)")
    a = ap.parse_args()
    nb = pathlib.Path(a.notebook).resolve()
    os.chdir(nb.parent)
    import matplotlib; matplotlib.use("Agg")
    for spec in filter(None, a.stub.split(",")):
        mod, name = spec.split(":")
        m = __import__(mod)
        if not hasattr(m, name):
            setattr(m, name, type(name, (), {"__init__": lambda self, *a, **k: None}))
            print(f"[stub] {mod}.{name}")
    g = {"__name__": "__main__", "__file__": str(nb), "display": lambda *x, **k: None,
         "get_ipython": lambda: None}
    failed = []
    for i, src in cells(nb):
        if a.dry_run:
            print(f"--- cell {i} ({len(src.splitlines())} lines)"); continue
        print(f"=== cell {i} ===", flush=True)
        try:
            exec(compile(src, f"{nb.name}:cell{i}", "exec"), g)
        except Exception as e:
            print(f"  CELL {i} FAILED: {type(e).__name__}: {e}", flush=True)
            failed.append(i)
            if not a.keep_going:
                raise
    print(f"notebook complete (failed cells: {failed or 'none'})")

if __name__ == "__main__":
    main()
