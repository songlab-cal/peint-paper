"""Extended Data Fig. 8b — categorical-Jacobian couplings against the true contact map.

The decoder categorical Jacobian for one structure is reduced to an L x L coupling matrix and
shown opposite that structure's contact map. The Top-L couplings are coloured by whether the
residue pair is genuinely in contact.

The raw Jacobian is an [L, A, L, A] tensor -- 32 MB for the published example -- but the panel
needs only the reduced L x L matrix and the true contacts, a few hundred KB together. Those
are cached, so the panel redraws with neither the tensor nor a model::

    python -m benchmarks.catjac_contact_map --from-npz            # from the cached matrices
    python -m benchmarks.catjac_contact_map --jacobian <x.npy> --pdb <x.pdb>   # from the tensor

``jac_to_couplings`` is the published reduction: centre each axis, take the Frobenius norm over
the two amino-acid axes, symmetrise the upper triangle the autoregressive decoder fills, drop
the diagonal, and apply the average product correction.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import paper_config as cfg
    DEFAULT_OUT = str(cfg.FIGURES_DIR)
    FIGURE_DATA = str(cfg.FIGURE_DATA_DIR)
except Exception:
    DEFAULT_OUT, FIGURE_DATA = ".", "."

CACHE = "catjac_contact_map_4k6e.npz"


def jac_to_couplings(jac, center=True, apc=True):
    """[L, A, L, A] categorical Jacobian -> [L, L] coupling matrix."""
    X = np.asarray(jac, dtype=np.float64).copy()
    if center:
        for axis in range(4):
            if X.shape[axis] > 1:
                X -= X.mean(axis, keepdims=True)
    c = np.sqrt(np.square(X).sum((1, 3)))
    L = c.shape[0]
    # the decoder fills only j >= i; mirror it
    sym = np.zeros((L, L), dtype=c.dtype)
    iu = np.triu_indices(L)
    sym[iu] = c[iu]
    sym = sym + sym.T - np.diag(np.diag(sym))
    np.fill_diagonal(sym, 0)
    if apc:
        sym = sym - (sym.sum(0, keepdims=True) * sym.sum(1, keepdims=True) / sym.sum())
        np.fill_diagonal(sym, 0)
    return sym


def cbeta_contacts(pdb_path, cutoff=8.0, min_sep=6):
    """True contacts from C-beta distances (C-alpha for glycine)."""
    import biotite.structure.io.pdb as pdb_io
    import biotite.structure as struc
    arr = pdb_io.PDBFile.read(pdb_path).get_structure(model=1)
    arr = arr[struc.filter_amino_acids(arr)]
    chain = arr[arr.chain_id == arr.chain_id[0]]
    coords, ok = [], []
    for res_id in np.unique(chain.res_id):
        res = chain[chain.res_id == res_id]
        sel = res[res.atom_name == ("CA" if res.res_name[0] == "GLY" else "CB")]
        if len(sel) == 0:
            sel = res[res.atom_name == "CA"]
        if len(sel):
            coords.append(sel.coord[0]); ok.append(res_id)
    coords = np.asarray(coords)
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
    sep = np.abs(np.arange(len(coords))[:, None] - np.arange(len(coords))[None, :])
    return ((d < cutoff) & (sep >= min_sep)).astype(np.int8)


def plot(couplings, contacts, out_dir, name="4K6E"):
    L = min(couplings.shape[0], contacts.shape[0])
    cp, ct = couplings[:L, :L], contacts[:L, :L]
    sep = np.abs(np.arange(L)[:, None] - np.arange(L)[None, :])
    mask = sep >= 6
    idx = np.argsort(cp[mask])[::-1][:L]                 # Top-L couplings
    ii, jj = np.where(mask)
    top_i, top_j = ii[idx], jj[idx]
    hit = ct[top_i, top_j] == 1
    precision = float(hit.mean())

    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    tri = np.tril(ct, -1)                                 # true contacts, lower triangle
    ty, tx = np.where(tri == 1)
    ax.scatter(tx, ty, s=2, c="0.75", marker="s", linewidths=0)
    up = top_i < top_j                                    # couplings, upper triangle
    ax.scatter(top_j[up & hit], top_i[up & hit], s=6, c="#1f77b4", linewidths=0, label="in contact")
    ax.scatter(top_j[up & ~hit], top_i[up & ~hit], s=6, c="#d62728", linewidths=0, label="not in contact")
    ax.set_xlim(0, L); ax.set_ylim(L, 0); ax.set_aspect("equal")
    ax.set_xlabel("residue"); ax.set_ylabel("residue")
    ax.set_title(f"{name}: Top-L catjac couplings vs contacts\nP@L = {precision:.3f}", fontsize=9)
    ax.legend(fontsize=7, markerscale=2, loc="lower right")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, "catjac_contact_map")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  P@L = {precision:.4f} over {len(top_i)} couplings, L = {L}")
    print(f"  wrote {stem}.{{pdf,png}}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jacobian", help="[L, A, L, A] .npy tensor")
    ap.add_argument("--pdb", help="structure for the true contact map")
    ap.add_argument("--from-npz", "--replot", dest="from_npz", action="store_true")
    ap.add_argument("--output-dir", default=DEFAULT_OUT)
    ap.add_argument("--name", default="4K6E")
    a = ap.parse_args()

    if a.from_npz:
        for path in (os.path.join(FIGURE_DATA, CACHE), os.path.join(a.output_dir, CACHE)):
            if os.path.exists(path):
                print(f"  reading {path}")
                z = np.load(path)
                plot(z["couplings"], z["contacts"], a.output_dir, a.name)
                return
        raise SystemExit(f"No {CACHE} found. Run with --jacobian and --pdb to build it.")

    if not (a.jacobian and a.pdb):
        raise SystemExit("need --jacobian and --pdb, or --from-npz")
    print(f"  reducing {a.jacobian}")
    couplings = jac_to_couplings(np.load(a.jacobian))
    contacts = cbeta_contacts(a.pdb)
    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, CACHE)
    np.savez_compressed(out, couplings=couplings.astype(np.float32), contacts=contacts)
    print(f"  wrote {out} ({os.path.getsize(out)/1024:.0f} KB)")
    plot(couplings, contacts, a.output_dir, a.name)


if __name__ == "__main__":
    main()
