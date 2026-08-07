"""Acceptance-degradation analysis: what does a more sophisticated accept/reject energy cost?

Motivating claim (reviewer framing): MCMC acceptance is governed by how well the PROPOSAL matches
the target ``pi ~ exp(-E)``. Add energy terms (LM -> LM+structure) and ``pi`` sharpens onto the
region that is good on *every* term at once. A **uniform** proposal is maximally mismatched, so each
added term multiplies the rejection (acceptance compounds down); an **informed** proposal that
already sits in the good region barely notices. So proposal quality sets the *slope* of cost vs.
energy sophistication — and a trained generator (PEINT), proposing whole plausible descendants, is
the high-quality proposal that keeps acceptance up even under a sophisticated energy.

This benchmark measures the **site-wise 2x2** analytically from two per-position conditionals we
already have working:

* **LM term**        — ESM2 masked conditional ``p_LM(a | context)``  (``paper.esm_mcmc``); this is
                       exactly the accept/reject criterion the ESM2-MCMC simulator uses.
* **structure term** — ESM-IF per-position conditional ``p_S(a | backbone)`` on the GT structure
                       (``paper.esmif``); the "structural projection" (cf. lm-design's structure loss).

At each root position (aligned to the GT structure) with current residue ``cur``, using
``p_LM``/``p_S`` softmaxed over the 20 residues and a structure weight ``lam``:

* uniform -> LM         :  mean_{a!=cur} min(1, p_LM(a)/p_LM(cur))              (~ ESM2-MCMC's acceptance)
* uniform -> LM+struct  :  mean_{a!=cur} min(1, [p_LM(a)/p_LM(cur)]*[p_S(a)/p_S(cur)]^lam)
* informed -> LM        :  1                                                    (Gibbs to the LM target)
* informed -> LM+struct :  sum_a p_LM(a) * min(1, (p_S(a)/p_S(cur))^lam)        (pays only the structure mismatch)

The headline is the **degradation ratio** (acceptance_LM / acceptance_LM+struct) per proposal: large
for uniform (the naive ESM2-MCMC pays 3-5x), small for informed. The 'informed' column is the
site-wise stand-in for a high-quality proposal; PEINT is its whole-sequence embodiment (see the
whole-sequence panel in ``esm_mcmc_eval`` follow-up).

Run::  python -m benchmarks.esm_mcmc_energy_cost [--families ...] [--lam 1.0]
Output -> figures/output/esm_mcmc/eval/energy_cost.csv  (+ printed table).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg

_STD = "ACDEFGHIKLMNPQRSTVWY"
_IDX = {a: i for i, a in enumerate(_STD)}
OUT_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "eval"


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _aligned_pairs(root_seq: str, gt_seq: str):
    """(root_idx, gt_idx) pairs where root aligns to GT, via one pairwise alignment (biotite)."""
    import biotite.sequence as bseq
    import biotite.sequence.align as balign

    def clean(s):  # ESM-IF's gt sequence can carry non-standard letters; map them to X for aligning
        return "".join(c if c in _STD else "X" for c in s.upper())

    matrix = balign.SubstitutionMatrix.std_protein_matrix()
    aln = balign.align_optimal(
        bseq.ProteinSequence(clean(root_seq)), bseq.ProteinSequence(clean(gt_seq)),
        matrix, gap_penalty=(-10, -1),
    )[0]
    return [(int(r), int(g)) for r, g in aln.trace if r >= 0 and g >= 0]


def family_acceptances(fam: str, lam: float) -> dict:
    """Compute the site-wise 2x2 acceptance means for one family (averaged over aligned positions)."""
    from paper import esmif  # applies the biotite shim before invf import
    import esm.inverse_folding as invf
    from paper import esm_mcmc
    from protevo.utils import read_msa

    root_seq = next(iter(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")).values()))
    coords, gt_seq = invf.util.load_coords(
        os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"), fam.split("_")[-1]
    )
    coord_finite = np.all(np.isfinite(np.asarray(coords, dtype=np.float32)), axis=(-1, -2))

    pairs = [(r, g) for r, g in _aligned_pairs(root_seq, gt_seq)
             if root_seq[r] in _IDX and g < len(coord_finite) and coord_finite[g]]
    if not pairs:
        raise ValueError("no scorable aligned positions")

    lm_logits = esm_mcmc.masked_conditional_logits(root_seq, positions=[r for r, _ in pairs])  # [P,20]
    st_all = esmif.per_position_logits(coords, gt_seq)                                          # [Lgt,20]
    st_logits = st_all[[g for _, g in pairs]]                                                   # [P,20]

    p_lm = _softmax(lm_logits)
    p_s = _softmax(st_logits)
    u_lm, u_ls, i_lm, i_ls = [], [], [], []
    for k, (r, _g) in enumerate(pairs):
        c = _IDX[root_seq[r]]
        r_lm = p_lm[k] / p_lm[k, c]                        # [20]  p_LM(a)/p_LM(cur)
        r_s = (p_s[k] / p_s[k, c]) ** lam                  # [20]  (p_S(a)/p_S(cur))^lam
        others = [a for a in range(20) if a != c]
        u_lm.append(np.mean(np.minimum(1.0, r_lm[others])))
        u_ls.append(np.mean(np.minimum(1.0, r_lm[others] * r_s[others])))
        i_lm.append(1.0)                                   # Gibbs proposal to the LM target
        i_ls.append(float(np.sum(p_lm[k] * np.minimum(1.0, r_s))))
    return {
        "family": fam, "n_sites": len(pairs),
        "uniform_LM": float(np.mean(u_lm)), "uniform_LMstruct": float(np.mean(u_ls)),
        "informed_LM": float(np.mean(i_lm)), "informed_LMstruct": float(np.mean(i_ls)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--lam", type=float, default=1.0, help="structure-term weight in E = LM + lam*struct")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.families:
        families = args.families
    else:
        from paper.generalization import eval_families
        def ok(f):
            return all(os.path.exists(p) for p in (
                os.path.join(str(cfg.ROOT_SEQ_DIR), f"{f}.txt"),
                os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{f}.pdb")))
        families = [f for f in eval_families() if ok(f)][: args.limit]

    print(f"energy-cost 2x2 over {len(families)} families, lam={args.lam}")
    rows = []
    for fam in families:
        try:
            rows.append(family_acceptances(fam, args.lam))
            r = rows[-1]
            print(f"  {fam}: uniform LM {r['uniform_LM']:.3f} -> LM+struct {r['uniform_LMstruct']:.3f} "
                  f"(x{r['uniform_LM']/max(r['uniform_LMstruct'],1e-9):.1f}) | "
                  f"informed LM {r['informed_LM']:.3f} -> LM+struct {r['informed_LMstruct']:.3f} "
                  f"(x{r['informed_LM']/max(r['informed_LMstruct'],1e-9):.1f})")
        except Exception as e:  # pragma: no cover
            print(f"  {fam}: skip ({e})")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "energy_cost.csv", index=False)
    m = df[["uniform_LM", "uniform_LMstruct", "informed_LM", "informed_LMstruct"]].mean()
    print("\n=== mean over families (acceptance) ===")
    print(f"                 LM-only    LM+struct   degradation(LM/LM+struct)")
    print(f"  uniform  :     {m.uniform_LM:.3f}      {m.uniform_LMstruct:.3f}       "
          f"x{m.uniform_LM/max(m.uniform_LMstruct,1e-9):.1f}")
    print(f"  informed :     {m.informed_LM:.3f}      {m.informed_LMstruct:.3f}       "
          f"x{m.informed_LM/max(m.informed_LMstruct,1e-9):.1f}")
    print(f"\nWrote {OUT_DIR / 'energy_cost.csv'}")


if __name__ == "__main__":
    main()
