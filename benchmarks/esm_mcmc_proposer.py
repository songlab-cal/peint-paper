"""PEINT vs. uniform as MCMC *proposers*, scored by the literal lm-design energy (reviewer response).

Tests the framing that a trained generator is the high-quality **proposal distribution** that keeps
an energy-guided MCMC affordable as the accept/reject criterion gets more sophisticated, where a
naive uniform proposer's acceptance collapses.

For a set of families (hash-seeded random subset of the eval set), we take the simulation root as
the MCMC parent and draw whole-sequence proposals two ways:

* **PEINT**   — ``model.generate`` descendants of the root at branch length ``t`` (the paper's
                simulator's generator; no likelihood filtering, matching the paper protocol).
* **uniform** — root + ``k`` random substitutions, ``k`` matched to PEINT's realised divergence
                so the two proposers are compared at the same distance from the parent.

Each proposal is scored by the three lm-design energy terms (:mod:`paper.lmdesign_energy`): LM
(ESM2 pseudo-LL), structure (``dist_cce_pos`` on the GT backbone), n-gram KL. Under an
independence-Metropolis step vs. the root, acceptance ``= min(1, exp(-(E_prop - E_root)))``; we
report it under LM-only, LM+struct, and the full energy, per proposer, plus the **efficiency gap**
(acceptance_PEINT / acceptance_uniform = the wall-clock ratio at equal per-trial cost) and how it
widens as terms are added.

Assets: ``scripts/fetch_lmdesign_assets.sh`` (distogram weights + n-gram stats). Run::

    python -m benchmarks.esm_mcmc_proposer --n-families 30 --n 16 --phrase "peint-vs-esm"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd

import paper_config as cfg
from protevo.utils import read_msa
from paper import esmif  # applies the biotite<->fair-esm shim before esm.inverse_folding is imported
from paper import esm_mcmc, lmdesign_energy as lm

_STD = "ACDEFGHIKLMNPQRSTVWY"
OUT_DIR = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "eval"


def _has_inputs(fam: str) -> bool:
    return all(os.path.exists(p) for p in (
        os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt"),
        os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"),
    ))


def select_families(phrase: str, n: int) -> list:
    """Deterministic hash-seeded random subset of the runnable eval families."""
    from paper.generalization import eval_families
    fams = [f for f in eval_families() if _has_inputs(f)]
    seed = int(hashlib.md5(phrase.encode("utf-8")).hexdigest(), 16) % (2 ** 32 - 1)
    rng = np.random.RandomState(seed)
    rng.shuffle(fams)
    return sorted(fams[:n])


def peint_proposals(root_seq, n, t, model, vocab, device):
    import torch
    x = torch.tensor([vocab.cls_idx] + vocab.encode(root_seq) + [vocab.eos_idx]).unsqueeze(0).repeat(n, 1).to(device)
    tt = torch.full((n, 1), float(t), device=device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        outs = model.generate(x=x, t=tt, device=device, max_decode_steps=2 * len(root_seq), p=1.0)
    if hasattr(model, "_reset_kv_cache"):
        model._reset_kv_cache()
    return [o for o in outs if 0 < len(o) <= 2 * len(root_seq)]


def uniform_proposals(root_seq, n, k, rng):
    L = len(root_seq)
    props = []
    for _ in range(n):
        s = list(root_seq)
        for pos in rng.sample(range(L), min(k, L)):
            s[pos] = rng.choice([a for a in _STD if a != s[pos]])
        props.append("".join(s))
    return props


def _identity_to_root(seq, root):
    if len(seq) == len(root):
        return float(np.mean([a == b for a, b in zip(seq, root)]))
    import biotite.sequence as bseq, biotite.sequence.align as bal
    cl = lambda s: "".join(c for c in s.upper() if c in _STD)
    aln = bal.align_optimal(bseq.ProteinSequence(cl(seq)), bseq.ProteinSequence(cl(root)),
                            bal.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1))[0]
    return float(bal.get_sequence_identity(aln))


def _accept(gaps):
    return float(np.mean(np.minimum(1.0, np.exp(-np.asarray(gaps)))))


def summarize(df: pd.DataFrame) -> dict:
    """Per-proposer acceptance under LM / +struct / +full, and the PEINT/uniform efficiency gap."""
    res = {}
    for prop in ["PEINT", "uniform"]:
        d = df[df.proposer == prop]
        res[prop] = {
            "n": len(d),
            "dLM": d.dLM.mean(), "dStruct": d.dStruct.mean(), "dNgram": d.dNgram.mean(),
            "acc_LM": _accept(d.dLM),
            "acc_LMstruct": _accept(d.dLM + d.dStruct),
            "acc_full": _accept(d.dLM + d.dStruct + d.dNgram),
        }
        res[prop]["degrade"] = res[prop]["acc_LM"] / max(res[prop]["acc_full"], 1e-9)
    if res["PEINT"]["n"] and res["uniform"]["n"]:
        res["gap_LM"] = res["PEINT"]["acc_LM"] / max(res["uniform"]["acc_LM"], 1e-9)
        res["gap_LMstruct"] = res["PEINT"]["acc_LMstruct"] / max(res["uniform"]["acc_LMstruct"], 1e-9)
        res["gap_full"] = res["PEINT"]["acc_full"] / max(res["uniform"]["acc_full"], 1e-9)
    return res


def write_report(res, df, families, args):
    lines = ["# PEINT vs. uniform proposer under the lm-design energy\n",
             f"Whole-sequence proposers scored by the literal lm-design energy (LM + structure + "
             f"n-gram). {len(families)} hash-seeded families (phrase `{args.phrase}`), "
             f"{args.n} proposals/proposer/family at branch length t={args.t}, divergence-matched.\n",
             "\n## Independence-Metropolis acceptance vs. the root (higher = fewer trials)\n",
             "| proposer | LM-only | +struct | +ngram (full) | degradation (LM/full) |",
             "|---|---|---|---|---|"]
    for p in ["PEINT", "uniform"]:
        r = res[p]
        lines.append(f"| {p} | {r['acc_LM']:.3f} | {r['acc_LMstruct']:.3f} | {r['acc_full']:.3f} | x{r['degrade']:.1f} |")
    lines.append("\n## Efficiency gap = acceptance_PEINT / acceptance_uniform (the wall-clock ratio)\n")
    lines.append(f"- LM-only: **{res.get('gap_LM', float('nan')):.1f}x**")
    lines.append(f"- LM+struct: **{res.get('gap_LMstruct', float('nan')):.1f}x**")
    lines.append(f"- full energy: **{res.get('gap_full', float('nan')):.1f}x**  "
                 "(the gap widens as the energy gets more sophisticated)")
    lines.append("\n## Mean per-term energy gap (proposal - root; lower = closer to the parent)\n")
    lines.append("| proposer | ΔLM | ΔStruct | ΔNgram |")
    lines.append("|---|---|---|---|")
    for p in ["PEINT", "uniform"]:
        r = res[p]
        lines.append(f"| {p} | {r['dLM']:+.3f} | {r['dStruct']:+.3f} | {r['dNgram']:+.3f} |")
    lines.append("\n_Energy = literal lm-design criterion (vendored, `paper/vendor/lm_design/`); "
                 "unit weights (LM/L + struct + ngram), which is conservative vs. lm-design's tuned "
                 "weights. Structure term scored on the GT backbone with the proposal threaded onto "
                 "its frame._\n")
    (OUT_DIR / "REBUTTAL_esm_mcmc_proposer.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--families", nargs="*", default=None, help="explicit family list (overrides seeding)")
    ap.add_argument("--n-families", type=int, default=30, help="hash-seeded random subset size")
    ap.add_argument("--phrase", default="peint-vs-esm-mcmc-proposer", help="hash phrase for family seeding")
    ap.add_argument("--n", type=int, default=16, help="proposals per proposer per family")
    ap.add_argument("--t", type=float, default=0.5, help="PEINT branch length for proposals")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed for uniform proposals")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    families = args.families or select_families(args.phrase, args.n_families)
    print(f"proposer experiment | {len(families)} families | n={args.n} t={args.t}")
    print(f"families: {', '.join(families)}")

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    esm_mcmc.load_model("esm2_t33_650M_UR50D")
    lm.load_struct_model()
    from protevo.models._loading import load_model as peint_load
    peint_model, vocab = peint_load(str(cfg.PEINT_CHECKPOINT), use_cached_model=True, device=device)
    rng = random.Random(args.seed)

    rows = []
    for i, fam in enumerate(families, 1):
        try:
            import esm.inverse_folding as invf
            root = next(iter(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")).values()))
            coords, gt_seq = invf.util.load_coords(
                os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"), fam.split("_")[-1])
            labels, contact = lm.target_distogram(coords)
            peint = peint_proposals(root, args.n, args.t, peint_model, vocab, device)
            if not peint:
                print(f"[{i}/{len(families)}] {fam}: no PEINT proposals, skip"); continue
            div = float(np.mean([1 - _identity_to_root(p, root) for p in peint]))
            k = max(1, int(round(div * len(root))))
            unif = uniform_proposals(root, args.n, k, rng)
        except Exception as e:  # pragma: no cover
            print(f"[{i}/{len(families)}] {fam}: skip ({e})"); continue

        e0 = lm.energy_terms(root, gt_seq, labels, contact)
        for label, props in [("PEINT", peint), ("uniform", unif)]:
            for s in props:
                e = lm.energy_terms(s, gt_seq, labels, contact)
                rows.append({"family": fam, "proposer": label,
                             "dLM": e["lm"] - e0["lm"], "dStruct": e["struct"] - e0["struct"],
                             "dNgram": e["ngram"] - e0["ngram"]})
        res = summarize(pd.DataFrame(rows))
        print(f"[{i}/{len(families)}] {fam}: L={len(root)} div~{div:.2f} k={k} | "
              f"acc PEINT {res['PEINT']['acc_full']:.3f} vs uniform {res['uniform']['acc_full']:.3f} "
              f"(full) | gap {res.get('gap_full', float('nan')):.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "proposer_energy_gaps.csv", index=False)
    res = summarize(df)
    write_report(res, df, families, args)
    print("\n=== FINAL (pooled) ===")
    for p in ["PEINT", "uniform"]:
        r = res[p]
        print(f"  {p:8s} acc  LM {r['acc_LM']:.3f}  +struct {r['acc_LMstruct']:.3f}  full {r['acc_full']:.3f}  "
              f"(degrade x{r['degrade']:.1f}) | dLM {r['dLM']:+.2f} dStruct {r['dStruct']:+.2f} dNgram {r['dNgram']:+.2f}")
    print(f"  efficiency gap PEINT/uniform: LM {res.get('gap_LM', float('nan')):.1f}x  "
          f"+struct {res.get('gap_LMstruct', float('nan')):.1f}x  full {res.get('gap_full', float('nan')):.1f}x")
    print(f"wrote {OUT_DIR/'proposer_energy_gaps.csv'} + REBUTTAL_esm_mcmc_proposer.md")


if __name__ == "__main__":
    main()
