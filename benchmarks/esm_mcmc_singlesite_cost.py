"""FAITHFUL lm-design single-site MCMC (fixes the 0.98 artifact).

Corrects two errors in singlesite_cost.py: (1) it used a GUIDED, self-inclusive proposal (ESM2
conditional, often re-drawing the current residue -> trivial accepts -> inflated 0.98); lm-design's
stage_fixedbb uses a UNIFORM proposal. (2) it computed the LM term as the full L-forward pseudo-LL;
lm-design masks only the mutated position (mask={i}) -> ONE forward for CE at i.

Faithful scheme here (per lm-design conf/config.yaml fixedbb): random position, UNIFORM proposal
over the 19 *other* residues (genuine mutations; lm-design includes self at 1/20, excluded here so
acceptance measures real moves), Metropolis on the full energy
    E = LM_w/L * CE_i  +  struct_w * dist_cce_pos  +  ngram_w * ngram_KL(orders 1-3)
with LM_w=2, struct_w=3, ngram_w=1. Sampled at fixed T=1 (lm-design anneals from 8 for optimisation;
T=1 is the sampling temperature). Cost = 2 forwards/step (1 masked LM + 1 distogram). We walk until
Hamming-from-root = k and report per-step acceptance + forwards per branch.
"""
import os, sys, math, random
import numpy as np
import torch

import paper_config as cfg
from peint.utils import read_msa
from paper import esmif  # biotite shim
from paper import esm_mcmc
from paper import lmdesign_energy as lm

_STD = "ACDEFGHIKLMNPQRSTVWY"
_IDX = {a: i for i, a in enumerate(_STD)}
DEV = "cuda" if torch.cuda.is_available() else "cpu"
FAMS = ["1a2t_1_A", "1acf_1_A", "1amx_1_A", "1bja_1_A"]
DIV = 0.38
LM_W, STRUCT_W, NGRAM_W = 2.0, 3.0, 1.0
NGRAM_ORDERS = (1, 2, 3)
TEMP = 1.0


def lmdesign_single_site(root, gt, labels, contact, k, rng, max_steps):
    L = len(root)
    state = list(root)
    cce = lm.struct_cce(lm.thread_to_gt(root, gt), labels, contact)   # structure energy of current state
    ng = lm.ngram_kl(root, orders=NGRAM_ORDERS)
    steps = accepts = forwards = 0
    ham = 0
    absLM, absS, absN = [], [], []      # |per-move energy contribution| of each term
    while ham < k and steps < max_steps:
        i = rng.randrange(L)
        cur = state[i]
        new = rng.choice([a for a in _STD if a != cur])              # UNIFORM over 19 others
        # LM term: one masked forward at i -> logits over 20 std AAs (softmax-invariant ratio)
        logits = esm_mcmc.masked_conditional_logits("".join(state), positions=[i])[0]
        forwards += 1
        dLM = (LM_W / L) * float(logits[_IDX[cur]] - logits[_IDX[new]])   # E_LM(xp)-E_LM(x)
        cand = state.copy(); cand[i] = new
        cce_xp = lm.struct_cce(lm.thread_to_gt("".join(cand), gt), labels, contact)
        forwards += 1
        dStruct = STRUCT_W * (cce_xp - cce)
        ng_xp = lm.ngram_kl("".join(cand), orders=NGRAM_ORDERS)
        dNgram = NGRAM_W * (ng_xp - ng)
        dE = dLM + dStruct + dNgram
        absLM.append(abs(dLM)); absS.append(abs(dStruct)); absN.append(abs(dNgram))
        steps += 1
        if rng.random() < min(1.0, math.exp(-dE / TEMP)):
            state, cce, ng = cand, cce_xp, ng_xp
            accepts += 1
        ham = sum(a != b for a, b in zip(state, root))
    return {"steps": steps, "accepts": accepts, "accept_rate": accepts / max(steps, 1),
            "forwards": forwards, "ham": ham, "k": k,
            "abs_dLM": float(np.mean(absLM)), "abs_dStruct": float(np.mean(absS)),
            "abs_dNgram": float(np.mean(absN))}


def main():
    import esm.inverse_folding as invf
    esm_mcmc.load_model("esm2_t33_650M_UR50D")
    lm.load_struct_model()
    rng = random.Random(0)
    rows = []
    for fam in FAMS:
        root = next(iter(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")).values()))
        coords, gt = invf.util.load_coords(os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"),
                                           fam.split("_")[-1])
        labels, contact = lm.target_distogram(coords)
        L = len(root); k = max(1, int(round(DIV * L)))
        r = lmdesign_single_site(root, gt, labels, contact, k, rng, max_steps=40 * k)
        rows.append({"family": fam, "L": L, **r})
        print(f"[{fam}] L={L} k={k} | per-step accept {r['accept_rate']:.3f} "
              f"({r['accepts']}/{r['steps']}) | forwards {r['forwards']} to reach ham {r['ham']}/{k}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    out_dir = os.path.join(str(cfg.FIGURES_DIR), "esm_mcmc", "eval")
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "singlesite_cost.csv"), index=False)
    print("\n=== FAITHFUL lm-design single-site (uniform proposal + full energy, T=1) ===")
    print(f"  per-step acceptance: {df.accept_rate.mean():.3f} (was mis-reported 0.98)")
    print(f"  forwards per branch: {df.forwards.mean():.0f}  (2/step: 1 masked-LM + 1 distogram)")
    print(f"  vs PEINT one branch ~ L forwards (generate) + 1 guardrail  [e.g. {df.L.mean():.0f}]")
    print("\n=== mean |per-move energy contribution| by term (which term filters?) ===")
    print(f"  LM  (LM_w/L * CE_i): {df.abs_dLM.mean():.4f}")
    print(f"  struct (3 * dist_cce): {df.abs_dStruct.mean():.4f}")
    print(f"  ngram: {df.abs_dNgram.mean():.4f}")
    print(f"  -> struct / LM ratio: {df.abs_dStruct.mean()/max(df.abs_dLM.mean(),1e-9):.1f}x")


if __name__ == "__main__":
    main()
