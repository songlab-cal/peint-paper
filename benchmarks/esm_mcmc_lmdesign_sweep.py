"""Map the evolution<->design spectrum: sweep length-independent LM weight lambda on one family,
and at each lambda measure acceptance, divergence, JSD (conservation) AND ESM-IF (structure).

lambda = per-move LM coefficient (no /L); struct_w=3 fixed. High lambda = strong LM filter =
evolutionary fidelity (Bitbol-like); low lambda = LM off = inverse-folding/design (lm-design-like).
Prediction along decreasing lambda: acceptance up, identity-to-root -> random, JSD -> worse
(conservation washed out), ESM-IF flat/high (structure term is all that filters; also circular).
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
import paper_config as cfg
from protevo.utils import write_msa
from paper import esm_mcmc, lmdesign_energy as lm
from benchmarks import esm_mcmc_lmdesign_simulate as sim
from benchmarks import esm_mcmc_eval as ev

FAM = "1a2t_1_A"                       # L=127; lm-design's effective LM weight here = 2/127 = 0.016
LAMS = [1.0, 0.3, 0.1, 0.03, 0.0]
SWEEP = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "lmdesign" / "sweep"


def jsd_esmif(lam_tag, seqdir):
    """JSD (vs Real) and ESM-IF LL of the lm-design leaves at this lambda, plus reference models."""
    ev.MCMC_MODEL = "lm-design"
    ev.FRAME_DIR = SWEEP / f"frame_{lam_tag}"
    frame = ev.build_frame_msas([FAM], seqdir)
    jp = ev.run_jsd([FAM], frame).set_index("model")["jsd"]                    # model -> JSD
    ll = ev.esmif_comparison(ev.run_esmif_a1([FAM], frame), [FAM]).loc[FAM]    # model -> ESM-IF LL
    return jp, ll


def main():
    esm_mcmc.load_model("esm2_t33_650M_UR50D")
    alphabet = esm_mcmc._ALPHABET
    proj, _, sdev = lm.load_struct_model()
    SWEEP.mkdir(exist_ok=True)
    print(f"lambda sweep on {FAM} (struct_w=3, T=1); lm-design endpoint ~ lambda=2/L=0.016\n", flush=True)

    rows, ref = [], None
    for lam in LAMS:
        named, st = sim.simulate_family(FAM, alphabet, proj, sdev, lam=lam, struct_w=3.0,
                                        max_batch=96, log=lambda *a: None)
        lam_tag = str(lam).replace(".", "_").replace("-", "m")
        seqdir = SWEEP / f"seqs_{lam_tag}"
        seqdir.mkdir(exist_ok=True)
        write_msa(named, str(seqdir / f"{FAM}.txt"))
        jp, ll = jsd_esmif(lam_tag, seqdir)
        rows.append({"lambda": lam, "acceptance": st["acceptance"], "id_to_root": st["leaf_identity_to_root"],
                     "JSD": float(jp.get("lm-design", np.nan)), "ESM_IF_LL": float(ll.get("lm-design", np.nan))})
        if ref is None:
            ref = {"PEINT_JSD": float(jp.get("PEINT (Progressive)", np.nan)),
                   "Real_JSD": float(jp.get("Real (other split)", np.nan)),
                   "PEINT_LL": float(ll.get("PEINT", np.nan)), "Real_LL": float(ll.get("Real", np.nan))}
        r = rows[-1]
        print(f"  lambda={lam:<5} acc {r['acceptance']:.3f} | id-to-root {r['id_to_root']:.3f} | "
              f"JSD {r['JSD']:.3f} | ESM-IF {r['ESM_IF_LL']:+.2f} | {st['seconds']:.0f}s", flush=True)

    pd.DataFrame(rows).to_csv(SWEEP / "sweep.csv", index=False)
    print("\n=== spectrum (evolution -> design), 1a2t ===")
    print("  reference — PEINT: JSD {:.3f} LL {:+.2f} | Real: JSD {:.3f} LL {:+.2f}".format(
        ref["PEINT_JSD"], ref["PEINT_LL"], ref["Real_JSD"], ref["Real_LL"]))
    print("  reference — Bitbol(1a2t): JSD 0.205 LL -3.12 | empirical id-to-root 0.345")


if __name__ == "__main__":
    main()
