"""The literal lm-design accept/reject energy, for the ESM-MCMC proposer experiment.

Reproduces the three energy terms of ESM ``lm-design`` (``calc_total_loss``) so we can score an
arbitrary sequence and use it as an MCMC accept/reject criterion:

* **LM**     — ESM2 masked pseudo-LL nll, ``mean_i -log p(x_i | x_\\i)`` (mask-1-out), the
               ``calc_sequence_loss`` term. Reuses the ESM2 model from :mod:`paper.esm_mcmc`.
* **struct** — ``dist_cce_pos``: the vendored ``LinearProjectionDistogramModel`` (a small learned
               affine map on frozen-ESM2 attention maps -> an 18-bin Cβ-Cβ distogram) scored by
               categorical cross-entropy against the ground-truth distogram, averaged over target
               contacts. The proposal is threaded onto the GT frame first, so indels are fine.
* **ngram**  — ``sum_{order 1..4} KL(observed n-gram || background)``, reimplemented ``nltk``-free.

The distogram projection + n-gram background stats are fetched by
``scripts/fetch_lmdesign_assets.sh`` into ``data/lm_design/`` (see ``paper_config``); the model
code is vendored under ``paper/vendor/lm_design/`` (MIT — see its ``NOTICE.md``).
"""

from __future__ import annotations

import os
import pickle
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import paper_config as cfg
from paper import esm_mcmc
from paper.vendor.lm_design.linear_projection import LinearProjectionDistogramModel
from paper.vendor.lm_design.loss import get_cce_loss
from paper.vendor.lm_design.pdb_loader import get_coords6d

_STD = "ACDEFGHIKLMNPQRSTVWY"
# lm-design 'LinearProjectionDist-1A' distance binning: linspace(2.5, 20, 17) -> digitize -> 18 bins;
# a pair is a "contact" if its distance bin is 0..5 (Cβ-Cβ < ~8 Å).
_DBINS = np.linspace(2.5, 20.0, 17)
_CONTACT_MAXBIN = 5
# n-gram alphabet ordering the background stats were built in (utils/ngram.py).
_NGRAM_ENCODE = list("LAGVSERTIDPKQNFYMHWC")

_STRUCT = None
_NGRAM = None


# ======================================================================================
# structure term
# ======================================================================================
def load_struct_model(base_model: Optional[torch.nn.Module] = None):
    """Load (once) the distogram projection, reusing the ESM2 base from :mod:`paper.esm_mcmc`.

    The projection state dict is ~12k params; the ESM2 backbone is shared with the LM term so we
    do not hold two copies of ESM2-650M.
    """
    global _STRUCT
    if _STRUCT is None:
        model, alphabet, device = esm_mcmc.load_model("esm2_t33_650M_UR50D")
        state = torch.load(str(cfg.require(cfg.LMDESIGN_WEIGHTS)), map_location="cpu")
        proj = LinearProjectionDistogramModel()
        proj.load_state_dict(state["model"])
        proj.base_model = base_model if base_model is not None else model
        _STRUCT = (proj.eval().to(device), alphabet, device)
    return _STRUCT


def struct_pdist(seq: str) -> torch.Tensor:
    """Predicted distogram distribution ``p(d_ij | seq)`` of shape ``[1, L, L, 18]``."""
    proj, alphabet, device = load_struct_model()
    idx = [alphabet.cls_idx] + [alphabet.get_idx(c) for c in seq] + [alphabet.eos_idx]
    oneh = F.one_hot(torch.tensor(idx, device=device), len(alphabet)).float().unsqueeze(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        out = proj(oneh)
    return F.softmax(out["logits"].permute(0, 2, 3, 1).float(), dim=-1)   # [1, L, L, 18]


def target_distogram(coords: np.ndarray):
    """GT Cβ distogram bin labels ``[L, L]`` (0..17) and off-diagonal contact mask ``[L, L]``.

    ``coords`` is ``[L, 3, 3]`` (N, CA, C), as returned by ``esm.inverse_folding.util.load_coords``.
    """
    proj, _, device = load_struct_model()
    xyz = np.transpose(np.asarray(coords, dtype=np.float64), (1, 0, 2))    # [3, L, 3]
    d, _, _, _ = get_coords6d(xyz, dmax=20.0, allow_missing_residue_coords=True)
    labels = np.digitize(np.nan_to_num(d, nan=999.9), _DBINS).astype(np.int64)   # [L, L]
    L = labels.shape[0]
    contact = (labels <= _CONTACT_MAXBIN) & ~np.eye(L, dtype=bool)
    return (torch.tensor(labels, device=device),
            torch.tensor(contact, device=device, dtype=torch.bool))


def struct_cce(threaded_seq: str, labels: torch.Tensor, contact: torch.Tensor) -> float:
    """``dist_cce_pos``: CCE of the predicted distogram vs GT labels over the contact mask."""
    p = struct_pdist(threaded_seq)                                          # [1, L, L, 18]
    return float(get_cce_loss(p, labels[None], contact[None].float())[0])


def thread_to_gt(seq: str, gt_seq: str) -> str:
    """Thread a (variable-length) sequence onto the GT frame: aligned columns take the sequence's
    residue, the rest keep the GT residue. Substitution- and indel-safe (one pairwise alignment)."""
    import biotite.sequence as bseq
    import biotite.sequence.align as balign
    clean = lambda s: "".join(c if c in _STD else "X" for c in s.upper())
    aln = balign.align_optimal(
        bseq.ProteinSequence(clean(seq)), bseq.ProteinSequence(clean(gt_seq)),
        balign.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1),
    )[0]
    out, cs = list(gt_seq), clean(seq)
    for p, g in aln.trace:
        if p >= 0 and g >= 0:
            out[g] = cs[p]
    return "".join(out)


# ======================================================================================
# LM term
# ======================================================================================
def lm_nll(seq: str, max_batch: int = 256) -> float:
    """ESM2 masked pseudo-LL nll per residue: ``mean_i -log p(x_i | x_\\i)`` (mask-1-out)."""
    _, alphabet, device = esm_mcmc.load_model("esm2_t33_650M_UR50D")
    base = esm_mcmc._encode(seq, alphabet, device)
    L = len(seq)
    nlls = np.zeros(L)
    for start in range(0, L, max_batch):
        pos = list(range(start, min(start + max_batch, L)))
        toks = base.unsqueeze(0).repeat(len(pos), 1).clone()
        for r, p in enumerate(pos):
            toks[r, p + 1] = alphabet.mask_idx
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = esm_mcmc.model_forward_logits(toks).float()
        lp = torch.log_softmax(logits, dim=-1)
        for r, p in enumerate(pos):
            nlls[p] = -float(lp[r, p + 1, alphabet.get_idx(seq[p])])
    return float(nlls.mean())


# ======================================================================================
# n-gram term (nltk-free reimplementation of utils/ngram.py)
# ======================================================================================
def _load_ngram_background():
    global _NGRAM
    if _NGRAM is None:
        base = str(cfg.require(cfg.LMDESIGN_NGRAM_DIR))
        out = []
        for fn in ["monogram_seg.p", "bigram_seg.p", "trigram_seg.p", "quadgram_seg.p"]:
            d = pickle.load(open(os.path.join(base, fn), "rb"))
            idx = {tuple(_NGRAM_ENCODE.index(ki) for ki in k): v
                   for k, v in d.items() if all(ki in _NGRAM_ENCODE for ki in k)}
            tot = sum(idx.values())
            out.append({k: max(v / tot, 1e-5) for k, v in idx.items()})
        _NGRAM = out
    return _NGRAM


def ngram_kl(seq: str, orders=(1, 2, 3, 4)) -> float:
    """Sum over n-gram orders of ``KL(observed || background)`` (lm-design's ``calc_ngram_loss``)."""
    bg = _load_ngram_background()
    enc = [_NGRAM_ENCODE.index(a) for a in seq if a in _NGRAM_ENCODE]
    total = 0.0
    for order in orders:
        grams = Counter(tuple(enc[i:i + order]) for i in range(len(enc) - order + 1))
        n = sum(grams.values())
        if not n:
            continue
        obs = {k: v / n for k, v in grams.items()}
        p = np.array(list(obs.values()))
        q = np.array([bg[order - 1].get(k, 1e-5) for k in obs.keys()])
        total += float(np.sum(p * np.log(p / q)))
    return total


# ======================================================================================
# combined energy
# ======================================================================================
def energy_terms(seq: str, gt_seq: str, labels, contact) -> dict:
    """The three lm-design energy terms for one sequence (structure scored on the GT frame)."""
    return {
        "lm": lm_nll(seq),
        "struct": struct_cce(thread_to_gt(seq, gt_seq), labels, contact),
        "ngram": ngram_kl(seq),
    }
