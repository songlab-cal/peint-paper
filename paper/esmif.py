"""ESM-IF (inverse folding) scoring for the reviewer-requested structural validation.

A reviewer asked us to corroborate the structural metrics with an inverse-folding
model. This module wraps ``esm_if1_gvp4_t16_142M_UR50`` to score, for a given
backbone, either

* the **conditional log-likelihood**  ``ll = mean_i log p(residue_i | backbone)``  or
* the **sequence recovery**           ``mean_i [ argmax_a p(a | backbone) == residue_i ]``

of an arbitrary target sequence, on GPU when available. Two entry points sit on top
of one scorer:

* :func:`score_pdb_self`      — score a PDB chain against **its own** sequence
                                (self-consistency of a predicted structure; Approach 2).
* :func:`score_target_on_coords` — score a threaded target sequence against a fixed
                                backbone, masking positions with no target residue
                                (a homolog on a real GT structure; Approach 1).

INTERPRETATION — two independent confounds, both established up front, because they
change which comparisons are valid (see ``figures/output/esmif/REPORT_esmif.md``):

1. **Memorization.** ESM-IF was trained on ~12M AlphaFold structures of UniRef50
   sequences. The empirical ("Real") leaves are natural sequences ESM-IF has
   effectively seen; PEINT / classical-simulator leaves are synthetic and
   out-of-distribution. A Real score is therefore memorization-inflated relative to
   any simulator. Consequences baked into the reporting:
     - the **memorization-free** comparison is PEINT vs the classical simulators
       (WAG / LG / LG4X / LG+C60 / LG+S256) — all synthetic and equally OOD;
     - the PEINT-vs-Real gap is an **upper bound** on the true gap (the bias runs
       against PEINT), so "PEINT tracks Real" is a conservative statement.

2. **Native is not maximally designable.** Inverse-folding scores are a two-sided
   target, not "higher is better": real native backbones recover ~0.5 (CATH average)
   while idealized de-novo backbones recover ~0.9. A generator that **overshoots** the
   native level (e.g. an over-conserved / consensus-like model) is not better. So we
   report the **signed gap to the Real/native level**, and we score primarily on the
   **real ground-truth structure** (Approach 1) whose non-ideal, native fold is exactly
   the reference we want. We deliberately do **not** use design->refold scTM
   "designability", which is non-monotonic and rewards idealized backbones.

Requires the geometric stack (torch_geometric / torch_scatter / torch_sparse /
torch_cluster) that ESM-IF's GVP-Transformer needs, and biotite; ``fair-esm`` 2.0.0
predates biotite's 1.0 rename of ``filter_backbone``, so we shim it below before the
inverse-folding package is imported.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

# --- biotite>=1.0 compat for fair-esm 2.0.0 (imports the pre-1.0 name at module load) ---
import biotite.structure as _bs

if not hasattr(_bs, "filter_backbone"):
    _bs.filter_backbone = _bs.filter_peptide_backbone

import esm  # noqa: E402  (after the shim, so the inverse_folding import below succeeds)
import esm.inverse_folding as invf  # noqa: E402

# Standard-residue token set: recovery is only meaningful where the target is a real
# amino acid, so placeholder / unknown positions are excluded from the recovery average.
_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

_MODEL = None
_ALPHABET = None
_DEVICE = None


def load_model():
    """Load (once) ``esm_if1_gvp4_t16_142M_UR50`` in eval mode on GPU if available.

    Returns ``(model, alphabet, device)``. Cached at module scope so the 142M-param
    model and its weights are loaded a single time per process.
    """
    global _MODEL, _ALPHABET, _DEVICE
    if _MODEL is None:
        model, alphabet = esm.pretrained.esm_if1_gvp4_t16_142M_UR50()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL, _ALPHABET, _DEVICE = model.eval().to(device), alphabet, device
    return _MODEL, _ALPHABET, _DEVICE


def _standard_aa_mask(seq: str) -> np.ndarray:
    """Boolean mask over ``seq`` positions that are one of the 20 standard residues."""
    return np.array([c in _STANDARD_AA for c in seq], dtype=bool)


def score(coords: np.ndarray, seq: str, score_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Score ``seq`` against backbone ``coords`` (both length L) under ESM-IF.

    ``coords`` is an ``L x 3 x 3`` array of N, CA, C coordinates (NaN rows allowed for
    missing backbone atoms, exactly as ``esm.inverse_folding.util.load_coords`` returns).
    ``score_mask`` (length L, optional) selects the positions to average over — used by
    Approach 1 to score only the columns a homolog actually threads onto the structure;
    when ``None`` every position with a finite backbone and a standard residue is scored.

    Returns a dict with:
      * ``ll``        — mean conditional log-likelihood over the scored positions
                        (matches ``util.score_sequence``'s ``ll_withcoord`` when the whole
                        finite-coord chain is scored);
      * ``ll_fullseq``— mean conditional log-likelihood over all non-pad positions;
      * ``recovery``  — fraction of scored standard-residue positions whose argmax matches;
      * ``n_scored``  — number of positions entering ``ll`` / ``recovery``.
    """
    model, alphabet, device = load_model()
    coords = np.asarray(coords, dtype=np.float32)
    if len(seq) != coords.shape[0]:
        raise ValueError(f"seq length {len(seq)} != coords length {coords.shape[0]}")

    batch_converter = invf.util.CoordBatchConverter(alphabet)
    batch_coords, confidence, _, tokens, padding_mask = batch_converter(
        [(coords, None, seq)], device=device
    )
    prev_output_tokens = tokens[:, :-1]
    target = tokens[:, 1:]
    with torch.no_grad():
        logits, _ = model.forward(batch_coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction="none")[0].cpu().numpy()
    pred = logits.argmax(1)[0].cpu().numpy()
    target = target[0].cpu().numpy()

    not_pad = target != alphabet.padding_idx
    coord_finite = np.all(np.isfinite(coords), axis=(-1, -2))  # per-residue, length L
    aa = _standard_aa_mask(seq)
    if score_mask is None:
        score_mask = np.ones(len(seq), dtype=bool)
    score_mask = np.asarray(score_mask, dtype=bool)

    # `loss`/`pred`/`target`/`not_pad` are over the L target positions (bos stripped by
    # the [:, 1:] shift); the coord/aa/score masks are over the L residues — same length.
    ll_positions = not_pad & coord_finite & score_mask
    rec_positions = ll_positions & aa
    n_scored = int(ll_positions.sum())
    ll = float(-np.sum(loss[ll_positions]) / n_scored) if n_scored else float("nan")
    ll_fullseq = float(-np.sum(loss[not_pad]) / np.sum(not_pad)) if not_pad.any() else float("nan")
    n_rec = int(rec_positions.sum())
    recovery = float((pred[rec_positions] == target[rec_positions]).mean()) if n_rec else float("nan")
    return {"ll": ll, "ll_fullseq": ll_fullseq, "recovery": recovery, "n_scored": n_scored}


def per_position_logits(coords: np.ndarray, seq: str) -> np.ndarray:
    """ESM-IF per-position logits over the 20 standard residues for ``seq`` on ``coords``.

    One forward pass on the backbone; the returned ``[L, 20]`` array (columns in ``_STANDARD_AA``
    order) is the structure-conditioned analogue of ESM2's masked conditional — used as the
    "structural projection" energy term in the acceptance-degradation analysis. Row ``i`` is the
    conditional at residue ``i`` (teacher-forced with ``seq``); a single-site swap's structure
    energy change is ``logit[i, a] - logit[i, cur]``.
    """
    model, alphabet, device = load_model()
    coords = np.asarray(coords, dtype=np.float32)
    batch_converter = invf.util.CoordBatchConverter(alphabet)
    batch_coords, confidence, _, tokens, padding_mask = batch_converter(
        [(coords, None, seq)], device=device
    )
    with torch.no_grad():
        logits, _ = model.forward(batch_coords, padding_mask, confidence, tokens[:, :-1])
    std_idx = [alphabet.get_idx(a) for a in _STANDARD_AA]
    return logits[0, std_idx, :].float().cpu().numpy().T          # [L, 20]


def score_pdb_self(pdb_path: str, chain: Optional[str] = None) -> Dict[str, float]:
    """Approach 2: score a PDB chain against **its own** extracted sequence.

    Reads coordinates and sequence straight from the structure (no MSA / alignment), so a
    predicted OmegaFold structure is self-contained and the residue<->coordinate mapping is
    exact — sidestepping the MSA-offset issue that afflicts per-residue analyses. ``chain``
    defaults to the sole chain in single-chain OmegaFold outputs.
    """
    coords, seq = invf.util.load_coords(pdb_path, chain)
    return score(coords, seq)


def score_target_on_coords(
    coords: np.ndarray, target_seq: str, score_mask: Sequence[bool]
) -> Dict[str, float]:
    """Approach 1: score a threaded ``target_seq`` (length L == len(coords)) on a fixed backbone.

    ``target_seq`` carries the homolog's residues on the columns it aligns to the reference
    structure and a placeholder (any non-standard char, e.g. ``'X'``) elsewhere; ``score_mask``
    marks the real (aligned) positions. The full backbone is kept so the encoder sees intact
    geometry; only the masked positions enter the average. Keeping the whole GT structure as
    the reference is deliberate (see the module docstring, confound #2).
    """
    return score(coords, target_seq, score_mask=np.asarray(score_mask, dtype=bool))


def score_batch(coords: np.ndarray, seqs: Sequence[str], score_masks=None,
                max_batch_l2: int = 25_000_000) -> list:
    """Score many target sequences against ONE shared backbone, adaptively sub-batched.

    Approach 1 threads every model's leaf onto the *same* GT backbone, so the coords are
    identical across a family's sequences; batching them (coords repeated per item) lets the
    GPU score a whole family at once. The decoder's self-attention memory scales with
    ``batch x L^2``, so long proteins would OOM at a fixed batch size — we therefore cap the
    sub-batch at ``max_batch_l2 // L^2`` (>=1), which is a no-op for typical lengths and only
    kicks in past ~L=820. Returns a list of the same per-item dicts as :func:`score`.
    ``score_masks`` is an optional list aligned with ``seqs`` (each a length-L boolean mask);
    ``None`` means score every finite-coord standard-residue position.
    """
    model, alphabet, device = load_model()
    coords = np.asarray(coords, dtype=np.float32)
    if score_masks is None:
        score_masks = [None] * len(seqs)
    L = coords.shape[0]
    coord_finite = np.all(np.isfinite(coords), axis=(-1, -2))
    batch_converter = invf.util.CoordBatchConverter(alphabet)
    sub = max(1, int(max_batch_l2 // max(L * L, 1)))
    out = []
    for start in range(0, len(seqs), sub):
        chunk = seqs[start:start + sub]
        cmask = score_masks[start:start + sub]
        batch_coords, confidence, _, tokens, padding_mask = batch_converter(
            [(coords, None, s) for s in chunk], device=device
        )
        target = tokens[:, 1:]
        with torch.no_grad():
            logits, _ = model.forward(batch_coords, padding_mask, confidence, tokens[:, :-1])
        loss = F.cross_entropy(logits, target, reduction="none").cpu().numpy()
        pred = logits.argmax(1).cpu().numpy()
        target = target.cpu().numpy()
        for i, seq in enumerate(chunk):
            not_pad = target[i] != alphabet.padding_idx
            aa = _standard_aa_mask(seq)
            sm = np.ones(len(seq), bool) if cmask[i] is None else np.asarray(cmask[i], bool)
            ll_pos = not_pad[: len(seq)] & coord_finite & sm
            rec_pos = ll_pos & aa
            n = int(ll_pos.sum())
            nr = int(rec_pos.sum())
            out.append({
                "ll": float(-loss[i][: len(seq)][ll_pos].sum() / n) if n else float("nan"),
                "recovery": float((pred[i][: len(seq)][rec_pos] == target[i][: len(seq)][rec_pos]).mean()) if nr else float("nan"),
                "n_scored": n,
            })
        del logits, loss, pred, target, batch_coords, tokens, padding_mask, confidence
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


# ======================================================================================
# Threading a homolog onto the reference (GT) structure  (Approach 1)
# ======================================================================================
# GOTCHA that cost us a day: in the simulator / PEINT MSAs the row literally named "seq1" is
# NOT the PDB reference chain — it is the simulation ROOT sequence (from root_sequences/, a
# different chain). Only mafft_add/old_sequences carries the true reference as "seq1", and it
# equals the GT structure sequence exactly. All per-model MSAs DO share the old_sequences
# column frame (mafft --add / IQ-TREE `-s MSA`), so map columns -> GT residues from
# old_sequences/seq1 for EVERY model. Using each model's own "seq1" mis-threads the simulated
# leaves (fake ~6% identity vs the true ~27%) and makes PEINT look near-random. Prefer the
# pairwise threader below, which is frame-independent and does not depend on mafft --add quality.

def reference_columns(reference_seq1_gapped: str, gap: str = "-") -> list:
    """Columns of the reference (old_sequences/seq1) that carry a residue, left to right.

    The r-th entry is the alignment column of the r-th reference residue == GT structure
    residue r (old_sequences/seq1 equals the GT sequence exactly). Apply to any model's MSA
    row in the shared frame to read its residue at each GT position.
    """
    return [i for i, ch in enumerate(reference_seq1_gapped) if ch != gap]


def thread_via_reference_frame(leaf_gapped: str, keep_columns: Sequence[int], gap: str = "-"):
    """Thread an aligned leaf onto the GT structure via the shared old_sequences frame.

    Cheap (no per-leaf alignment) but only as good as mafft --add placed the leaf. Returns
    ``(target_seq, score_mask)`` of length ``len(keep_columns) == GT length``.
    """
    L = len(keep_columns)
    target = ["X"] * L
    mask = np.zeros(L, dtype=bool)
    for r, col in enumerate(keep_columns):
        a = leaf_gapped[col]
        if a != gap and a.upper() in _STANDARD_AA:
            target[r] = a.upper()
            mask[r] = True
    return "".join(target), mask


def thread_to_structure(leaf_seq: str, gt_seq: str, gap_open: int = -10, gap_extend: int = -1):
    """Thread a raw (ungapped) homolog onto the GT structure by direct pairwise alignment.

    Frame-independent — bypasses the canonical MSA entirely, so it is immune to the seq1
    mislabelling and to mafft --add misplacement. Returns ``(target_seq, score_mask, identity)``
    where ``target_seq``/``score_mask`` have length ``len(gt_seq)`` and carry the leaf's residue
    on each GT position it aligns to. Returns ``None`` if the leaf has no scorable residues.
    """
    import biotite.sequence as bseq
    import biotite.sequence.align as balign

    clean = "".join(c for c in leaf_seq.upper() if c in _STANDARD_AA)
    if not clean:
        return None
    matrix = balign.SubstitutionMatrix.std_protein_matrix()
    alignment = balign.align_optimal(
        bseq.ProteinSequence(clean), bseq.ProteinSequence(gt_seq), matrix,
        gap_penalty=(gap_open, gap_extend),
    )[0]
    identity = float(balign.get_sequence_identity(alignment))
    L = len(gt_seq)
    target = ["X"] * L
    mask = np.zeros(L, dtype=bool)
    for leaf_i, gt_i in alignment.trace:
        if leaf_i >= 0 and gt_i >= 0:
            target[gt_i] = clean[leaf_i]
            mask[gt_i] = True
    return "".join(target), mask, identity
