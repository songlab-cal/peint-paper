"""lm-design MCMC as an evolutionary simulator: evolve the root down the same trees as PEINT/Bitbol,
using the full lm-design energy (uniform proposal + LM_w/L*CE_i + struct_w*dist_cce_pos + ngram) as
the accept/reject, hamming mode (per branch, walk until Hamming-from-parent = round(bl*L)).

Batched rolling frontier (like paper.esm_mcmc.simulate_family). Each MCMC step over the active
frontier: (1) one masked ESM2 forward -> LM term; (2) thread each candidate onto the GT frame via
the fixed root<->GT map (substitution-only, so the map is computed once/family) and one distogram
forward -> struct term; (3) ngram (cheap). Structure term uses the GT structure, so the structural
evals downstream are partly circular for this model — JSD/divergence are the clean axes.

Outputs figures/output/esm_mcmc/lmdesign/sequences/<fam>.txt (root + leaves), same format as the
Bitbol hamming run, so the existing eval scripts consume it.
"""
import os, sys, time, math
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

import paper_config as cfg
from protevo.io import read_tree
from protevo.utils import read_msa, write_msa
from paper import esmif  # biotite shim
from paper import esm_mcmc
from paper import lmdesign_energy as lm
from paper.vendor.lm_design.loss import get_cce_loss

_STD = "ACDEFGHIKLMNPQRSTVWY"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
LM_W, STRUCT_W, NGRAM_W = 2.0, 3.0, 1.0
NGRAM_ORDERS = (1, 2, 3)
TEMP = 1.0
OUT = Path(cfg.FIGURES_DIR) / "esm_mcmc" / "lmdesign" / "sequences"


def root_gt_map(root_seq, gt_seq):
    """For each root position, the GT column it aligns to (-1 if none). One pairwise alignment."""
    import biotite.sequence as bseq, biotite.sequence.align as bal
    cl = lambda s: "".join(c if c in _STD else "X" for c in s.upper())
    aln = bal.align_optimal(bseq.ProteinSequence(cl(root_seq)), bseq.ProteinSequence(cl(gt_seq)),
                            bal.SubstitutionMatrix.std_protein_matrix(), gap_penalty=(-10, -1))[0]
    m = np.full(len(root_seq), -1, dtype=np.int64)
    for r, g in aln.trace:
        if r >= 0 and g >= 0:
            m[r] = g
    return m


class _Chain:
    __slots__ = ("node", "tokens", "parent", "target", "cap", "cce", "ng", "attempted", "hamming")
    def __init__(self, node, tokens, target, cap, cce, ng):
        self.node, self.tokens, self.parent = node, tokens, tokens.clone()
        self.target, self.cap, self.cce, self.ng = target, cap, cce, ng
        self.attempted, self.hamming = 0, 0
    def done(self):
        return self.hamming >= self.target or self.attempted >= self.cap


def simulate_family(fam, alphabet, proj, struct_dev, neff=1.0, max_batch=96,
                    max_hamming_frac=0.9, attempt_cap_mult=6, seed=0, log=print,
                    lam=None, struct_w=STRUCT_W):
    # lam = length-independent effective LM weight (no /L). None => lm-design's LM_w/L (length-dependent).
    eff_lm = None  # resolved per-family below once L is known
    import esm.inverse_folding as invf
    torch.manual_seed(seed)
    gen = torch.Generator(device=DEV).manual_seed(seed)
    t0 = time.time()

    tree = read_tree(os.path.join(str(cfg.TREE_DIR), f"{fam}.txt"))
    root_label, root_seq = next(iter(read_msa(os.path.join(str(cfg.ROOT_SEQ_DIR), f"{fam}.txt")).items()))
    new_root = esm_mcmc._reroot_at_root_label(tree, root_label)
    L = len(root_seq)
    eff_lm = lam if lam is not None else (LM_W / L)   # length-independent knob, or lm-design's LM_w/L
    max_h = int(round(max_hamming_frac * L))

    coords, gt_seq = invf.util.load_coords(
        os.path.join(str(cfg.GROUND_TRUTH_STRUCTURE_DIR), f"{fam}.pdb"), fam.split("_")[-1])
    labels, contact = lm.target_distogram(coords)                       # [Lgt,Lgt]
    Lgt = labels.shape[0]
    gt_tokens = torch.tensor([alphabet.get_idx(c if c in _STD else "A") for c in gt_seq], device=DEV)
    rmap = torch.tensor(root_gt_map(root_seq, gt_seq), device=DEV)      # [L] -> GT col or -1
    cov = rmap >= 0
    cov_idx = torch.nonzero(cov).squeeze(-1)                            # root positions with a GT col
    cov_gt = rmap[cov_idx]                                             # their GT cols
    K = len(alphabet)

    def struct_cce_batch(seq_tok_batch):                               # [B, L] root-frame token ids
        B = seq_tok_batch.shape[0]
        threaded = gt_tokens.unsqueeze(0).repeat(B, 1)                 # [B, Lgt]
        threaded[:, cov_gt] = seq_tok_batch[:, cov_idx]
        toks = torch.cat([torch.full((B, 1), alphabet.cls_idx, device=DEV), threaded,
                          torch.full((B, 1), alphabet.eos_idx, device=DEV)], dim=1)
        oneh = F.one_hot(toks, K).float()
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
            out = proj(oneh)
        p = F.softmax(out["logits"].permute(0, 2, 3, 1).float(), dim=-1)   # [B, Lgt, Lgt, 18]
        return get_cce_loss(p, labels[None].expand(B, -1, -1), contact[None].expand(B, -1, -1).float())

    def ngram_of(tok_row):
        return lm.ngram_kl(esm_mcmc._decode(tok_row, alphabet), orders=NGRAM_ORDERS)

    root_tok = esm_mcmc._encode(root_seq, alphabet, DEV)
    root_cce = float(struct_cce_batch(root_tok[1:-1].unsqueeze(0))[0])
    root_ng = ngram_of(root_tok)

    seqs = {new_root: root_seq}
    active = []
    stats = {"attempts": 0, "accepts": 0, "branches": 0, "cap_hits": 0}

    def enqueue(parent_node, parent_tok, parent_cce, parent_ng):
        for ch in parent_node.children:
            n = min(int(round(ch.dist * L * neff)), max_h)
            if n <= 0:
                seqs[ch] = seqs[parent_node]
                enqueue(ch, parent_tok, parent_cce, parent_ng)
            else:
                stats["branches"] += 1
                active.append(_Chain(ch, parent_tok.clone(), n, max(n * attempt_cap_mult, 1),
                                     parent_cce, parent_ng))

    enqueue(new_root, root_tok, root_cce, root_ng)

    steps = 0
    while active:
        for s in range(0, len(active), max_batch):
            chunk = active[s:s + max_batch]
            B = len(chunk)
            batch = torch.stack([c.tokens for c in chunk])            # [B, L+2]
            rows = torch.arange(B, device=DEV)
            pos = torch.randint(0, L, (B,), generator=gen, device=DEV) + 1
            cur = batch[rows, pos]
            cur_rank = esm_mcmc._TOK2RANK[cur]
            off = torch.randint(0, len(_STD) - 1, (B,), generator=gen, device=DEV)
            new = esm_mcmc._STD_TOKENS[(cur_rank + 1 + off) % len(_STD)]
            # LM term: masked forward
            masked = batch.clone(); masked[rows, pos] = alphabet.mask_idx
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(DEV == "cuda")):
                lg = esm_mcmc.model_forward_logits(masked)[rows, pos].float()
            dLM = eff_lm * (lg[rows, cur] - lg[rows, new])
            # struct term on the candidate (current with new residue at pos)
            cand = batch.clone(); cand[rows, pos] = new
            cce_cand = struct_cce_batch(cand[:, 1:-1])
            cur_cce = torch.tensor([c.cce for c in chunk], device=DEV)
            dStruct = struct_w * (cce_cand - cur_cce)
            # ngram term (cpu, small)
            dNg = torch.zeros(B, device=DEV)
            cand_ng = [0.0] * B
            for j in range(B):
                cand_ng[j] = ngram_of(cand[j])
                dNg[j] = NGRAM_W * (cand_ng[j] - chunk[j].ng)
            dE = dLM + dStruct + dNg
            acc = (torch.rand(B, generator=gen, device=DEV) < torch.exp(-dE / TEMP).clamp(max=1.0))
            for j, c in enumerate(chunk):
                c.attempted += 1
                stats["attempts"] += 1
                if bool(acc[j]):
                    c.tokens = cand[j]; c.cce = float(cce_cand[j]); c.ng = cand_ng[j]
                    stats["accepts"] += 1
                c.hamming = int((c.tokens[1:-1] != c.parent[1:-1]).sum())
        steps += 1
        finished = [c for c in active if c.done()]
        active = [c for c in active if not c.done()]
        for c in finished:
            seqs[c.node] = esm_mcmc._decode(c.tokens, alphabet)
            if c.hamming < c.target:
                stats["cap_hits"] += 1
            enqueue(c.node, c.tokens, c.cce, c.ng)
        if steps % 20 == 0:
            log(f"  [{fam}] step {steps} frontier {len(active)} nodes {len(seqs)} "
                f"acc {stats['accepts']/max(stats['attempts'],1):.2f} caps {stats['cap_hits']} {time.time()-t0:.0f}s")

    named = {}
    for n, sq in seqs.items():
        nm = "root" if n is new_root else (n.name or f"unnamed-{len(named)}")
        if nm in named:
            nm = f"unnamed-{len(named)}"
        named[nm] = sq
    leaves = new_root.get_leaves()
    leaf_id = np.mean([np.mean([a == b for a, b in zip(seqs[l], root_seq)]) for l in leaves if l in seqs])
    stats.update(family=fam, L=L, seconds=time.time() - t0, n_records=len(named),
                 acceptance=stats["accepts"] / max(stats["attempts"], 1),
                 leaf_identity_to_root=float(leaf_id))
    return named, stats


def main():
    import argparse, pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", nargs="*", default=["1bja_1_A"])
    ap.add_argument("--max-batch", type=int, default=96)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    esm_mcmc.load_model("esm2_t33_650M_UR50D")
    alphabet = esm_mcmc._ALPHABET
    proj, _, sdev = lm.load_struct_model()
    rows = []
    for fam in args.families:
        named, st = simulate_family(fam, alphabet, proj, sdev, max_batch=args.max_batch)
        write_msa(named, str(OUT / f"{fam}.txt"))
        rows.append(st)
        print(f"[{fam}] DONE {st['seconds']:.0f}s | acc {st['acceptance']:.3f} | "
              f"leaf-id-to-root {st['leaf_identity_to_root']:.3f} | caps {st['cap_hits']}/{st['branches']} "
              f"| wrote {st['n_records']} records", flush=True)
    pd.DataFrame(rows).to_csv(Path(cfg.FIGURES_DIR) / "esm_mcmc" / "lmdesign" / "sim_stats.csv", index=False)


if __name__ == "__main__":
    main()
