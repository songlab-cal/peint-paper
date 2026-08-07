"""ESM2-MCMC evolutionary simulator (reviewer response: situating PEINT vs. protein LMs).

A reviewer asked us to place PEINT in the context of extant protein language models.
One way to do that: run the *same* simulation protocol as PEINT — the same phylogenetic
trees, evolving from the same root sequences — but drive the substitutions with **ESM2**
instead of PEINT, via MCMC. Whatever eval we then apply to PEINT's simulated MSAs we can
apply to ESM2's, and compare.

The algorithm is adapted from the two repositories the reviewer pointed at:

* **Bitbol-Lab Phylogeny-ESM2** (``MSAGenerator/MSAGeneratorESM.py``) — evolve a root
  sequence *down a tree*. For a branch of length ``b`` on a length-``L`` sequence, take
  ``n = round(b * L * neff)`` MCMC *attempts* (``neff`` = mutations/site per unit branch
  length; ``neff=1`` makes ``n`` the expected number of substitution events). Each attempt:
  pick a random position, mask it, run ESM2 once, read the softmax at the masked position,
  propose a residue, and **accept via the Metropolis ratio**
  ``p = softmax[proposed] / softmax[current]`` (accept iff ``rand < p`` — which also covers
  ``p >= 1``).
* **ESM lm-design** (``examples/lm-design/lm_design.py``) — the same Metropolis-with-ESM
  idea, framed as MCMC over sequence space. lm-design *anneals* the temperature to optimise
  one sequence; we are *simulating* evolution, so we sample at ``T=1`` (no annealing) to keep
  the stationary distribution equal to ESM2's masked pseudo-likelihood rather than collapsing
  to its mode.

Two deliberate choices, both established with the user:

* **Attempts, not accepted moves.** ``num_mutations`` counts proposal *attempts* per branch
  (Bitbol-faithful). The acceptance rate is then an informative diagnostic — with UNIFORM
  proposals most random substitutions are rejected, so the acceptance rate directly measures
  "how often is a random mutation compatible with the ESM2 context". Realised divergence is
  therefore ``acceptance x (b * L)`` per branch, i.e. *below* the branch length; ``neff`` is
  the knob to rescale it once the acceptance rate is known.
* **Uniform proposals** over the 19 *other* residues (a symmetric proposal, so the Metropolis
  ratio is exactly the target-density ratio and the chain samples ESM2's conditional). This is
  what makes the acceptance rate meaningful; sampling the proposal from ESM2 itself would push
  acceptance toward 1 and hide the diagnostic. Bitbol proposes uniformly over all 20 (self
  included, via ``np.random.randint``); excluding self means every *attempt* is a real
  candidate substitution and the acceptance rate is not inflated by no-op self-proposals.

Substitution-only, fixed length (ESM2 masked-marginal MCMC has no indel move) — exactly like
the classical WAG/LG arms, and unlike PEINT (which can indel). So ESM2-MCMC slots in as another
substitution-only simulator.

GPU strongly recommended (loads ``esm2_t33_650M_UR50D``). The family simulator batches every
currently-active branch of a tree into one masked forward pass per MCMC step (a "rolling
frontier"): a branch becomes active once its parent is finalised, and all active branches share
the family's single sequence length, so the batch is rectangular and needs no padding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch

import esm

# The 20 standard residues; MCMC only ever places these, so the vocab->rank lookup below is
# only populated for them (non-standard tokens map to -1 and never occur mid-run).
_STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

_MODEL = None
_ALPHABET = None
_DEVICE = None
_MODEL_NAME = None
# GPU lookup tables, built once with the model: STD_TOKENS[rank]->token id, TOK2RANK[token]->rank.
_STD_TOKENS = None
_TOK2RANK = None


def load_model(model_name: str = "esm2_t33_650M_UR50D"):
    """Load (once) an ESM2 model in eval mode on GPU if available.

    Returns ``(model, alphabet, device)``. Cached at module scope so the weights load a single
    time per process. Also builds the standard-residue token lookup tables used by the sampler.
    """
    global _MODEL, _ALPHABET, _DEVICE, _MODEL_NAME, _STD_TOKENS, _TOK2RANK
    if _MODEL is not None and _MODEL_NAME != model_name:
        raise RuntimeError(
            f"esm_mcmc.load_model already loaded {_MODEL_NAME!r}; a second model "
            f"{model_name!r} in one process is unsupported."
        )
    if _MODEL is None:
        model, alphabet = getattr(esm.pretrained, model_name)()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _MODEL, _ALPHABET, _DEVICE, _MODEL_NAME = model.eval().to(device), alphabet, device, model_name
        std_tokens = [alphabet.get_idx(a) for a in _STANDARD_AA]
        _STD_TOKENS = torch.tensor(std_tokens, dtype=torch.long, device=device)
        tok2rank = torch.full((len(alphabet),), -1, dtype=torch.long, device=device)
        for rank, tok in enumerate(std_tokens):
            tok2rank[tok] = rank
        _TOK2RANK = tok2rank
    return _MODEL, _ALPHABET, _DEVICE


# ======================================================================================
# tokenisation (manual: all sequences in a family share length L, so no BatchConverter)
# ======================================================================================
def _encode(seq: str, alphabet, device) -> torch.Tensor:
    """``seq`` -> 1-D LongTensor ``[bos] + residues + [eos]`` (length L+2)."""
    ids = [alphabet.cls_idx] + [alphabet.get_idx(c) for c in seq] + [alphabet.eos_idx]
    return torch.tensor(ids, dtype=torch.long, device=device)


def _decode(tokens: torch.Tensor, alphabet) -> str:
    """Inverse of :func:`_encode` — strip bos/eos, map token ids back to residues."""
    return "".join(alphabet.get_tok(int(t)) for t in tokens[1:-1])


# ======================================================================================
# one batched MCMC step (the shared core of _mcmc_branch and simulate_family)
# ======================================================================================
def _mcmc_step(tokens: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """One Metropolis step for every chain in the batch, in place on ``tokens``.

    ``tokens`` is ``[B, L+2]`` (bos + L residues + eos), all rows the same length. For each row
    independently: pick a random residue position, mask it, run ESM2, propose one of the 19
    *other* standard residues uniformly, and accept with probability ``min(1, exp(logit_new -
    logit_old))`` (the softmax normaliser cancels in the ratio, so logits suffice). Returns a
    boolean ``[B]`` accept mask on the CPU; ``tokens`` is mutated in place for accepted rows.

    EXTENSION POINT — composable energy terms (reuse lm-design's). Right now the only energy is
    the ESM2 masked conditional, so ``log_accept = logit_new - logit_old``. To add further
    accept/reject criteria, REUSE lm-design's own energy terms (``examples/lm-design/lm_design.py``,
    GitHub-only — vendor them; not in installed fair-esm) rather than reinventing:
      * ``calc_sequence_loss``  — masked-marginal pseudo-LL (weight ``LM_w/L``); ~what we already do.
      * ``calc_structure_loss`` — distogram CCE of a predicted structure vs. target coords
                                  (``CHOSEN_LOSSES=['dist_cce_pos']``, weight ``struct_w``); the
                                  "structural projection" term — needs a folding trunk + target PDB.
      * ``calc_ngram_loss``     — KL of n-gram frequencies to a reference (weight ``ngram_w``).
    Combine as ``E = LM_w/L*seq + struct_w*struct + ngram_w*ngram`` and set
    ``log_accept = E(old) - E(new)``. Track the acceptance rate PER active term-set: each extra
    term lowers acceptance, so the acceptance rate is an empirical price tag for that term (and
    these same terms could be added to PEINT itself).
    """
    model, alphabet, device = load_model(_MODEL_NAME or "esm2_t33_650M_UR50D")
    B, Lp2 = tokens.shape
    L = Lp2 - 2
    rows = torch.arange(B, device=device)

    # position in [0, L): token index is +1 for the prepended bos.
    pos = torch.randint(0, L, (B,), generator=gen, device=device) + 1
    cur_tok = tokens[rows, pos]
    cur_rank = _TOK2RANK[cur_tok]                                  # 0..19
    offset = torch.randint(0, len(_STANDARD_AA) - 1, (B,), generator=gen, device=device)
    new_rank = (cur_rank + 1 + offset) % len(_STANDARD_AA)         # guaranteed != cur_rank
    new_tok = _STD_TOKENS[new_rank]

    masked = tokens.clone()
    masked[rows, pos] = alphabet.mask_idx
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
        logits = model(masked)["logits"]                          # [B, L+2, vocab]
    logits_at_pos = logits[rows, pos].float()                     # [B, vocab]
    ratio = torch.exp(logits_at_pos[rows, new_tok] - logits_at_pos[rows, cur_tok])
    u = torch.rand(B, generator=gen, device=device)
    accept = u < ratio
    tokens[rows, pos] = torch.where(accept, new_tok, cur_tok)
    return accept.detach().cpu()


def _mcmc_branch(
    seq: str,
    num_mutations: int,
    model_name: str = "esm2_t33_650M_UR50D",
    seed: int = 0,
) -> Tuple[str, int, int]:
    """Evolve one sequence by ``num_mutations`` MCMC *attempts* (single-chain reference path).

    This is the ``_mcmc_branch(seq, num_mutations)`` entry point: take exactly ``num_mutations``
    Metropolis attempts on ``seq`` and return ``(evolved_seq, n_attempted, n_accepted)``. The
    family simulator uses the same :func:`_mcmc_step` core batched across a whole tree; this
    one-chain version exists for testing and for callers that want to evolve a single branch.
    """
    _, alphabet, device = load_model(model_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    tokens = _encode(seq, alphabet, device).unsqueeze(0)          # [1, L+2]
    accepted = 0
    for _ in range(num_mutations):
        accepted += int(_mcmc_step(tokens, gen).sum())
    return _decode(tokens[0], alphabet), num_mutations, accepted


# ======================================================================================
# family simulation: evolve the whole tree with a batched rolling frontier
# ======================================================================================
@dataclass
class _Chain:
    """One active branch being evolved. ``target`` is interpreted per the family's target mode:

    * ``"attempts"`` — run exactly ``target`` MCMC proposal attempts (Bitbol-faithful).
    * ``"hamming"``  — run until the sequence differs from ``parent`` at ``target`` sites, or the
      attempt budget ``cap`` is exhausted (``parent`` is the branch's starting tokens; ``hamming``
      is the current site-difference count from it).
    """
    node: object                    # ete3 TreeNode (the child endpoint of the branch)
    tokens: torch.Tensor            # [L+2], evolves in place
    target: int                     # attempts target OR Hamming target (by mode)
    cap: int                        # max attempts (== target in attempts mode)
    parent: Optional[torch.Tensor] = None   # branch-start tokens (hamming mode only)
    attempted: int = 0
    accepted: int = 0
    hamming: int = 0

    def done(self) -> bool:
        if self.parent is None:                 # attempts mode
            return self.attempted >= self.target
        return self.hamming >= self.target or self.attempted >= self.cap  # hamming mode


@dataclass
class FamilyStats:
    family: str
    length: int
    n_nodes: int = 0
    n_branches: int = 0             # branches with >=1 attempt (num_mutations>0)
    total_attempts: int = 0
    total_accepted: int = 0
    n_leaves: int = 0
    n_cap_hits: int = 0             # hamming mode: branches that exhausted the attempt budget
    mean_leaf_identity_to_root: float = float("nan")   # realised: fraction of sites == root
    mean_root_to_leaf_distance: float = float("nan")   # expected: subs/site along the tree
    seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.total_accepted / self.total_attempts if self.total_attempts else float("nan")


def _reroot_at_root_label(cherry_tree, root_label: str):
    """Reroot the tree at the root-sequence leaf, exactly as the PEINT simulator does.

    ``cherryml.io.Tree.to_ete3()`` preserves names and branch lengths; ``set_outgroup`` on the
    root-sequence leaf places the root sequence at the (new) internal root and makes every other
    leaf a descendant to be evolved. Returns the new ete3 root node.
    """
    ete = cherry_tree.to_ete3()
    ete.set_outgroup(ete & root_label)
    return ete.get_tree_root()


def simulate_family(
    tree,                            # cherryml.io.Tree
    root_label: str,
    root_seq: str,
    family: str,
    model_name: str = "esm2_t33_650M_UR50D",
    target_mode: str = "attempts",   # "attempts" (Bitbol-faithful) or "hamming" (realise divergence)
    neff: float = 1.0,               # attempts mode: attempts = round(bl * L * neff)
    max_hamming_frac: float = 0.9,   # hamming mode: cap the per-branch Hamming target at this*L
    attempt_cap_mult: int = 40,      # hamming mode: attempt budget = attempt_cap_mult * target
    max_batch: int = 256,
    seed: int = 0,
    progress_cb: Optional[Callable[[dict], None]] = None,
    heartbeat_every: int = 2000,
) -> Tuple[Dict[str, str], FamilyStats]:
    """Evolve ``root_seq`` down ``tree`` with ESM2-MCMC; return (node_name->seq, stats).

    Per-branch work depends on ``target_mode``:

    * ``"attempts"`` — branch ``child`` gets ``round(child.dist * L * neff)`` proposal *attempts*
      (Bitbol-faithful). Deterministic cost; realised divergence is ``acceptance x`` that.
    * ``"hamming"``  — branch ``child`` runs until its sequence differs from the parent at
      ``round(child.dist * L)`` sites (branch length read directly as the fraction of mutated
      sites: ``0.1 -> 10%``), so realised per-branch divergence matches the branch length. The
      target is capped at ``max_hamming_frac * L`` and the attempt budget at
      ``attempt_cap_mult * target``; branches that exhaust the budget without reaching target are
      counted in ``stats.n_cap_hits`` (logged, never silent).

    All branches whose parent is finalised are evolved concurrently (batched into one masked
    forward per step, sub-batched at ``max_batch``). ``progress_cb`` receives a dict roughly every
    ``heartbeat_every`` steps and at each branch finalisation, for live logging.
    """
    if target_mode not in ("attempts", "hamming"):
        raise ValueError(f"target_mode must be 'attempts' or 'hamming', got {target_mode!r}")
    _, alphabet, device = load_model(model_name)
    gen = torch.Generator(device=device).manual_seed(seed)
    t0 = time.time()

    new_root = _reroot_at_root_label(tree, root_label)
    L = len(root_seq)
    max_hamming = int(round(max_hamming_frac * L))
    stats = FamilyStats(family=family, length=L)
    stats.n_nodes = sum(1 for _ in new_root.traverse())

    # Realised-vs-expected divergence: expected = mean root->leaf distance (subs/site).
    leaves = new_root.get_leaves()
    stats.n_leaves = len(leaves)
    stats.mean_root_to_leaf_distance = float(
        np.mean([new_root.get_distance(leaf) for leaf in leaves])
    ) if leaves else float("nan")

    seqs: Dict[object, str] = {new_root: root_seq}      # node -> finalised sequence
    active: List[_Chain] = []

    def enqueue_children(parent) -> None:
        """Schedule each child branch; zero-target branches copy the parent seq and recurse."""
        for child in parent.children:
            toks = _encode(seqs[parent], alphabet, device)
            if target_mode == "attempts":
                n = int(round(child.dist * L * neff))
                chain = _Chain(child, toks, target=n, cap=n) if n > 0 else None
            else:  # hamming
                n = min(int(round(child.dist * L)), max_hamming)
                chain = (_Chain(child, toks, target=n, cap=max(n * attempt_cap_mult, 1),
                                parent=toks.clone()) if n > 0 else None)
            if chain is None:
                seqs[child] = seqs[parent]
                enqueue_children(child)
            else:
                stats.n_branches += 1
                active.append(chain)

    enqueue_children(new_root)

    steps = 0
    since_beat = 0
    while active:
        # Step every active chain once, sub-batched to <= max_batch rows per forward.
        for start in range(0, len(active), max_batch):
            chunk = active[start:start + max_batch]
            batch = torch.stack([c.tokens for c in chunk])          # [b, L+2]
            acc = _mcmc_step(batch, gen)                            # per-row accept mask (bool)
            # Hamming from each chain's branch-start parent (hamming mode), vectorised per chunk.
            if target_mode == "hamming":
                parents = torch.stack([c.parent for c in chunk])
                ham = (batch[:, 1:-1] != parents[:, 1:-1]).sum(1).cpu()
            for j, (c, row_tokens, a) in enumerate(zip(chunk, batch, acc)):
                c.tokens = row_tokens
                c.attempted += 1
                c.accepted += int(a)
                if target_mode == "hamming":
                    c.hamming = int(ham[j])
                stats.total_attempts += 1
                stats.total_accepted += int(a)
        steps += 1
        since_beat += 1

        finished = [c for c in active if c.done()]
        active = [c for c in active if not c.done()]
        for c in finished:
            seqs[c.node] = _decode(c.tokens, alphabet)
            if c.parent is not None and c.hamming < c.target:
                stats.n_cap_hits += 1
            enqueue_children(c.node)

        if progress_cb is not None and (since_beat >= heartbeat_every or (finished and not active)):
            since_beat = 0
            progress_cb({
                "family": family, "step": steps, "frontier": len(active),
                "nodes_done": len(seqs), "n_nodes": stats.n_nodes,
                "attempts": stats.total_attempts, "accepted": stats.total_accepted,
                "acceptance": stats.acceptance_rate, "cap_hits": stats.n_cap_hits,
                "elapsed": time.time() - t0,
            })

    stats.seconds = time.time() - t0
    # Name every record uniquely. The ancestral root sequence is "root" (matching the PEINT
    # simulator's output convention); leaves keep their tree names (seqN) so downstream leaf
    # selection works; any unnamed internal nodes introduced by rerooting get a unique fallback.
    named: Dict[str, str] = {}
    for n, s in seqs.items():
        name = "root" if n is new_root else n.name
        if not name or name in named:
            name = f"unnamed-{len(named)}"
        named[name] = s
    # Realised divergence at the leaves: mean fraction of sites differing from the root seq.
    leaf_ids = [
        np.mean([a != b for a, b in zip(seqs[leaf], root_seq)])
        for leaf in leaves if leaf in seqs
    ]
    stats.mean_leaf_identity_to_root = float(1.0 - np.mean(leaf_ids)) if leaf_ids else float("nan")
    return named, stats


# ======================================================================================
# scoring utilities (for the energy-cost / acceptance-degradation analysis)
# ======================================================================================
def masked_conditional_logits(
    seq: str, positions=None, model_name: str = "esm2_t33_650M_UR50D", max_batch: int = 128
) -> np.ndarray:
    """ESM2 masked-conditional logits over the 20 standard residues at each queried position.

    For each position ``p``, masks that single position and reads ESM2's logits there — the same
    conditional that drives the MCMC accept/reject. Returns an ``np.ndarray`` of shape
    ``[len(positions), 20]`` with columns in ``_STANDARD_AA`` order (raw logits; softmax/ratios are
    the caller's to take — the acceptance ratio ``p(a)/p(cur)`` is normaliser-independent).
    """
    _, alphabet, device = load_model(model_name)
    positions = list(range(len(seq))) if positions is None else list(positions)
    base = _encode(seq, alphabet, device)                         # [L+2]
    out = np.zeros((len(positions), len(_STANDARD_AA)), dtype=np.float32)
    for start in range(0, len(positions), max_batch):
        chunk = positions[start:start + max_batch]
        toks = base.unsqueeze(0).repeat(len(chunk), 1).clone()
        for r, p in enumerate(chunk):
            toks[r, p + 1] = alphabet.mask_idx                    # +1 for the prepended bos
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            logits = model_forward_logits(toks)
        for r, p in enumerate(chunk):
            out[start + r] = logits[r, p + 1, _STD_TOKENS].float().cpu().numpy()
    return out


def model_forward_logits(tokens: torch.Tensor) -> torch.Tensor:
    """Thin wrapper: ESM2 logits for a token batch (kept separate so callers can reuse it)."""
    model, _, _ = load_model(_MODEL_NAME or "esm2_t33_650M_UR50D")
    return model(tokens)["logits"]
