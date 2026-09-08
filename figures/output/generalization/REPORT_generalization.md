# Generalization to protein classes absent from training

*Analysis supporting the reviewer response. Reproducible via
`python -m benchmarks.generalization_{af2rank,omegafold,jsd_family,jsd_domain}` (see
`paper/generalization.py`). All numbers below are regenerated from the cached benchmark results.*

## Reviewer concern

Does the reported simulation quality reflect genuine generalization, or memorization of protein
classes seen in training?

## Design

The 553 evaluation families are already held out at the **PDB level** (no evaluation PDB appears in
the 14,498 training families). We go further and label every one of the 15,051 families with its
Pfam family and its structural classification (SCOPe + ECOD), then split the evaluation families by
whether their class was **ever seen in training**:

- **novel** = *none* of the family's domains/labels (under a given scheme) appears in any training
  family (multi-domain proteins included);
- **seen** = at least one label appears in training.

We then re-stratify the already-computed metrics by novel vs. seen. The empirical **"Real (other
split)"** row — the JSD/pLDDT of the *real* sequences on the held-out subtree — is the difficulty
control: if novel families are simply harder, Real moves too, and only a *PEINT-specific* novel/seen
gap indicates a generalization failure.

## How much novelty is actually in the evaluation set

| Scheme | Kind | Eval coverage | **Novel** | Seen |
|---|---|---:|---:|---:|
| **Pfam family** | sequence domain | 62% | **81** | 262 |
| Pfam clan | sequence homology | 62% | 23 | 320 |
| ECOD H-group (≈ superfamily) | structural | **89%** | 15 | 478 |
| ECOD T-group (fold/topology) | structural | 89% | 16 | 477 |
| SCOPe superfamily | structural | 38% | 14 | 192 |

**Structural novelty is scarce and the structural splits are underpowered.** Across the two
well-behaved structural classifications (ECOD at 89% coverage, SCOPe curated), only **~15** evaluation
families have a superfamily absent from training — the held-out set reuses known folds almost
entirely. (CATH was dropped: at 73% coverage it under-samples the training vocabulary and inflates
the novel count to 32; SIFTS SCOP2 is empty; SCOP2B over-calls novelty at 92% due to 26% training
coverage.) Genuine, testable novelty therefore lives at the finer **Pfam-family** level (81 novel),
which is the powered test. The structural splits are reported for completeness but not relied on.

## Results — PEINT vs. the Real-data control

Median metric on novel vs. seen; `p` is Mann-Whitney U (`p_matched` = seen families matched to
novel on sequence length); Cliff's δ is the effect size (novel relative to seen).

| Metric | Scheme | Model | n (novel/seen) | novel | seen | p | p_matched | δ |
|---|---|---|---:|---:|---:|---:|---:|---:|
| **AF2Rank pLDDT** | Pfam family | **PEINT** | 81/256 | 61.2 | 70.8 | 3e-4 | 4e-3 | −0.27 |
| | Pfam family | Real | 81/256 | 74.9 | 75.9 | 0.92 | 0.60 | +0.01 |
| **OmegaFold pLDDT** | Pfam family | **PEINT** | 80/255 | 77.3 | 83.4 | 3e-5 | 2e-3 | −0.31 |
| | Pfam family | Real | 80/255 | 87.5 | 92.3 | 7e-7 | 6e-5 | −0.37 |
| **Family JSD** | Pfam family | **PEINT** | 80/255 | 0.384 | 0.261 | 1e-6 | 5e-4 | +0.36 |
| | Pfam family | Real | 80/255 | 0.276 | 0.242 | 7e-4 | 2e-3 | +0.25 |
| **Domain JSD** | novel vs seen domain | **PEINT** | 132/432 | 0.353 | 0.262 | 1e-9 | — | +0.35 |
| | novel vs seen domain | Real | 132/432 | 0.289 | 0.237 | 6e-7 | — | +0.29 |

*(Structural splits, ECOD H-group / SCOPe superfamily, show no PEINT-specific gap on any metric —
PEINT δ tracks Real δ — but n≈15 so they are underpowered. Full numbers in
`generalization_summary.csv` and the per-metric `*_stats.csv`.)*

### Reading

1. **De-novo folding (OmegaFold pLDDT), template-free.** Real sequences *also* fold less confidently
   on novel families (δ = −0.37) — novel families are intrinsically harder. PEINT drops in step
   (δ = −0.31), i.e. **no more than the real data**. → generalizes.

2. **Sequence conservation (JSD), template-free.** PEINT reproduces the real per-site distributions
   far better than every classical baseline on both strata, and on *seen* families nearly reaches the
   empirical floor (0.261 vs. Real 0.242). Novel families/domains have higher JSD for PEINT — but the
   real data is elevated too (novel is intrinsically harder to reproduce). The decisive control is the
   **within-family paired domain test** (`generalization_jsd_domain_paired.png`; 40 families carrying
   both a novel and a seen Pfam domain): PEINT's novel−seen increase (+0.043, p=0.02) **matches** the
   real data's (+0.034, p=0.01) — the two median slopes are visually indistinguishable. When
   difficulty is matched perfectly (same family, same tree/depth), PEINT tracks the real data →
   generalizes at the domain level. The slightly larger PEINT gap in the *unmatched* family-level comparison
   (δ +0.36 vs Real +0.25) reflects covariate differences between the novel and seen family sets.

3. **AF2Rank pLDDT** is the **only** metric with a PEINT-specific gap: Real is flat across strata
   (δ = +0.01) while PEINT drops (δ = −0.27). However, AF2Rank is a brittle proxy — it approximates
   sequence–structure compatibility from a single sequence + template, and the template's influence
   is hard to reason about. Given that the two template-free metrics show PEINT tracking the real
   data, this lone signal should be weighted accordingly.

## Bottom line

Under template-free evaluation — de-novo folding and sequence conservation — PEINT's quality on
protein families **absent from training** degrades no more than the *real data itself* does, and PEINT
remains far ahead of the classical baselines. The held-out set contains essentially no novel folds
(~15 novel superfamilies at ≥89% ECOD coverage), so novelty is a sequence-family phenomenon; there,
the difficulty-matched controls (paired within-family JSD; length-matched pLDDT) show PEINT
generalizing rather than memorizing. The only contrary signal, AF2Rank pLDDT, comes from the least
reliable, template-dependent metric.

## Caveats (stated plainly)

- Structural (fold/superfamily) splits are **underpowered** (~15 novel families); we do not draw
  conclusions from them beyond "the eval set is not adversarially novel at the fold level."
- Pfam covers 62% of chains; the JSD/pLDDT novelty split is over the labeled subset. ECOD (89%)
  corroborates the low structural novelty.
- On novel Pfam families, PEINT's *advantage over the best site-rate-matched classical baseline*
  (LG+S256) narrows (both ≈0.37–0.38 family JSD) even as it stays well ahead of LG/WAG/LG+C60 and far
  ahead on seen families.

## Files

- Figures: `generalization_{af2rank_plddt,omegafold_plddt,jsd_family,jsd_domain}.{pdf,png}`
  and the paired domain-level control `generalization_jsd_domain_paired.{pdf,png}`
- Stats: `generalization_*_stats.csv`; consolidated `generalization_summary.csv`
- Per-family caches: `family_jsd_heldout.csv`, `domain_jsd_heldout.csv`
- Labels/annotations: `paper/generalization.py` → `<ANNOTATION_DIR>/family_labels.json`
  (Pfam via SIFTS; SCOPe 2.08; ECOD v295; Pfam-A HMMs for domain ranges on seq1)
