"""ESM-IF Approach 1 (GT-structure conditioned log-likelihood) for PEINT + Real only,
one revision at a time.

Reuses benchmarks.esmif_validation.run_approach1 unchanged (thread each non-root leaf onto the
experimental PDB via seq1's non-gap reference columns, score p(leaf | GT structure) under
esm_if1). The seq1 reference frame makes the two revisions' (different) mafft-add alignments
comparable, so Real should score ~identically across revisions.

Run once per revision by setting PEINT_PAPER_RESULTS_DIR before invoking:

    PEINT_PAPER_RESULTS_DIR=<rev1> python -m benchmarks.esmif_approach1_peint --tag esm2_rev1
    PEINT_PAPER_RESULTS_DIR=<rev2> python -m benchmarks.esmif_approach1_peint --tag esmc_rev2
"""
import argparse
import os

import paper_config as cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="output tag, e.g. esm2_rev1 / esmc_rev2")
    ap.add_argument("--limit", type=int, default=None, help="first N families (smoke test)")
    args = ap.parse_args()

    import benchmarks.esmif_validation as ev
    ev.MODEL_ORDER = ["PEINT", "Real"]  # restrict to PEINT + Real (functions read this global)

    fams = ev.eval_families()
    if args.limit:
        fams = fams[: args.limit]
    print(f"[{args.tag}] RESULTS_DIR={os.environ.get('PEINT_PAPER_RESULTS_DIR')}  "
          f"{len(fams)} families  models={ev.MODEL_ORDER}")

    a1, a1_leaf = ev.run_approach1(fams)
    out = str(cfg.FIGURES_DIR)
    os.makedirs(out, exist_ok=True)
    a1.to_csv(f"{out}/esmif_a1_{args.tag}.csv", index=False)
    a1_leaf.to_csv(f"{out}/esmif_a1_{args.tag}_perleaf.csv", index=False)
    print(f"\n[{args.tag}] median LL per model:")
    print(a1.groupby("model")["ll"].agg(["count", "median", "mean"]))
    print(f"saved: {out}/esmif_a1_{args.tag}.csv")


if __name__ == "__main__":
    main()
