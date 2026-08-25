#!/usr/bin/env python
"""Regenerate the figure panels that appear in the paper — and only those.

`figures/output/` accumulates exploratory output; this is the whitelist. Each entry
records how a panel is produced, which conda env it needs, and what it writes, so
`--list` doubles as the documentation of the released figure set.

    scripts/render_panels.py --list
    scripts/render_panels.py --dry-run                 # print the exact commands
    scripts/render_panels.py --only figure3_jsd        # one panel (or a group prefix)
    scripts/render_panels.py --group main              # main-text panels
    scripts/render_panels.py --check                   # just report what is missing

Two conda envs are involved and they cannot share a process, so every panel is run as a
subprocess with an explicit interpreter. Point PEINT_PAPER_PY_ESMC / PEINT_PAPER_PY_PROTEVO
at them; they default to the current interpreter, which is right only if it happens to
satisfy the panel's requirements.
"""

import argparse
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Interpreters. "esmc" covers everything except the panels that need cherryml + ete3 +
# AliSim/MAFFT (the PCP panels) or the Historian event counter.
PY = {
    "esmc": os.environ.get("PEINT_PAPER_PY_ESMC", sys.executable),
    "protevo": os.environ.get("PEINT_PAPER_PY_PROTEVO", sys.executable),
}


@dataclass
class Panel:
    name: str
    group: str                     # "main" | "extended"
    module: str
    argv: list = field(default_factory=list)
    env: str = "esmc"
    outputs: list = field(default_factory=list)   # relative to figures/
    depends_on: list = field(default_factory=list)
    cost: str = "seconds"
    note: str = ""
    # argv that redraws this panel from a saved table instead of recomputing it. Only the
    # panels where recomputing is genuinely expensive have one.
    from_csv_argv: list = field(default_factory=list)


PANELS = [
    # ---------------- main text ----------------
    Panel("figure2_likelihood_eval", "main", "figures.figure2_ll_eval_esmc",
          outputs=["output/figure2_likelihood_eval_test_esmc.pdf",
                   "output/figure2_likelihood_eval_train_held_out_esmc.pdf"],
          cost="GPU; ~15 min",
          note="Needs a GPU and HF_HOME. Reads the peint repo's local_data + _cache_peint."),
    # Two entries for one script, because no single env can do both halves: ESM-C generation
    # needs peint-esmc's transformers, OmegaFold lives in protevo-env. The sequence sims are
    # cached, so the folding pass re-reads them instead of rebuilding any model.
    Panel("figure2_simulation_generate", "main", "figures.figure2_simulation",
          argv=["--skip-structures"], env="esmc",
          outputs=["output/figure2_simulation_mutations.pdf"],
          cost="GPU; ~2 min",
          note="Star-topology simulation from one sequence: both PEINT backbones + WAG + LG. "
               "Needs a GPU and Historian. Populates the sequence cache the folding pass reads."),
    Panel("figure2_simulation_plddt", "main", "figures.figure2_simulation",
          env="protevo",
          outputs=["output/figure2_simulation_plddt.pdf"],
          depends_on=["figure2_simulation_generate"],
          cost="GPU; ~1.5 h (folds ~1200 structures)",
          note="OmegaFold pLDDT vs time. Must run in protevo-env (the only env with omegafold); "
               "the sequence sims are cache hits here, so no ESM-C model is constructed."),
    Panel("figure3_af2rank_ecdf", "main", "figures.figure3_structure_metrics",
          argv=["--panels", "af2"],
          outputs=["output/figure3_af2rank_plddt_ecdf.pdf"], cost="~2 min"),
    Panel("figure3_omegafold_ecdf", "main", "figures.figure3_structure_metrics",
          argv=["--panels", "omegafold"],
          outputs=["output/figure3_omegafold_plddt_ecdf.pdf"], cost="~10 min",
          note="Walks ~16k OmegaFold PDBs. Its CSV is the input to generalization_omegafold."),
    Panel("figure3_jsd_boxplot", "main", "figures.figure3_conservation",
          argv=["--panels", "boxplot"],
          outputs=["output/figure3_conservation_jsd_boxplot.pdf"], cost="~30-60 min"),
    Panel("figure3_conservation_logo", "main", "figures.figure3_conservation",
          argv=["--panels", "logo"],
          outputs=["output/figure3_conservation_lg_s256.pdf"], cost="seconds",
          note="Needs logomaker."),

    # ---------------- extended data ----------------
    Panel("pcp_panels", "extended", "figures.figure3_pcp_mutation_counts", env="protevo",
          outputs=["output/parent_child_pairs_all_models.pdf",
                   "output/back_mutation_and_root_leaf.pdf",
                   "output/per_family_median_mutation_rate_all_models.pdf"],
          cost="~30-45 min warm, or ~20 s with --from-csv",
          from_csv_argv=["--replot"],
          note="Needs iqtree2 + MAFFT on PATH. Warm only if PEINT_PAPER_PROTEVO_CACHE_DIR / "
               "PEINT_PAPER_CHERRYML_CACHE_DIR point at prebuilt caches; cold it re-runs AliSim. "
               "--from-csv redraws from the saved aggregation tables instead."),
    Panel("historian_indel_esmc_vs_rev1", "extended", "benchmarks.historian_compare_esmc_vs_rev1",
          env="protevo",
          outputs=["output/historian_indel_esmc_vs_rev1_refine.pdf",
                   "output/historian_indel_length_cdf_refine.pdf"],
          cost="~5 min"),
    Panel("historian_indel_vs_length", "extended", "benchmarks.historian_indel_vs_length",
          env="protevo",
          outputs=["output/historian_indel_vs_length.pdf"], cost="~5 min"),
    Panel("threedi_jsd_boxplot", "extended", "benchmarks.threedi_jsd_all_models",
          argv=["--skip-3di-generation"], env="protevo",
          outputs=["output/esmc_summary/conservation_jsd_3di_boxplot.pdf"],
          cost="~30-60 min",
          note="Without --skip-3di-generation this runs ProstT5 on a GPU."),
    Panel("generalization_jsd_domain", "extended", "benchmarks.generalization_jsd_domain",
          outputs=["output/generalization/generalization_jsd_domain.pdf",
                   "output/generalization/generalization_jsd_domain_paired.pdf"],
          cost="seconds",
          note="Cached in figures/output/generalization/domain_jsd_heldout.csv."),
    Panel("generalization_omegafold", "extended", "benchmarks.generalization_omegafold",
          outputs=["output/generalization/generalization_omegafold_plddt.pdf"],
          depends_on=["figure3_omegafold_ecdf"], cost="seconds"),
    Panel("blast_similarity", "extended", "benchmarks.blast_similarity",
          outputs=["output/blast_similarity.pdf",
                   "output/blast_similarity_with_nohit.pdf"],
          cost="~40 s",
          note="Best-hit %identity of simulated leaves vs BLAST nr, family medians. Reads the "
               "two blast_sequences dirs (rev1 + rev2 ESM-C); no model or GPU. The main panel "
               "is hits-only (the original definition) and each row is annotated with the "
               "families and sequence hit-rate behind it, because most classical-simulator "
               "sequences have no hit at all; blast_similarity_coverage.csv has the raw counts."),
    Panel("esmif_validation", "extended", "benchmarks.esmif_validation",
          argv=["--approach", "both"], env="protevo",
          outputs=["output/esmif/esmif_approach1_gt_likelihood.pdf",
                   "output/esmif/esmif_approach1_divergence_controlled.pdf",
                   "output/esmif/esmif_approach2_selfconsistency.pdf"],
          cost="GPU; minutes if the CSVs are complete, ~2 h cold",
          note="Inverse-folding validation. Needs protevo-env (torch_geometric + fair-esm). "
               "Reuses esmif_gt_likelihood{,_perleaf}.csv and esmif_selfconsistency.csv, "
               "recomputing only (family, model) pairs missing from BOTH tables."),
    Panel("esm_mcmc_spectrum", "extended", "figures.figure_esm_mcmc_spectrum",
          outputs=["output/figure_esm_mcmc_spectrum.pdf"], cost="seconds",
          note="Reads 5 CSVs under figures/output/esm_mcmc/."),
    Panel("vep_params", "extended", "figures.figure5_vep", argv=["--plot", "params"],
          outputs=["params/params_vs_spearman.pdf",
                   "params/params_vs_spearman_OrganismalFitness.pdf"], cost="seconds"),
    Panel("vep_by_base_lm", "extended", "figures.figure5_vep", argv=["--plot", "by_base_lm"],
          outputs=["spearman_agg/multimodel_esm2_150_650_esmc.pdf",
                   "spearman_agg/multimodel_by_base_lm.pdf",
                   "mutational_depth/mutational_depth_by_base_lm.pdf"], cost="seconds"),
]

BY_NAME = {p.name: p for p in PANELS}


def _select(args):
    sel = PANELS
    if args.group:
        sel = [p for p in sel if p.group == args.group]
    if args.only:
        wanted = []
        for token in args.only:
            hits = [p for p in PANELS if p.name == token or p.name.startswith(token)]
            if not hits:
                sys.exit(f"no panel matches {token!r}; try --list")
            wanted.extend(hits)
        sel = [p for p in sel if p in wanted] or wanted
    # Pull in prerequisites, keeping declaration order.
    needed = {p.name for p in sel}
    for p in list(sel):
        needed.update(p.depends_on)
    return [p for p in PANELS if p.name in needed]


def cmd_list():
    print(f"{len(PANELS)} panel groups -> "
          f"{sum(len(p.outputs) for p in PANELS)} output files\n")
    for group in ("main", "extended"):
        print(f"--- {group} ---")
        for p in (x for x in PANELS if x.group == group):
            dep = f"  [after {', '.join(p.depends_on)}]" if p.depends_on else ""
            print(f"  {p.name:32s} env={p.env:8s} {p.cost}{dep}")
            for o in p.outputs:
                print(f"      figures/{o}")
            if p.note:
                print(f"      note: {p.note}")
        print()


def cmd_check(sel):
    missing = 0
    for p in sel:
        for o in p.outputs:
            path = REPO_ROOT / "figures" / o
            if not path.exists():
                print(f"  MISSING  figures/{o}   (run --only {p.name})")
                missing += 1
    print("all selected outputs present" if not missing else f"{missing} missing")

    # An output can be present and stale, or absent because its inputs are. Report the data
    # side too, scoped to the panels actually selected, so --check answers both halves of
    # "can I regenerate this?" -- the inventory lives in data/MANIFEST.toml.
    try:
        from paper import manifest
    except Exception as exc:                            # tomli missing, manifest malformed
        print(f"(data check unavailable: {exc})")
        return 1 if missing else 0

    needed, seen = [], set()
    for panel in sel:
        for r in manifest.roles(panel=panel.name):
            if r.name not in seen:
                seen.add(r.name)
                needed.append(r)
    _, absent = manifest.check(needed)
    absent = [r for r in absent if r.get("tier") != "on_request"]
    if absent:
        print(f"{len(absent)} data role(s) missing for the selected panels: "
              f"{', '.join(r.name for r in absent)}")
        hint = ("scripts/check_local_data.py --panels " + " ".join(p.name for p in sel)
                if len(sel) <= 4 else "scripts/check_local_data.py")
        print("  " + hint)
    else:
        print(f"all {len(needed)} data role(s) for the selected panels present")
    return 1 if missing else 0


def run(panel, dry, from_csv=False):
    interp = PY[panel.env]
    if from_csv and panel.from_csv_argv:
        argv = [interp, "-m", panel.module, *panel.from_csv_argv]
    elif from_csv and not panel.from_csv_argv:
        # No replot path: the panel is already cheap, or reads a saved table by default.
        argv = [interp, "-m", panel.module, *panel.argv]
    else:
        argv = [interp, "-m", panel.module, *panel.argv]
    print(f"\n=== {panel.name}  (env={panel.env}, {panel.cost})")
    print("    " + " ".join(argv))
    if dry:
        return 0
    if interp == sys.executable and not os.environ.get(
            "PEINT_PAPER_PY_ESMC" if panel.env == "esmc" else "PEINT_PAPER_PY_PROTEVO"):
        print(f"    (using the current interpreter; set "
              f"PEINT_PAPER_PY_{panel.env.upper()} to pin the right env)")
    r = subprocess.run(argv, cwd=REPO_ROOT)
    if r.returncode:
        print(f"    FAILED (exit {r.returncode})")
        return r.returncode
    for o in panel.outputs:
        path = REPO_ROOT / "figures" / o
        print(f"    {'wrote' if path.exists() else 'MISSING'}  figures/{o}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="Show the whitelist and exit.")
    ap.add_argument("--check", action="store_true", help="Report missing outputs and exit.")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    ap.add_argument("--only", nargs="+", metavar="NAME", help="Panel name or name prefix.")
    ap.add_argument("--group", choices=("main", "extended"))
    ap.add_argument("--from-csv", action="store_true",
                    help="Prefer each panel's replot-from-saved-table path where it has one.")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue after a panel fails instead of stopping.")
    args = ap.parse_args()

    if args.list:
        cmd_list()
        return 0
    sel = _select(args)
    if args.check:
        return cmd_check(sel)

    if not args.dry_run:
        for tool, why in (("mafft", "pcp_panels"), ("iqtree2", "pcp_panels")):
            if any(p.name == why for p in sel) and shutil.which(tool) is None:
                print(f"warning: {tool} not on PATH; {why} needs it for a cold run")

    failures = []
    for p in sel:
        if run(p, args.dry_run, args.from_csv) and not args.dry_run:
            failures.append(p.name)
            if not args.keep_going:
                break
    if failures:
        print(f"\nfailed: {', '.join(failures)}")
        return 1
    print(f"\n{len(sel)} panel group(s) {'listed' if args.dry_run else 'rendered'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
