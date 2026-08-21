"""Data helpers for the base-LM VEP comparison figures (figure5_vep).

Kept out of figures/figure5_vep.py (already large) so the figure module stays
thin plotting + orchestration. Everything here is pure data/config — no
matplotlib — and takes the results dir as an argument rather than importing it,
so it has no dependency back on the figure module.

The comparison covers three frozen backbones, each with its released zero-shot
baseline and the PEINT model trained on it, in the paired display order
    ESM2-150 | PEINT ESM2-150 | ESM-C | PEINT ESM-C | ESM2-650 | PEINT ESM2-650.
Per-family Spearman comes from each run's ``spearman_results.csv``; base-pLM
mutational-depth curves are derived from the ProteinGym release on demand
(they ship only for the PEINT runs).
"""

import pandas as pd

# (base_lm_label, released_baseline_column, peint_run_dir)
BASE_LM_CONFIG = [
    ("ESM2-150M", "ESM2_150M", "peint_esm2_150m"),
    ("ESM-C 300M", "ESMC-300M", "peint_esmc300m"),
    ("ESM2-650M", "ESM2_650M", "peint_650m"),
]

# One hue family per backbone; base = light, PEINT = dark.
BASE_LM_PAIR_COLORS = {
    "ESM2-150M": ("#9ecae1", "#08519c"),
    "ESM-C 300M": ("#fdae6b", "#d94801"),
    "ESM2-650M": ("#a1d99b", "#238b45"),
}


def base_lm_run_and_model_names():
    """(run_dirs, display_names) for the six entries in paired display order.

    Keys are ``"<label>|base"`` / ``"<label>|peint"``; run_dirs maps them to the
    result sub-dir, display_names to the legend label.
    """
    run_names, model_names = {}, {}
    for lm, base_col, peint_run in BASE_LM_CONFIG:
        run_names[f"{lm}|base"] = base_col
        run_names[f"{lm}|peint"] = peint_run
        model_names[f"{lm}|base"] = lm
        model_names[f"{lm}|peint"] = f"PEINT ({lm})"
    return run_names, model_names


def base_lm_palette():
    """Six colors in the paired config order (base light, PEINT dark)."""
    palette = []
    for lm, _, _ in BASE_LM_CONFIG:
        palette.extend(BASE_LM_PAIR_COLORS[lm])
    return palette


def load_spearman_overlap(results_dir, run_dirs):
    """Load ``spearman_results.csv`` for each run dir, restricted to the family
    set common to all of them (fair comparison).

    ``run_dirs`` maps an arbitrary key -> run sub-dir under ``results_dir``; the
    returned long df keeps that key in a ``key`` column.
    """
    dfs, overlap = [], None
    for key, rd in run_dirs.items():
        df = pd.read_csv(results_dir / rd / "spearman_results.csv")
        df["key"] = key
        dfs.append(df)
        fams = set(df["family"])
        overlap = fams if overlap is None else (overlap & fams)
    df = pd.concat(dfs, ignore_index=True)
    return df[df["family"].isin(overlap)].reset_index(drop=True)


def released_depth_spearman(column, families=None, max_len=1022):
    """Per (family, assay_type, mutational_depth) Spearman for a released baseline
    column, computed from the ProteinGym release.

    Depth = number of substitutions in ``mutant`` (":"-separated), bucketed to
    "5+". Schema matches PEINT's ``spearman_by_mutation_depth.csv`` so base pLMs
    slot into the same depth plot. ``max_len`` drops long assays to match the
    PEINT evaluation (skips target sequences > 1022).
    """
    import scipy.stats
    from protevo.vep.official_baselines import RELEASED_SCORES_DIR, DMS_REFERENCE

    ref = pd.read_csv(DMS_REFERENCE).set_index("DMS_id")
    ids = list(ref.index) if families is None else [f for f in families if f in ref.index]
    rows = []
    for dms_id in ids:
        tseq = ref.loc[dms_id, "target_seq"]
        if not isinstance(tseq, str) or len(tseq) > max_len:
            continue
        sf = RELEASED_SCORES_DIR / f"{dms_id}.csv"
        if not sf.exists():
            continue
        df = pd.read_csv(
            sf,
            usecols=lambda c: c in ("DMS_score", "mutant") or c == column,
            low_memory=False,
        )
        if column not in df.columns:
            continue
        assay = ref.loc[dms_id, "coarse_selection_type"]
        depth = df["mutant"].str.count(":").fillna(0).astype(int) + 1
        df = df.assign(_bin=depth.where(depth < 5, other=5))
        for b, g in df.groupby("_bin"):
            pair = g[["DMS_score", column]].dropna()
            if pair["DMS_score"].nunique() < 2 or pair[column].nunique() < 2:
                continue
            rho = scipy.stats.spearmanr(pair["DMS_score"], pair[column]).correlation
            label = "5+" if b >= 5 else str(int(b))
            rows.append((label, dms_id, assay, len(pair), float(rho)))
    return pd.DataFrame(
        rows, columns=["mutational_depth", "family", "assay_type", "count", "spearman"]
    )


def materialize_official_depth(results_dir, columns, families, overwrite=False):
    """Write ``spearman_by_mutation_depth.csv`` for each released base pLM (the
    PEINT runs already ship theirs), restricted to ``families`` for fairness.
    """
    for col in columns:
        out = results_dir / col / "spearman_by_mutation_depth.csv"
        if out.exists() and not overwrite:
            continue
        d = released_depth_spearman(col, families=families)
        out.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(out, index=False)
        print(f"materialized depth baseline: {out} ({len(d)} rows)")
