"""Figure: the evolution<->design spectrum of ESM2-MCMC simulation, endpoint quality, and cost.

Three panels (reviewer response — situating PEINT vs protein-LM MCMC simulators):

(a) **Spectrum.** Sweeping the length-independent LM weight lambda of an lm-design-style energy
    (LM + structure + n-gram) traces one monotonic curve from evolutionary fidelity (high lambda,
    strong ESM2 pseudolikelihood filter, Bitbol-like) to inverse-folding/design (low lambda). As
    lambda falls, acceptance rises but conservation (JSD) collapses toward random.
(b) **Endpoints.** Per-family JSD-to-real by model: the evolutionary MCMC (Bitbol) and PEINT match
    Real and beat the classical simulators; the design-energy endpoint (lm-design) is the worst
    (near-random sequences wash out conservation).
(c) **Cost.** Generation wall-clock (7 families) — PEINT (one-shot autoregressive) is far cheaper
    than either MCMC; the evolutionary MCMC (Bitbol, low acceptance) is the slowest.

Reads the benchmark outputs under figures/output/esm_mcmc/ (regenerate with
benchmarks/esm_mcmc_lmdesign_{simulate,sweep,eval}.py and benchmarks/esm_mcmc_eval.py). The
per-method runtimes are one-off measurements recorded below with provenance.

Run:  python -m figures.figure_esm_mcmc_spectrum
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import paper_config as cfg
from paper.plot_style import _set_publication_style

BASE = Path(cfg.FIGURES_DIR) / "esm_mcmc"
OUT_STEM = Path(cfg.FIGURES_DIR) / "figure_esm_mcmc_spectrum"

# Reference JSD (1a2t) for the spectrum panel's guide lines (from the eval).
PEINT_JSD_REF, REAL_JSD_REF = 0.212, 0.167

# Generation wall-clock, 7 families (excl. 1bf2 which lm-design did not run), one-off measurements:
#   PEINT per-family timing (benchmarks + scratch timing), lm-design sim_stats, Bitbol hamming stats.
RUNTIME_MIN = {"PEINT": 10.8, "lm-design": 53.1, "Bitbol": 154.5}

# Model display + grouping for the endpoints panel.
EVOLUTIONARY = ["Real", "PEINT", "Bitbol"]
CLASSICAL = ["WAG", "LG", "LG4X", "LG+C60", "LG+S256"]
DESIGN = ["lm-design"]
JSD_RENAME = {"Real (other split)": "Real", "PEINT (Progressive)": "PEINT", "ESM-MCMC": "Bitbol"}


def load_jsd_by_model() -> dict:
    """Per-family JSD-to-real arrays per model over the common (non-1bf2) families."""
    lmd = pd.read_csv(BASE / "lmdesign" / "eval" / "jsd.csv")          # lm-design + standard models
    fams = sorted(lmd.family.unique())
    vals = {}
    for m in lmd.model.unique():
        vals[JSD_RENAME.get(m, m)] = lmd[lmd.model == m].sort_values("family")["jsd"].to_numpy()
    bit = pd.read_csv(BASE / "eval" / "jsd_comparison.csv")            # has ESM-MCMC (Bitbol)
    bit = bit[bit.family.isin(fams) & (bit.model == "ESM-MCMC")].sort_values("family")
    vals["Bitbol"] = bit["jsd"].to_numpy()
    return vals


def main():
    _set_publication_style()
    sweep = pd.read_csv(BASE / "lmdesign" / "sweep" / "sweep.csv").sort_values("lambda", ascending=False)
    vals = load_jsd_by_model()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 3.4))

    # --- (a) spectrum: lambda vs acceptance (left) and JSD (right) ---
    x = np.arange(len(sweep))
    labels = [("0" if l == 0 else f"{l:g}") for l in sweep["lambda"]]
    c_acc, c_jsd = "#4C72B0", "#C44E52"
    l_acc, = ax1.plot(x, sweep["acceptance"], "-o", color=c_acc, ms=4, lw=1.4)
    ax1.set_ylabel("acceptance", color=c_acc, fontsize=9)
    ax1.tick_params(axis="y", labelcolor=c_acc)
    ax1.set_ylim(0, 1)
    axr = ax1.twinx()
    l_jsd, = axr.plot(x, sweep["JSD"], "-s", color=c_jsd, ms=4, lw=1.4)
    l_ref = axr.axhline(PEINT_JSD_REF, ls="--", lw=0.8, color="0.4")
    axr.set_ylabel("JSD to real  (lower = better)", color=c_jsd, fontsize=9)
    axr.tick_params(axis="y", labelcolor=c_jsd)
    ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=8)
    ax1.set_xlabel("LM weight  λ   (evolutionary  →  design)", fontsize=9)
    ax1.set_title("(a) evolution–design spectrum", fontsize=10)
    ax1.legend([l_acc, l_jsd, l_ref], ["acceptance", "JSD to real", "PEINT/Real JSD"],
               loc="upper left", fontsize=7, frameon=False)

    # --- (b) endpoints: JSD by model (bar = median, error bars = family IQR) ---
    order = [m for m in EVOLUTIONARY + CLASSICAL + DESIGN if m in vals]
    colors = {**{m: "#55A868" for m in EVOLUTIONARY}, **{m: "0.6" for m in CLASSICAL},
              **{m: "#C44E52" for m in DESIGN}}
    med = np.array([np.median(vals[m]) for m in order])
    p25 = np.array([np.percentile(vals[m], 25) for m in order])
    p75 = np.array([np.percentile(vals[m], 75) for m in order])
    ax2.bar(range(len(order)), med, color=[colors[m] for m in order], width=0.72,
            yerr=[med - p25, p75 - med], capsize=2, error_kw={"lw": 0.7, "ecolor": "0.25"})
    ax2.axhline(np.median(vals["Real"]), ls="--", lw=0.8, color="#55A868")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels(order, rotation=40, ha="right", fontsize=7.5)
    ax2.set_ylabel("JSD to real  (lower = better)", fontsize=9)
    ax2.set_title("(b) endpoint conservation", fontsize=10)
    from matplotlib.patches import Patch
    ax2.legend(handles=[Patch(color="#55A868", label="evolutionary"), Patch(color="0.6", label="classical"),
                        Patch(color="#C44E52", label="design")], fontsize=6.5, frameon=False, loc="upper left")

    # --- (c) runtime ---
    meth = ["PEINT", "lm-design", "Bitbol"]
    rc = {"PEINT": "#55A868", "lm-design": "#C44E52", "Bitbol": "#4C72B0"}
    bars = ax3.bar(range(len(meth)), [RUNTIME_MIN[m] for m in meth], color=[rc[m] for m in meth], width=0.62)
    for b, m in zip(bars, meth):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{RUNTIME_MIN[m]:.0f}m",
                 ha="center", va="bottom", fontsize=7.5)
    ax3.set_xticks(range(len(meth))); ax3.set_xticklabels(meth, fontsize=8)
    ax3.set_ylabel("generation time (min, 7 families)", fontsize=9)
    ax3.set_title("(c) cost", fontsize=10)
    ax3.set_ylim(0, max(RUNTIME_MIN.values()) * 1.18)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT_STEM}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {OUT_STEM}.png / .pdf")
    print("JSD medians:", {k: round(float(np.median(v)), 3) for k, v in vals.items()})


if __name__ == "__main__":
    main()
