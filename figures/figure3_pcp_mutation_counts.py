import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, FrozenSet, Tuple
import math

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
import pandas as pd
from ete3 import Tree
from cherryml import caching as cherryml_caching

from protevo.utils import read_msa
from protevo.io import read_tree, write_msa
from protevo.simulation import simulate_alisim_evolution_subtree
from protevo.simulation._alisim import _UDM_NEX_PATH
from protevo import caching as protevo_caching

from paper.alignment import run_mafft_add
from paper.historian import esmc_historian_dirs
from paper.model_style import model_colors
import paper_config as cfg

# ESM-C (rev2) PEINT progressive historian reconstruction (remove-dummy output), added as a
# sixth model alongside the rev1 WAG/LG/LG+S256/PEINT-ESM2/Real. Projected into the empirical
# frame exactly like PEINT-ESM2, then matched to the real branches. The "refine" run, the
# same one the indel panels use. Resolved lazily so importing this module (for get_branches /
# match_branches, which several benchmarks do) never depends on the rev2 data being present.
def esmc_recon_dir():
    return esmc_historian_dirs(cfg.RESULTS_R2_DIR, "refine")["reconstructions"]
# Models whose trees are in the protevo (read_tree) format rather than AliSim .full.treefile.
_PROTEVO_TREE_MODELS = ("PEINT", "PEINT_ESMC")

@protevo_caching.cached_parallel_computation(
    parallel_arg="families",
    exclude_args=["num_processes"],
    output_dirs=[
        "output_msa_dir",
    ],
    write_extra_log_files=True,
)
def copy_gap_pattern(
    real_msa_dir: str,
    sim_msa_dir: str,
    families: List[str],
    num_processes: int,
    output_msa_dir: Optional[str] = None,
):

    for family in families:

        real_msa = read_msa(
            os.path.join(
                real_msa_dir,
                family + '.txt'
            )
        )

        sim_msa = read_msa(
            os.path.join(
                sim_msa_dir,
                family + '.txt'
            )
        )

        assert len(list(real_msa.values())[0]) == len(list(sim_msa.values())[0]), "MSAs must be the same length"

        new_msa = {}
        for l,s in sim_msa.items():
            rs = real_msa.get(l.replace('i', 'internal-')) #relabeled internal nodes to fit size of alisim labels
            matched_gap_seq = ''.join(
                [c2 if c2 == '-' else c1 for c1, c2 in zip(s, rs)]
            )
            new_msa[l] = matched_gap_seq

            if l.startswith('s'):
                #leaf nodes, should have correct gaps
                assert matched_gap_seq == s, "Leaves should have identical gap patterns"
        
        output_msa_path = os.path.join(output_msa_dir, family + '.txt')
        write_msa(new_msa, output_msa_path)

def get_node_descendants(node: Tree) -> frozenset:
    return frozenset(leaf.name for leaf in node.get_leaves())

def match_branches(original_branches: dict, simulated_branches:dict, eval_leaf_set: Optional[Set[str]]):

    '''Match the branches between trees. This is to fix issues arising from potential 
    renaming of internal nodes.
    Each branch is defined by the leaves downstream of it.
    To maintain only the branches that we need, we'll provide a set of leaves, and only look at branches
    whose leaves are in that set
    '''

    matched = dict()
    original_only = dict()
    simulated_only = dict()

    leafsets = set(original_branches.keys()) | set(simulated_branches.keys())

    for leafset in leafsets:

        if eval_leaf_set and leafset.difference(eval_leaf_set):
            #leafset has stuff not in the eval leaves, not the part of the tree we want
            continue

        orig = original_branches.get(leafset)
        sim = simulated_branches.get(leafset)

        if orig and sim:
            #that set of leaves found in both
            matched[leafset] = {
                'leaves': leafset,
                'orig_mutations': orig['mutations'],
                'sim_mutations': sim['mutations'],
                'orig_sites': orig['num_sites'],
                'sim_sites': sim['num_sites'],
                'orig_back_mutations': orig['back_mutations'],
                'sim_back_mutations': sim['back_mutations'],
                'orig_back_sites': orig['back_num_sites'],
                'sim_back_sites': sim['back_num_sites'],
                # AliSim re-optimises branch lengths per model (--site-rate MODEL), so each
                # model evolves the MSA along its own tree. Keep the shared historian length
                # (orig) as a common reference, and each model's own length (sim) for the
                # per-branch rate plots, which must use the length that model actually evolved.
                'orig_branch_length': orig['branch_length'],
                'sim_branch_length': sim['branch_length'],
                'branch_name': orig['child']
            }
        elif orig:
            original_only[leafset] = orig
        else:
            simulated_only[leafset] = sim

    return {
        'matched': matched,
        'original_only': original_only,
        'simulated_only': simulated_only
    }

def get_mutations_for_branch(
    seq1: str,
    seq2: str,
    positions: Optional[Set[int]] = None
):

    assert len(seq1) == len(seq2), "Sequences must be the same length to count mutations"

    mutations = 0
    num_sites = 0

    if positions:
        s1 = ''.join([seq1[pos] for pos in positions])
        s2 = ''.join([seq2[pos] for pos in positions])
    else:
        s1 = seq1
        s2 = seq2

    for c1, c2 in zip(s1.upper(), s2.upper()):
        #upper because historian writes internal nodes with lowercase
        if c1 == '-' or c2 == '-':
            #don't count mutations when there's a gap. Don't want to penalize terrible alignments
            continue
        else:
            num_sites += 1
            if c1 != c2:
                mutations += 1
    
    return mutations, num_sites

_AA_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
_AA_INDEX = {a: i for i, a in enumerate(_AA_ALPHABET)}


def get_per_site_mutation_counts(tree: Tree, sequences: Dict[str, str], length: int):
    '''For each alignment column, count branches on which the parent->child sequence differs
    (gaps in either endpoint exclude that column from that branch's contribution).
    Also accumulates back-mutations: at each grandparent->parent->child triple, a back-mutation
    at site i is grandparent[i] == child[i] != parent[i] (all three non-gap). The back-mutation
    count is attributed to the parent->child branch.
    Returns (mut_counts, branch_counts, back_counts, triple_counts) of length `length`.'''
    mut_counts = np.zeros(length, dtype=int)
    branch_counts = np.zeros(length, dtype=int)
    back_counts = np.zeros(length, dtype=int)
    triple_counts = np.zeros(length, dtype=int)
    for node in tree.traverse():
        if node.is_root():
            continue
        parent = node.up
        parent_seq = sequences.get(parent.name)
        child_seq = sequences.get(node.name)
        if not parent_seq or not child_seq:
            continue
        for i, (p, c) in enumerate(zip(parent_seq.upper(), child_seq.upper())):
            if p == '-' or c == '-':
                continue
            branch_counts[i] += 1
            if p != c:
                mut_counts[i] += 1
        if parent.is_root():
            continue
        grandparent_seq = sequences.get(parent.up.name)
        if not grandparent_seq:
            continue
        for i, (g, p, c) in enumerate(zip(grandparent_seq.upper(), parent_seq.upper(), child_seq.upper())):
            if g == '-' or p == '-' or c == '-':
                continue
            triple_counts[i] += 1
            if g == c and g != p:
                back_counts[i] += 1
    return mut_counts, branch_counts, back_counts, triple_counts


def get_per_site_leaf_entropy(msa: Dict[str, str], leaf_names: Set[str], length: int):
    '''Per-column Shannon entropy of empirical amino-acid frequencies across `leaf_names`
    (gaps and non-standard characters excluded). Returns (entropies, observation_counts).
    Entropies are nan where no observations remain.'''
    counts = np.zeros((length, 20), dtype=int)
    for name, seq in msa.items():
        if name not in leaf_names:
            continue
        for i, c in enumerate(seq.upper()):
            idx = _AA_INDEX.get(c)
            if idx is not None:
                counts[i, idx] += 1
    totals = counts.sum(axis=1)
    entropies = np.full(length, np.nan)
    for i in range(length):
        if totals[i] == 0:
            continue
        p = counts[i] / totals[i]
        nz = p[p > 0]
        entropies[i] = float(-np.sum(nz * np.log(nz)))
    return entropies, totals


def load_udm_profiles(nex_path: str) -> np.ndarray:
    '''Parse the 256 UDM profile frequency vectors (each 20-dim) from the IQ-TREE .nex
    model definition, ordered by component index C0000..C0255 — the same order the
    -wspm .siteprob posterior columns (p1..p256) follow.'''
    pattern = re.compile(r'frequency\s+UDM0256LCLR_C(\d{4})\s*=\s*([0-9.eE+\- ]+)')
    profiles = {}
    with open(nex_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            freqs = [float(x) for x in m.group(2).split()]
            if len(freqs) == 20:
                profiles[int(m.group(1))] = np.array(freqs)
    if len(profiles) != 256:
        raise ValueError(f"Expected 256 UDM profiles, parsed {len(profiles)} from {nex_path}")
    return np.stack([profiles[i] for i in range(256)])


def _shannon_entropy(p: np.ndarray) -> float:
    nz = p[p > 0]
    return float(-np.sum(nz * np.log(nz)))


def compute_profile_entropies(siteprob_path: str, udm_profiles: np.ndarray, length: int):
    '''From a -wspm .siteprob file (Site + posterior over the 256 profiles), compute per site:
      - profile_post_entropy: entropy of the posterior over profiles (assignment uncertainty),
      - effective_profile_entropy: entropy of the posterior-mean AA distribution
        (Σ_k P(k) profile_k) — how narrow the site's effective profile is.
    Indexed by alignment site (0-based) via the .siteprob Site column. Returns two
    length-`length` arrays, nan where a site is absent.'''
    post_ent = np.full(length, np.nan)
    eff_ent = np.full(length, np.nan)
    n_profiles = udm_profiles.shape[0]
    with open(siteprob_path) as f:
        f.readline()  # header: Site p1 ... p256
        for line in f:
            parts = line.split()
            if not parts:
                continue
            site = int(parts[0]) - 1  # .siteprob Site is 1-based
            if site < 0 or site >= length:
                continue
            probs = np.array([float(x) for x in parts[1:]])
            if probs.size != n_profiles:
                continue
            post_ent[site] = _shannon_entropy(probs)
            effective = probs @ udm_profiles
            total = effective.sum()
            if total > 0:
                eff_ent[site] = _shannon_entropy(effective / total)
    return post_ent, eff_ent


def count_back_mutations(grandparent_seq: str, parent_seq: str, child_seq: str):
    '''Back-mutations attributed to the parent->child branch: positions where
    grandparent[i] == child[i] != parent[i] (gaps excluded). Returns
    (count, num_scored), num_scored being the positions with all three non-gap --
    the denominator for a per-branch reversal rate.'''
    count = 0
    num_scored = 0
    for g, p, c in zip(grandparent_seq.upper(), parent_seq.upper(), child_seq.upper()):
        if g == '-' or p == '-' or c == '-':
            continue
        num_scored += 1
        if g != p and c == g:
            count += 1
    return count, num_scored


def get_branches(
    tree: Tree,
    sequences: Dict[str, str]
):

    '''Eval node set contains the leaves to analyze. Get the lowest internal node that contains them all,
    and then run down the tree, such that for each branch, you get the number of mutations between parent and
    child.
    For these trees, a branch is also a set of leaves that is downstream of this branch. So we can save this data
    in a dictionary of {leaf_set} -> num_mutations, num_positions.
    Back-mutations are attributed to a branch using the (grandparent, parent, child) triple — None when no grandparent.
    '''

    branches = dict()

    for node in tree.traverse():
        if node.is_root():
            #root doesn't have a parent. We're defining on the bottom of the branch
            continue

        parent = node.up
        child_leaves = get_node_descendants(node)
        parent_seq = sequences.get(parent.name)
        child_seq = sequences.get(node.name)
        grandparent = parent.up
        grandparent_seq = sequences.get(grandparent.name) if grandparent is not None else None

        mutations = 0
        num_sites = 0
        back_mutations = None
        back_num_sites = None

        if parent_seq and child_seq:
            mutations, num_sites = get_mutations_for_branch(parent_seq, child_seq)
            if grandparent_seq:
                back_mutations, back_num_sites = count_back_mutations(grandparent_seq, parent_seq, child_seq)

        branches[child_leaves] = {
            'parent': parent.name,
            'child': node.name,
            'branch_length': node.dist,
            'mutations': mutations,
            'num_sites': num_sites,
            'back_mutations': back_mutations,
            'back_num_sites': back_num_sites,
        }

    return branches



# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
# Split out of the __main__ block so the panels can be redrawn from the two tables the
# aggregation writes (simulation_internal_analysis.csv, path_saturation.csv) instead of
# repeating the ~30 minute branch-matching pass. `--replot` below does exactly that.

# Internal simulator keys, their display names, and the plot order. The keys are what the
# aggregation writes into the CSVs; the labels are what the paper shows, and they have to
# match the other figures rather than leaking these internal spellings into a legend.
MODEL_ORDER = ['WAG', 'LG', 'LG_S256', 'PEINT', 'PEINT_ESMC', 'Real (Inferred)']
SCATTER_ORDER = ['WAG', 'LG', 'LG_S256', 'PEINT', 'PEINT_ESMC']
MODEL_LABELS = {
    'WAG': 'WAG',
    'LG': 'LG',
    'LG_S256': 'LG+S256',
    'PEINT': 'PEINT (ESM2)',
    'PEINT_ESMC': 'PEINT (ESM-C)',
    'Real (Inferred)': 'Real (inferred)',
}


def model_palette():
    """Internal simulator key -> color, via the shared canonical map."""
    mc = model_colors()
    return {
        'WAG': mc['WAG'],
        'LG': mc['LG'],
        'LG_S256': mc['LG+S256'],
        'PEINT': mc['PEINT (ESM2)'],
        'PEINT_ESMC': mc['PEINT (ESM-C)'],
        'Real (Inferred)': mc['Real'],
    }


def _save(fig, stem, dpi=300):
    os.makedirs(str(cfg.FIGURES_DIR), exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(str(cfg.FIGURES_DIR), f'{stem}.{ext}'),
                    dpi=dpi, bbox_inches='tight')
    print(f'  wrote {cfg.FIGURES_DIR}/{stem}.{{pdf,png}}')


def plot_parent_child_pairs(df):
    """Per-branch fraction of mutated sites, boxed by (per-model) branch length."""
    df = df.copy()
    df['branch_length_q'] = df['branch_length'].apply(lambda x: math.floor(x * 10) / 10)
    subdf = df[df.branch_length < 1.1]   # very few longer, likely trouble with inference

    mpl.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = model_palette()
    branch_lengths = sorted(subdf['branch_length_q'].unique())
    n_simulators = len(MODEL_ORDER)

    # Keep the n-box group inside the unit-wide bin slot.
    step = 0.9 / n_simulators
    box_width = step * 0.8
    positions, all_data, all_colors = [], [], []
    for i, bl in enumerate(branch_lengths):
        for j, sim in enumerate(MODEL_ORDER):
            data = subdf[(subdf['branch_length_q'] == bl) & (subdf['simulator'] == sim)]['mutations']
            if len(data) > 0:
                positions.append(i + (j - n_simulators / 2 + 0.5) * step)
                all_data.append(data)
                all_colors.append(colors[sim])

    bp = ax.boxplot(all_data, positions=positions, widths=box_width, patch_artist=True,
                    boxprops=dict(linewidth=0.5),
                    whiskerprops=dict(linewidth=0.5),
                    flierprops={"marker": "o", "markersize": 0.3},
                    medianprops=dict(linestyle='--', color='black', linewidth=0.5))
    for patch, color in zip(bp['boxes'], all_colors):
        patch.set_facecolor(color)

    # Upper left: the boxes rise left-to-right, so 'best' put the legend over the widest bins.
    ax.legend(handles=[Patch(facecolor=colors[sim], edgecolor='black', linewidth=0.5,
                             label=MODEL_LABELS[sim]) for sim in MODEL_ORDER],
              title='Model', fontsize=9, loc='upper left', framealpha=0.9)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    sns.despine()
    ax.tick_params(width=0.5, length=2, which='both')
    ax.set_ylim(0, 1)
    ax.set_xticks(ticks=range(len(branch_lengths)),
                  labels=[f'{bl:.1f}' for bl in branch_lengths], fontsize=10)
    ax.set_yticks(ticks=np.arange(0, 1, 0.1),
                  labels=[f'{t:.1f}' for t in np.arange(0, 1, 0.1)], fontsize=10)
    ax.set_xlabel('Branch Length (per-model)', fontsize=12)
    ax.set_ylabel('Fraction Mutated Sites', fontsize=12)
    _save(fig, 'parent_child_pairs_all_models')
    plt.close(fig)


def plot_per_family_scatter(df):
    """Per-family median mutation rate, each simulator against real."""
    colors = model_palette()
    per_family_medians = df.groupby(['simulator', 'family']).agg({'mutations': 'median'}).reset_index()
    medians = per_family_medians.pivot(index='family', columns='simulator', values='mutations')

    fig, axs = plt.subplots(1, len(SCATTER_ORDER),
                            figsize=(2.75 * len(SCATTER_ORDER), 2.5), sharey=True)
    for ax, sim in zip(axs, SCATTER_ORDER):
        x, y = medians['Real (Inferred)'], medians[sim]
        keep = ~(x.isna() | y.isna())
        ax.scatter(x[keep], y[keep], alpha=0.6, s=20, color=colors[sim])
        ax.plot([0, 0.4], [0, 0.4], 'k-', linewidth=1, label='y=x')
        ax.tick_params(width=0.5, length=2, which='both')
        ax.set_xlim(0, 0.4)
        ax.set_ylim(0, 0.4)
        ax.set_xlabel(MODEL_LABELS['Real (Inferred)'], fontsize=10)
        ax.set_ylabel(MODEL_LABELS[sim], fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        sns.despine()
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    fig.suptitle('Per family median mutation rate (Eval Subtrees)')
    _save(fig, 'per_family_median_mutation_rate_all_models')
    plt.close(fig)


def _binned_mean(x, y, bins):
    """Mean and SEM of y within each x-bin; only bins with data are returned."""
    idx = np.digitize(x, bins) - 1
    centers, means, sems = [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        means.append(float(np.mean(y[sel])))
        sems.append(float(np.std(y[sel]) / math.sqrt(n)))
    return np.array(centers), np.array(means), np.array(sems)


def plot_back_mutation_and_root_leaf(df, path_df):
    """Two companion panels, both read off each model's own simulated tree:

    (a) per-branch reversal rate (A->B->A) vs the model's own branch length — the cycling
        signal, a profile-shape property that persists at matched branch length;
    (b) root->leaf saturation — substitutions summed along the path vs the net root->leaf
        divergence (both per non-gap site). Distance above y=x is homoplasy: the reverted
        or convergent substitutions hidden from an endpoint comparison.
    """
    colors = model_palette()
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # ---- (a) per-branch back-mutation rate vs (per-model) branch length ----
    ax = axs[0]
    bdf = df[df['back_num_sites'].notna() & (df['back_num_sites'] > 0)
             & (df['branch_length'] < 1.1)].copy()
    bdf['back_fraction'] = bdf['back_mutations'] / bdf['back_num_sites']
    bl_bins = np.arange(0, 1.1 + 1e-9, 0.1)
    for sim in MODEL_ORDER:
        d = bdf[bdf['simulator'] == sim]
        if d.empty:
            continue
        c, m, se = _binned_mean(d['branch_length'].to_numpy(), d['back_fraction'].to_numpy(), bl_bins)
        ax.plot(c, m, '-o', ms=3, lw=1, color=colors[sim], label=MODEL_LABELS[sim])
        ax.fill_between(c, m - se, m + se, color=colors[sim], alpha=0.2, lw=0)
    ax.set_xlabel('Branch Length (per-model)', fontsize=11)
    ax.set_ylabel('Back-mutations / scored triple', fontsize=11)
    ax.legend(fontsize=8, frameon=False, title='Model')

    # ---- (b) root->leaf saturation: path-summed vs net divergence ----
    ax = axs[1]
    # Require a minimum root<->leaf overlap: with only a handful of shared non-gap sites the
    # per-site ratios blow up (root_leaf_sites down to 1 -> path/site in the hundreds). The
    # 1st percentile of overlap is ~40 sites, so >=30 drops only degenerate near-empty paths.
    # Then restrict to paths present for every simulator so all arms sum the same set.
    n_sim = path_df['simulator'].nunique()
    enough_overlap = path_df[path_df['root_leaf_sites'] >= 30].copy()
    per_path_sims = enough_overlap.groupby(['family', 'leaf'])['simulator'].transform('nunique')
    shared = enough_overlap[per_path_sims == n_sim].copy()
    shared['net_per_site'] = shared['root_leaf_hamming'] / shared['root_leaf_sites']
    shared['path_per_site'] = shared['path_mutations'] / shared['root_leaf_sites']
    lim = float(shared['net_per_site'].quantile(0.99))
    net_bins = np.linspace(0, lim, 16)
    for sim in MODEL_ORDER:
        d = shared[shared['simulator'] == sim]
        if d.empty:
            continue
        c, m, se = _binned_mean(d['net_per_site'].to_numpy(), d['path_per_site'].to_numpy(), net_bins)
        ax.plot(c, m, '-o', ms=3, lw=1, color=colors[sim], label=MODEL_LABELS[sim])
        ax.fill_between(c, m - se, m + se, color=colors[sim], alpha=0.2, lw=0)
    ax.plot([0, lim], [0, lim], 'k--', lw=0.8, label='y = x (no homoplasy)')
    ax.set_xlim(0, lim)
    ax.set_xlabel('Net root→leaf divergence / site', fontsize=11)
    ax.set_ylabel('Path-summed substitutions / site', fontsize=11)
    ax.legend(fontsize=8, frameon=False)

    for ax in axs:
        ax.tick_params(width=0.5, length=2, which='both')
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
    sns.despine()
    fig.tight_layout()
    _save(fig, 'back_mutation_and_root_leaf', dpi=160)
    plt.close(fig)


def render_all(df, path_df):
    plot_parent_child_pairs(df)
    plot_per_family_scatter(df)
    plot_back_mutation_and_root_leaf(df, path_df)


def _writable_out(base, name):
    """Where a pipeline output goes: the authoritative tree if we own it, else DERIVED_DIR.

    The data trees are inputs for everyone except the machine that produced them, and a
    published/shared copy is read-only. Deciding on ``os.access(W_OK)`` rather than a config
    flag keeps the producing machine bit-identical -- same output path, so every protevo
    cache key stays warm -- while a read-only mirror diverts the writes instead of raising.
    """
    shipped = Path(base) / name
    if shipped.is_dir() and os.access(shipped, os.W_OK):
        return str(shipped)
    out = Path(cfg.DERIVED_DIR) / name
    out.mkdir(parents=True, exist_ok=True)
    return str(out)


def _find_table(dirs, stem):
    """First existing <dir>/<stem>.csv{,.gz} across dirs, else None."""
    for d in dirs:
        for name in (f'{stem}.csv', f'{stem}.csv.gz'):
            p = Path(d) / name
            if p.exists():
                return p
    return None


def replot_from_csv(distances_dir=None):
    """Redraw all three panels from the saved aggregation tables (seconds, not ~30 min).

    Looks in --distances-dir if given, then the shipped figure_data copies, then the
    directory the full pipeline writes to. The figure_data copies are gzipped and carry
    only the columns these panels read, which is why they are ~19 MB rather than ~400 MB.
    """
    search = ([distances_dir] if distances_dir else
              [cfg.FIGURE_DATA_DIR / 'leaf_distances', cfg.LEAF_DISTANCES_DIR])
    internal = _find_table(search, 'simulation_internal_analysis')
    saturation = _find_table(search, 'path_saturation')
    missing = [s for s, f in (('simulation_internal_analysis', internal),
                              ('path_saturation', saturation)) if f is None]
    if missing:
        raise FileNotFoundError(
            f"Could not find {', '.join(missing)} (.csv or .csv.gz) in "
            f"{[str(s) for s in search]}. Run this module without --replot to build them, "
            f"or point --distances-dir at a directory holding both.")
    print(f'replotting from {internal.parent}')
    render_all(pd.read_csv(internal), pd.read_csv(saturation))



def _run_full_pipeline():
    
    # Absolute, from config. These used to be relative, which silently tied the whole
    # benchmark to one working directory: run it from anywhere else and the AliSim /
    # mafft-add / copy_gap_pattern cache missed and everything recomputed.
    protevo_caching.set_cache_dir(str(cfg.PROTEVO_CACHE_DIR))
    protevo_caching.set_log_level(9)
    protevo_caching.set_dir_levels(3)

    cherryml_caching.set_cache_dir(str(cfg.CHERRYML_CACHE_DIR))
    cherryml_caching.set_log_level(9)
    cherryml_caching.set_dir_levels(3)

    np.random.seed(0)
    num_processes = 20 

    families_file = str(cfg.SIM_FAMILIES_FILE)
    with open(families_file, 'r') as f:
        families = [line.strip() for line in f.readlines()]

    output_dir = str(cfg.RESULTS_DIR)
    historian_msa_dir = os.path.join(str(cfg.SIMULATIONS_DIR), 'real_msa_historian')
    historian_tree_dir = os.path.join(str(cfg.SIMULATIONS_DIR), 'real_subtree_historian')
    peint_historian_dir = os.path.join(str(cfg.SIMULATIONS_DIR), 'peint_msa_historian')
    simulation_dir = str(cfg.SIMULATIONS_DIR)
    rerooted_tree_dir = str(cfg.TREE_DIR)

    distances_dir = _writable_out(output_dir, 'leaf_distances')
    output_lg_dir = _writable_out(simulation_dir, 'lg_subtree_simulation')

    renamed_lg_dir = simulate_alisim_evolution_subtree(
        tree_dir = historian_tree_dir,
        msa_dir = historian_msa_dir,
        families = families,
        evolutionary_model = 'LG',
        num_processes=8,
        output_msa_dir = output_lg_dir
    )['output_msa_dir']


    gap_transferred_lg_dir = copy_gap_pattern(
        real_msa_dir = historian_msa_dir,
        sim_msa_dir = renamed_lg_dir,
        families = families,
        num_processes = 8
    )['output_msa_dir']

    output_wag_dir = _writable_out(simulation_dir, 'wag_subtree_simulation')

    renamed_wag_dir = simulate_alisim_evolution_subtree(
        tree_dir = historian_tree_dir,
        msa_dir = historian_msa_dir,
        families = families,
        evolutionary_model = 'WAG',
        num_processes=8,
        output_msa_dir = output_wag_dir
    )['output_msa_dir']

    gap_transferred_wag_dir = copy_gap_pattern(
        real_msa_dir = historian_msa_dir,
        sim_msa_dir = renamed_wag_dir,
        families = families,
        num_processes = 8
    )['output_msa_dir']

    output_lg_s256_dir = _writable_out(simulation_dir, 'lg_s256_subtree_simulation')

    renamed_lg_s256_dir = simulate_alisim_evolution_subtree(
        tree_dir = historian_tree_dir,
        msa_dir = historian_msa_dir,
        families = families,
        evolutionary_model = 'LG+S256',
        num_processes=2,
        output_msa_dir=output_lg_s256_dir
    )['output_msa_dir']

    gap_transferred_lg_s256_dir = copy_gap_pattern(
        real_msa_dir = historian_msa_dir,
        sim_msa_dir = renamed_lg_s256_dir,
        families = families,
        num_processes = 2,
    )['output_msa_dir']
    
    # Project PEINT MSAs into the empirical (historian) reference frame so all simulators
    # share the same alignment columns. PEINT's progressive model allows insertions, so
    # without --keeplength its MSA has a different column count than the empirical frame.
    # Gap-stripping the input lets MAFFT --add re-place each PEINT node sequence (leaves
    # AND historian-inferred internals) against the empirical profile; --keeplength drops
    # any PEINT-only insertion columns. Node IDs are preserved so the PEINT tree's
    # name-based sequence lookup keeps working.
    peint_unaligned_dir = os.path.join(simulation_dir, 'peint_msa_unaligned')
    if not os.path.exists(peint_unaligned_dir):
        os.makedirs(peint_unaligned_dir)
    for family in families:
        src = os.path.join(peint_historian_dir, family + '.txt')
        dst = os.path.join(peint_unaligned_dir, family + '.txt')
        if os.path.exists(dst) or not os.path.exists(src):
            continue
        src_msa = read_msa(src)
        with open(dst, 'w') as f:
            for name, seq in src_msa.items():
                f.write(f'>{name}\n{seq.replace("-", "")}\n')

    peint_mafft_out = run_mafft_add(
        existing_alignment_dir=historian_msa_dir,
        new_sequences_dirs=(peint_unaligned_dir,),
        new_sequences_names=('peint',),
        families=families,
        num_processes=8,
        extra_command_line_args=('--keeplength',),
        output_all_sequences_msa_dir=os.path.join(simulation_dir, 'peint_subtree_emp_frame_all'),
        output_new_sequences_msa_dir=os.path.join(simulation_dir, 'peint_subtree_emp_frame'),
        output_old_sequences_msa_dir=os.path.join(simulation_dir, 'peint_subtree_emp_frame_ref'),
    )
    peint_empirical_frame_dir = peint_mafft_out['output_new_sequences_msa_dir']

    # ESM-C PEINT (rev2): identical empirical-frame projection as PEINT-ESM2 above, from the
    # rev2 progressive historian reconstruction. Same shared sim trees (rerooted_tree_dir), so
    # node IDs line up. This is the one extra MAFFT --add --keeplength cache (ESM-C wasn't in
    # rev1); WAG/LG/LG+S256 are gap-transferred (cached) and need no MAFFT.
    esmc_unaligned_dir = os.path.join(simulation_dir, 'esmc_msa_unaligned')
    if not os.path.exists(esmc_unaligned_dir):
        os.makedirs(esmc_unaligned_dir)
    esmc_recon = esmc_recon_dir()
    esmc_families = [f for f in families if os.path.exists(os.path.join(esmc_recon, f + '.txt'))]
    for family in esmc_families:
        dst = os.path.join(esmc_unaligned_dir, family + '.txt')
        if os.path.exists(dst):
            continue
        src_msa = read_msa(os.path.join(esmc_recon, family + '.txt'))
        with open(dst, 'w') as f:
            for name, seq in src_msa.items():
                f.write(f'>{name}\n{seq.replace("-", "")}\n')

    esmc_mafft_out = run_mafft_add(
        existing_alignment_dir=historian_msa_dir,
        new_sequences_dirs=(esmc_unaligned_dir,),
        new_sequences_names=('esmc',),
        families=esmc_families,
        num_processes=8,
        extra_command_line_args=('--keeplength',),
        output_all_sequences_msa_dir=os.path.join(simulation_dir, 'esmc_subtree_emp_frame_all'),
        output_new_sequences_msa_dir=os.path.join(simulation_dir, 'esmc_subtree_emp_frame'),
        output_old_sequences_msa_dir=os.path.join(simulation_dir, 'esmc_subtree_emp_frame_ref'),
    )
    esmc_empirical_frame_dir = esmc_mafft_out['output_new_sequences_msa_dir']

    model_dirs = {
        'PEINT': {
            'tree': rerooted_tree_dir,
            'msa': peint_empirical_frame_dir
        },
        'LG': {
            'tree': renamed_lg_dir,
            'msa': gap_transferred_lg_dir #Alisim saves the tree there too
        },
        'WAG': {
            'tree': renamed_wag_dir,
            'msa': gap_transferred_wag_dir
        },
        'LG_S256': {
            'tree': renamed_lg_s256_dir,
            'msa': gap_transferred_lg_s256_dir,
            # .siteprob (-wspm) lands in the raw sim dir, not the gap-transferred one.
            'siteprob_dir': renamed_lg_s256_dir,
        },
        # ESM-C last so the boxplot builder's per-branch model loop (which `break`s on a
        # zero-site branch) never short-circuits the established models when ESM-C lacks a
        # branch; ESM-C is projected into the empirical frame exactly like PEINT-ESM2.
        'PEINT_ESMC': {
            'tree': rerooted_tree_dir,
            'msa': esmc_empirical_frame_dir
        },
    }

    udm_profiles = load_udm_profiles(_UDM_NEX_PATH)

    matched_model_data = {
        m: dict() for m in list(model_dirs.keys())
    }

    per_site_rows = []
    path_rows = []

    def _accumulate_diagnostics(label, tree, msa, leaf_set, family_name, empirical_entropy,
                                siteprob_path=None, udm_profiles=None):
        length = len(next(iter(msa.values())))
        mut_counts, branch_counts, back_counts, triple_counts = get_per_site_mutation_counts(tree, msa, length)
        leaf_entropy, leaf_obs = get_per_site_leaf_entropy(msa, leaf_set, length)
        # empirical_entropy is the historian-leaf entropy in the empirical frame; LG/WAG/
        # LG_S256/Real share this frame natively, PEINT is projected here via mafft --keeplength.
        # Length mismatch should not happen — assert to catch it loudly.
        assert len(empirical_entropy) == length, (
            f"{label} {family_name}: msa length {length} != empirical frame length {len(empirical_entropy)}"
        )
        # Per-site profile entropies from the -wspm posterior (LG_S256 only; nan otherwise).
        profile_post_ent = np.full(length, np.nan)
        effective_profile_ent = np.full(length, np.nan)
        if siteprob_path and udm_profiles is not None and os.path.exists(siteprob_path):
            profile_post_ent, effective_profile_ent = compute_profile_entropies(
                siteprob_path, udm_profiles, length
            )
        for i in range(length):
            per_site_rows.append([
                family_name, label, i,
                int(mut_counts[i]), int(branch_counts[i]),
                int(back_counts[i]), int(triple_counts[i]),
                float(leaf_entropy[i]), int(leaf_obs[i]),
                float(empirical_entropy[i]),
                float(profile_post_ent[i]), float(effective_profile_ent[i]),
            ])

    for family in families:

        real_tree = Tree(
            os.path.join(
                historian_tree_dir,
                family + '.txt'
            ),
            format=1
        )

        real_msa = read_msa(
            os.path.join(
                historian_msa_dir,
                family + '.txt'
            )
        )

        eval_leaf_set = set(
            [n.name for n in real_tree.get_leaves()]
        )

        empirical_length = len(next(iter(real_msa.values())))
        empirical_entropy, _ = get_per_site_leaf_entropy(real_msa, eval_leaf_set, empirical_length)

        _accumulate_diagnostics('Real (Inferred)', real_tree, real_msa, eval_leaf_set, family, empirical_entropy)

        for model in model_dirs:

            if model in _PROTEVO_TREE_MODELS:
                sim_tree = read_tree(
                    os.path.join(
                        model_dirs[model]['tree'],
                        family + '.txt'
                    )
                ).to_ete3() #we save sim trees in the protevo tree format
            else:
                sim_tree = Tree(
                    os.path.join(
                        model_dirs[model]['tree'],
                        family + '.full.treefile'
                    ),
                    format=1
                )

            sim_msa = read_msa(
                os.path.join(
                    model_dirs[model]['msa'],
                    family + '.txt'
                )
            )

            original_branches = get_branches(
                real_tree,
                real_msa
            )

            sim_branches = get_branches(
                sim_tree,
                sim_msa
            )

            match_output = match_branches(
                original_branches,
                sim_branches,
                eval_leaf_set = eval_leaf_set
            )

            matched_model_data[model][family] = match_output['matched']

            siteprob_dir = model_dirs[model].get('siteprob_dir')
            siteprob_path = os.path.join(siteprob_dir, family + '.siteprob') if siteprob_dir else None
            _accumulate_diagnostics(model, sim_tree, sim_msa, eval_leaf_set, family, empirical_entropy,
                                    siteprob_path=siteprob_path, udm_profiles=udm_profiles)

    ## Path saturation computed from already-matched branches so every simulator is on
    ## the historian subtree (same eval_leaf_set, same matched-branch restriction). The
    ## subtree root in each simulator is identified by descendant-leafset equality with
    ## eval_leaf_set — the same definition of "same branch" that match_branches uses —
    ## so AliSim's renamed internals (internal-113 -> i118) and PEINT's full-tree
    ## namespace all resolve to the topologically correct node. (family, simulator)
    ## pairs that can't resolve are soft-skipped + logged; downstream restricts to the
    ## intersection of (family, leaf) tuples present in every simulator.

    def _find_subtree_root_name(tree, eval_leaves):
        '''Return the node name whose descendant leaf set equals eval_leaves, or None.'''
        target = frozenset(eval_leaves)
        for node in tree.traverse():
            if frozenset(L.name for L in node.get_leaves()) == target:
                return node.name
        return None

    print("Computing path saturation per (family, simulator, leaf) from matched branches")
    path_skip_log = []
    for family in families:
        real_tree = Tree(os.path.join(historian_tree_dir, family + '.txt'), format=1)
        real_msa = read_msa(os.path.join(historian_msa_dir, family + '.txt'))
        eval_leaf_set = set(n.name for n in real_tree.get_leaves())

        # Resolve each simulator's subtree-root by descendant-leafset equality, then
        # check that the resulting node name has a sequence in that simulator's MSA.
        sim_state = {}
        real_root = _find_subtree_root_name(real_tree, eval_leaf_set)
        if real_root and real_root in real_msa:
            sim_state['Real (Inferred)'] = (real_root, real_msa)
        else:
            path_skip_log.append((family, 'Real (Inferred)',
                                  f'no node with descendants==eval_leaf_set, or root sequence missing'))

        for model in model_dirs:
            sim_msa = read_msa(os.path.join(model_dirs[model]['msa'], family + '.txt'))
            if model in _PROTEVO_TREE_MODELS:
                sim_tree_for_root = read_tree(
                    os.path.join(model_dirs[model]['tree'], family + '.txt')
                ).to_ete3()
            else:
                sim_tree_for_root = Tree(
                    os.path.join(model_dirs[model]['tree'], family + '.full.treefile'),
                    format=1,
                )
            root_name = _find_subtree_root_name(sim_tree_for_root, eval_leaf_set)
            if root_name is None:
                path_skip_log.append((family, model, 'no node with descendants==eval_leaf_set'))
                continue
            if root_name not in sim_msa:
                path_skip_log.append((family, model, f'root {root_name!r} missing from MSA'))
                continue
            sim_state[model] = (root_name, sim_msa)

        # Take the intersection of matched leafsets across all simulators. match_branches
        # already restricts to eval_leaf_set, but the matched *keys* can still differ
        # (e.g. a unary root-stub branch present in the historian tree but collapsed by
        # AliSim). Summing over the intersection makes every arm sum exactly the same
        # branches — the strictest apples-to-apples we can get.
        per_model_matched = {m: matched_model_data[m].get(family) for m in model_dirs}
        if any(d is None for d in per_model_matched.values()):
            path_skip_log.append((family, '*', 'matched-branch dict missing for at least one simulator'))
            continue
        shared_leafsets = set.intersection(*[set(d.keys()) for d in per_model_matched.values()])
        if not shared_leafsets:
            path_skip_log.append((family, '*', 'no shared matched branches'))
            continue

        leaf_to_leafsets = {L: [] for L in eval_leaf_set}
        for lf in shared_leafsets:
            for L in lf:
                if L in leaf_to_leafsets:
                    leaf_to_leafsets[L].append(lf)

        # orig_mutations is identical across every model's matched dict by construction
        # (same original_branches feed match_branches), so reading from any one is fine.
        ref_matched = next(iter(per_model_matched.values()))

        if 'Real (Inferred)' in sim_state:
            root_name, msa = sim_state['Real (Inferred)']
            for L in eval_leaf_set:
                if L not in msa:
                    continue
                path_muts = sum(ref_matched[lf]['orig_mutations'] for lf in leaf_to_leafsets[L])
                hamming, sites = get_mutations_for_branch(msa[root_name], msa[L])
                path_rows.append([family, 'Real (Inferred)', L, int(path_muts), int(hamming), int(sites)])

        for model in model_dirs:
            if model not in sim_state:
                continue
            root_name, msa = sim_state[model]
            matched = per_model_matched[model]
            for L in eval_leaf_set:
                if L not in msa:
                    continue
                path_muts = sum(matched[lf]['sim_mutations'] for lf in leaf_to_leafsets[L])
                hamming, sites = get_mutations_for_branch(msa[root_name], msa[L])
                path_rows.append([family, model, L, int(path_muts), int(hamming), int(sites)])

    if path_skip_log:
        print(f"Path saturation: skipped {len(path_skip_log)} (family, simulator) entries; first 10:")
        for fam, sim, reason in path_skip_log[:10]:
            print(f"  {fam} / {sim}: {reason}")

    ## aggregate all information
    all_data = []

    print("Aggregating Information Across Families and Models")
    print(matched_model_data.keys())

    for f, matched_data in matched_model_data['PEINT'].items():
        for peint_branch, peint_branch_data in matched_data.items():
            #check if branch is in wag or lg
            lg_branch_data = matched_model_data['LG'][f].get(peint_branch)
            wag_branch_data = matched_model_data['WAG'][f].get(peint_branch)
            lg_s256_branch_data = matched_model_data['LG_S256'][f].get(peint_branch)
            # ESM-C is NOT required for a branch to be kept (keeps the established bars
            # unchanged); its row is added only where it has the matched branch.
            peint_esmc_branch_data = matched_model_data['PEINT_ESMC'].get(f, {}).get(peint_branch)

            if not lg_branch_data or not wag_branch_data or not lg_s256_branch_data:
                continue

            # Shared historian length for this branch (identical across models, reference only);
            # each model's own re-optimised length is used below for the per-branch rate.
            orig_bl = peint_branch_data['orig_branch_length']
            orig_muts = peint_branch_data['orig_mutations']
            orig_sites = peint_branch_data['orig_sites']

            if orig_sites == 0:
                continue

            branch_name = peint_branch_data['branch_name']
            orig_frac_muts = orig_muts / orig_sites
            orig_back = peint_branch_data['orig_back_mutations']
            orig_back_sites = peint_branch_data['orig_back_sites']

            all_data.append(
                [branch_name, orig_bl, orig_bl, orig_frac_muts, orig_muts, orig_sites,
                 orig_back, orig_back_sites, 'Real (Inferred)', f]
            )

            for model_name in matched_model_data:
                model_branch = eval(f"{model_name.lower()}_branch_data")
                if model_branch is None:  # ESM-C may lack this branch; others are guaranteed above
                    continue
                sim_muts = model_branch['sim_mutations']
                sim_sites = model_branch['sim_sites']
                sim_back = model_branch['sim_back_mutations']
                sim_back_sites = model_branch['sim_back_sites']
                sim_bl = model_branch['sim_branch_length']

                if sim_sites == 0:
                    break

                sim_frac_muts = sim_muts / sim_sites

                all_data.append(
                    [branch_name, sim_bl, orig_bl, sim_frac_muts, sim_muts, sim_sites,
                     sim_back, sim_back_sites, model_name, f]
                )

    data_columns = ['branch_name', 'branch_length', 'orig_branch_length', 'mutations',
                    'mutation_count', 'num_sites', 'back_mutations', 'back_num_sites',
                    'simulator', 'family']
    df = pd.DataFrame(all_data, columns = data_columns)

    df.to_csv(
        os.path.join(
            distances_dir,
            f"simulation_internal_analysis.csv"
        ),
        index=False
    )

    per_site_df = pd.DataFrame(
        per_site_rows,
        columns=['family', 'simulator', 'site', 'mutation_count', 'branch_count',
                 'back_mutation_count', 'triple_count',
                 'leaf_entropy', 'leaf_obs',
                 'empirical_leaf_entropy', 'profile_post_entropy', 'effective_profile_entropy'],
    )
    per_site_df.to_csv(
        os.path.join(distances_dir, 'per_site_diagnostics.csv'),
        index=False,
    )

    path_df = pd.DataFrame(
        path_rows,
        columns=['family', 'simulator', 'leaf', 'path_mutations', 'root_leaf_hamming', 'root_leaf_sites'],
    )
    path_df.to_csv(
        os.path.join(distances_dir, 'path_saturation.csv'),
        index=False,
    )
    render_all(df, path_df)


def _cli():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--replot', action='store_true',
                    help='Redraw the panels from the saved aggregation tables instead of '
                         'redoing the branch matching (seconds rather than ~30 minutes).')
    ap.add_argument('--distances-dir', default=None,
                    help='Where simulation_internal_analysis.csv / path_saturation.csv live '
                         '(default: cfg.LEAF_DISTANCES_DIR).')
    return ap.parse_args()


if __name__ == "__main__":
    _args = _cli()
    if _args.replot:
        replot_from_csv(_args.distances_dir)
    else:
        _run_full_pipeline()
