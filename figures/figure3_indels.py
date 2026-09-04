import os
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib as mpl
from scipy.stats import pearsonr, spearmanr

from peint import caching as peint_caching
from peint.utils import read_msa

from paper.historian import (
    prepare_simulated_vs_real_historian,
    add_dummy_nodes,
    run_historian,
    remove_dummy_nodes_from_historian_output,
    get_all_evolutionary_counts_from_historian_output,
)
from paper.alignment import run_mafft, sanitize_fastas
import paper_config as cfg

if __name__ == "__main__":
    num_processes = 20

    protevo_caching.set_cache_dir("_cache_protevo")
    peint_caching.set_read_only(False)  # remove this line when training a new model
    peint_caching.set_log_level(9)

    FIG_OUT = str(cfg.FIGURES_DIR)
    os.makedirs(FIG_OUT, exist_ok=True)

    real_tree_dir = str(cfg.REAL_TREE_DIR)
    real_sequences_dir = str(cfg.REAL_MSA_DIR)

    simulated_sequences_dir = str(cfg.SIMULATED_MSA_DIR)
    simulated_tree_dir = str(cfg.TREE_DIR)
    simulated_root_seqs_dir = str(cfg.ROOT_SEQ_DIR)

    all_families = sorted([f.replace('.txt', '') for f in os.listdir(simulated_sequences_dir) if f.endswith('.txt')])
    print(f"Found {len(all_families)} families in the simulation output directory.")

    with open(str(cfg.HELDOUT_FAMILIES_FILE), 'r') as f:
        heldout_families = [line.strip() for line in f.readlines()]

    sim_families = [f for f in all_families if f in heldout_families]
    print(f"Using {len(sim_families)} families for the simulation analysis.")

    ############ SIMULATION ANALYSIS ############

    #1. Prepare the MSAs and trees - get the subtree away from the rerooted node

    sanitized_real_sequences_dir = sanitize_fastas(
        data_dir=real_sequences_dir,
        families=sim_families,
        num_processes=num_processes,
    )

    print("Preparing sequences and trees for historian...")
    prepared_sequences_dirs = prepare_simulated_vs_real_historian(
        simulated_tree_dir=simulated_tree_dir,
        simulated_sequences_dir=simulated_sequences_dir,
        root_sequences_dir=simulated_root_seqs_dir,
        real_tree_dir=real_tree_dir,
        real_sequences_dir=sanitized_real_sequences_dir['output_sequences_dir'],
        families=sim_families,
        num_processes=num_processes
    )
    print(f"Prepared sequences directory: {prepared_sequences_dirs}")

    #2. Add dummy nodes close to internal nodes for the simulated trees
    print("Adding dummy nodes to the sequences and trees...")
    dummy_node_dirs = add_dummy_nodes(
        sequences_dir=prepared_sequences_dirs['output_simulated_sequences_dir'],
        tree_dir=prepared_sequences_dirs['output_simulated_tree_dir'],
        families=sim_families,
        num_processes=num_processes,
        dummy_branch_length=0.005, #default is 0.005
    )

    #3. Use MAFFT to create a guide alignment for historian
    print("Running MAFFT to create guide trees for historian...")
    real_mafft_guide_dirs = run_mafft(
        data_dir=prepared_sequences_dirs['output_real_sequences_dir'],
        families=sim_families,
        num_processes=num_processes,
    )
    simulated_mafft_guide_dirs = run_mafft(
        data_dir=dummy_node_dirs['output_sequences_dir'],
        families=sim_families,
        num_processes=num_processes,
    )

    #4. Run historian on the prepared sequences and trees
    print("Running Historian to analyze the sequences and trees...")
    real_historian_output_dirs = run_historian(
        sequences_dir=real_mafft_guide_dirs['output_msa_dir'],
        tree_dir=prepared_sequences_dirs['output_real_tree_dir'],
        families=sim_families,
        num_processes=num_processes,
        historian_path='historian',
        extra_command_line_args=['-allspan', '-band', '40', '-refine']  # Use the same options as in the simulation analysis
    )
    simulated_historian_output_dirs = run_historian(
        sequences_dir=simulated_mafft_guide_dirs['output_msa_dir'],
        tree_dir=dummy_node_dirs['output_tree_dir'],
        families=sim_families,
        num_processes=num_processes,
        historian_path='historian',
        extra_command_line_args = ['-allspan', '-band', '40', '-refine']
    )
    print("Simulation analysis completed.")
    print(f"Output directories (simulated): {simulated_historian_output_dirs}")
    print(f"Output directories (real): {real_historian_output_dirs}")

    #5. Remove dummy nodes from the simulated historian output
    non_dummy_sequences_dir = remove_dummy_nodes_from_historian_output(
        sequences_dir=simulated_historian_output_dirs['output_sequences_dir'],
        families=sim_families,
        num_processes=num_processes,
    )

    #6. Count evolutionary events with the pure-Python counter (parent<->child over the
    #   recon tree), instead of `historian count`, which aborts on prepared trees whose
    #   root is unnamed. The Python counter skips missing/unnamed nodes gracefully.
    simulated_evolutionary_counts = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=non_dummy_sequences_dir['output_sequences_dir'],
        tree_dir=prepared_sequences_dirs['output_simulated_tree_dir'],
        families=sim_families,
        num_processes=num_processes,
    )
    real_evolutionary_counts = get_all_evolutionary_counts_from_historian_output(
        sequences_dir=real_historian_output_dirs['output_sequences_dir'],
        tree_dir=prepared_sequences_dirs['output_real_tree_dir'],
        families=sim_families,
        num_processes=num_processes,
    )

    ### PLOT
    print("Plotting the results...")

    def _indel_counts(events_dir, family):
        """Historian-style indel counts from the per-branch event CSV: ins/del = number of
        indel events (gap opens); insExt/delExt = extensions = sum(length - 1)."""
        path = os.path.join(events_dir, f"{family}.txt")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        counts = {'ins': 0, 'del': 0, 'insExt': 0, 'delExt': 0}
        for event_type, base in (('insertion', 'ins'), ('deletion', 'del')):
            rows = df[df['event_type'] == event_type]
            n = len(rows)
            counts[base] = int(n)
            counts[base + 'Ext'] = int(rows['length'].sum() - n) if n else 0
        return {'indel': counts}

    sim_counts = {}
    real_counts = {}
    for fam in sim_families:
        sc = _indel_counts(simulated_evolutionary_counts['output_events_dir'], fam)
        rc = _indel_counts(real_evolutionary_counts['output_events_dir'], fam)
        if sc is not None:
            sim_counts[f"{fam}.txt"] = sc
        if rc is not None:
            real_counts[f"{fam}.txt"] = rc

    sim_indels = []
    real_indels = []
    sim_indel_ext = []
    real_indel_ext = []
    for f in sim_counts:
        if f not in real_counts:
            print(f"Missing in real counts: {f}")
        else:
            fname = f.replace('.txt', '')
            if fname in heldout_families:
                sim_indels.append(sim_counts[f]['indel']['ins'] + sim_counts[f]['indel']['del'])
                real_indels.append(real_counts[f]['indel']['ins'] + real_counts[f]['indel']['del'])
                sim_indel_ext.append(sim_counts[f]['indel']['insExt'] + sim_counts[f]['indel']['delExt'])
                real_indel_ext.append(real_counts[f]['indel']['insExt'] + real_counts[f]['indel']['delExt'])
            

    sns.set_theme(style='white')
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.minor.left'] = True
    plt.rcParams['grid.linewidth'] = 0.5  # Reduced from 0.5
    plt.rcParams.update({
        'font.size': 8,        
        'axes.labelsize': 8,   
        'xtick.labelsize': 7,  
        'ytick.labelsize': 7, 
    })
    mpl.rcParams['pdf.fonttype'] = 42
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)


    sns.scatterplot(x=real_indels, y=sim_indels, ax=ax, linewidth=0.1, s=15, alpha=0.7, edgecolor='black', color='green')
    pearson = pearsonr(real_indels, sim_indels)[0]
    spearman = spearmanr(real_indels, sim_indels)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Events (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Events')
    ax.set_xlim(0, max((max(real_indels), max(sim_indels))) * 1.1)
    ax.set_ylim(0, max((max(real_indels), max(sim_indels))) * 1.1)
    ax.set_xticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=1000))
    ax.set_yticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=1000))
    ax.set_xticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=500), minor=True)
    ax.set_yticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=500), minor=True)

    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)


    sns.scatterplot(x=real_indels, y=sim_indels, ax=ax, linewidth=0.1, s=15, alpha=0.7, color='green')
    pearson = pearsonr(real_indels, sim_indels)[0]
    spearman = spearmanr(real_indels, sim_indels)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Events (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Events')
    ax.set_xlim(10, max((max(real_indels), max(sim_indels))) * 1.1)
    ax.set_ylim(10, max((max(real_indels), max(sim_indels))) * 1.1)

    ax.plot([10, max((max(real_indels), max(sim_indels))) * 1.1], [10, max((max(real_indels), max(sim_indels))) * 1.1], color='black', linestyle='--', linewidth=0.5)


    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels_logscale.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)


    max_indels = max((max(real_indels), max(sim_indels)))
    ax.hexbin(
        real_indels,
        sim_indels,
        gridsize=50,
        cmap='Blues',
        mincnt=1,
        extent = (0, max_indels * 1.1, 0, max_indels * 1.1),
        vmin=0,
        vmax=20,
        linewidths=0.
    )

    ax.plot([0, max_indels * 1.1], [0, max_indels * 1.1], color='black', linestyle='--', linewidth=0.5)

    pearson = pearsonr(real_indels, sim_indels)[0]
    spearman = spearmanr(real_indels, sim_indels)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Events (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Events')
    ax.set_xlim(0, max((max(real_indels), max(sim_indels))) * 1.1)
    ax.set_ylim(0, max((max(real_indels), max(sim_indels))) * 1.1)
    ax.set_xticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=1000))
    ax.set_yticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=1000))
    ax.set_xticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=500), minor=True)
    ax.set_yticks(np.arange(0, max((max(real_indels), max(sim_indels))) * 1.1, step=500), minor=True)

    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels_density.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)


    ax.set_xscale('log')
    ax.set_yscale('log')

    max_indels = max((max(real_indels), max(sim_indels)))

    # Option 1: Let hexbin handle the extent automatically
    hb = ax.hexbin(
        real_indels,
        sim_indels,
        gridsize=20,
        cmap='Blues',
        mincnt=1,
        xscale='log',
        yscale='log',
        vmin=1,
        vmax=20,
        linewidths=0.,
        extent = (np.log10(10), np.log10(max_indels * 1.1), 
              np.log10(10), np.log10(max_indels * 1.1))
    )

    ax.plot([10, max_indels * 1.1], [10, max_indels * 1.1], 
            color='black', linestyle='--', linewidth=0.5)

    pearson = pearsonr(real_indels, sim_indels)[0]
    spearman = spearmanr(real_indels, sim_indels)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Events (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Events')
    ax.set_xlim(10, max((max(real_indels), max(sim_indels))) * 1.1)
    ax.set_ylim(10, max((max(real_indels), max(sim_indels))) * 1.1)
    
    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels_density_logscale.pdf'), bbox_inches='tight')


    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)


    sns.scatterplot(x=real_indel_ext, y=sim_indel_ext, ax=ax, linewidth=0.1, s=15, alpha=0.5, color='green')
    pearson = pearsonr(real_indel_ext, sim_indel_ext)[0]
    spearman = spearmanr(real_indel_ext, sim_indel_ext)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Event Lengths (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Event Lengths')
    ax.set_xlim(0, max((max(real_indel_ext), max(sim_indel_ext))) * 1.1)
    ax.set_ylim(0, max((max(real_indel_ext), max(sim_indel_ext))) * 1.1)

    max_indel_val = max((max(real_indel_ext), max(sim_indel_ext)))

    step_size = max(1000, max_indel_val / 10)

    ax.plot([1, max_indel_val * 1.1], [1, max_indel_val * 1.1], color='black', linestyle='--', linewidth=0.5)

    ax.set_xticks(np.arange(0, max_indel_val * 1.1, step=step_size))
    ax.set_yticks(np.arange(0, max_indel_val * 1.1, step=step_size))
    ax.set_xticks(np.arange(0, max_indel_val * 1.1, step=step_size /2), minor=True)
    ax.set_yticks(np.arange(0, max_indel_val * 1.1, step=step_size/2), minor=True)

    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels_ext.pdf'), bbox_inches='tight')

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)


    sns.scatterplot(x=real_indel_ext, y=sim_indel_ext, ax=ax, linewidth=0.1, s=15, alpha=0.5, color='green')
    pearson = pearsonr(real_indel_ext, sim_indel_ext)[0]
    spearman = spearmanr(real_indel_ext, sim_indel_ext)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Event Lengths (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Event Lengths')
    ax.set_xlim(1, max((max(real_indel_ext), max(sim_indel_ext))) * 1.1)
    ax.set_ylim(1, max((max(real_indel_ext), max(sim_indel_ext))) * 1.1)

    max_indel_val = max((max(real_indel_ext), max(sim_indel_ext)))

    step_size = max(1000, max_indel_val / 10)

    ax.plot([1, max_indel_val * 1.1], [1, max_indel_val * 1.1], color='black', linestyle='--', linewidth=0.5)

    ax.set_xscale('log')
    ax.set_yscale('log')

    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels_ext_logscale.pdf'), bbox_inches='tight')

    #####EXTENSIONS DENSITY PLOT#####
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    ax.hexbin(
        real_indel_ext,
        sim_indel_ext,
        gridsize=50,
        cmap='Blues',
        mincnt=1,
        extent = (0, max_indel_val * 1.1, 0, max_indel_val * 1.1),
        vmin=0,
        vmax=50,
        linewidths=0.
    )

    pearson = pearsonr(real_indel_ext, sim_indel_ext)[0]
    spearman = spearmanr(real_indel_ext, sim_indel_ext)[0]
    ax.text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax.transAxes, fontsize=8, verticalalignment='top')
    ax.set_xlabel('Real Indel Event Lengths (inferred)')
    ax.set_ylabel('PEINT Simulated Indel Event Lengths')
    ax.set_xlim(0, max((max(real_indel_ext), max(sim_indel_ext))) * 1.1)
    ax.set_ylim(0, max((max(real_indel_ext), max(sim_indel_ext))) * 1.1)

    max_indel_val = max((max(real_indel_ext), max(sim_indel_ext)))

    step_size = max(1000, max_indel_val / 10)

    ax.set_xticks(np.arange(0, max_indel_val * 1.1, step=step_size))
    ax.set_yticks(np.arange(0, max_indel_val * 1.1, step=step_size))
    ax.set_xticks(np.arange(0, max_indel_val * 1.1, step=step_size /2), minor=True)
    ax.set_yticks(np.arange(0, max_indel_val * 1.1, step=step_size/2), minor=True)

    ax.tick_params(width = 0.5, length = 2, which = 'both')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_indels_ext_density.pdf'), bbox_inches='tight')

    ## COMPARE SEQUENCE LENGTHS WITH INDEL FREQUENCIES

    plotted_families = set(sim_counts.keys()).intersection(set(real_counts.keys()))

    sim_lengths_and_indels = []
    real_lengths_and_indels = []
    for f in plotted_families:
        sim_seqs = read_msa(os.path.join(simulated_sequences_dir, f))
        real_seqs = read_msa(os.path.join(real_sequences_dir, f))

        fam_sim_median_length = np.median([len(s) for s in sim_seqs.values()])
        fam_real_median_length = np.median([len(s) for s in real_seqs.values()])

        sim_indel_count = sim_counts[f]['indel']['ins'] + sim_counts[f]['indel']['del']
        real_indel_count = real_counts[f]['indel']['ins'] + real_counts[f]['indel']['del']

        sim_lengths_and_indels.append((fam_sim_median_length, sim_indel_count))
        real_lengths_and_indels.append((fam_real_median_length, real_indel_count))

    fig, ax = plt.subplots(1, 2, figsize=(5, 2))

    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.3)

    # plot sim data
    sns.scatterplot(
        x=[l[0] for l in sim_lengths_and_indels],
        y=[l[1] for l in sim_lengths_and_indels],
        ax=ax[0],
        linewidth=0.1,
        s=15,
        alpha=0.5,
        color='green'
    )
    pearson = pearsonr([l[0] for l in sim_lengths_and_indels],
                       [l[1] for l in sim_lengths_and_indels])[0]
    spearman = spearmanr([l[0] for l in sim_lengths_and_indels],
                         [l[1] for l in sim_lengths_and_indels])[0]
    ax[0].text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax[0].transAxes, fontsize=8, verticalalignment='top')
    ax[0].set_xlabel('PEINT Simulated Median Sequence Length')
    ax[0].set_ylabel('PEINT Simulated Indel Events')
    ax[0].set_xlim(0, max([l[0] for l in sim_lengths_and_indels]) * 1.1)
    ax[0].set_ylim(0, max([l[1] for l in sim_lengths_and_indels]) * 1.1)

    sns.scatterplot(
        x=[l[0] for l in real_lengths_and_indels],
        y=[l[1] for l in real_lengths_and_indels],
        ax=ax[1],
        linewidth=0.1,
        s=15,
        alpha=0.5,
        color='blue'
    )
    pearson = pearsonr([l[0] for l in real_lengths_and_indels],
                       [l[1] for l in real_lengths_and_indels])[0]
    spearman = spearmanr([l[0] for l in real_lengths_and_indels],
                         [l[1] for l in real_lengths_and_indels])[0]
    ax[1].text(0.05, 0.95, f'Pearson: {pearson:.2f}\nSpearman: {spearman:.2f}',
            transform=ax[1].transAxes, fontsize=8, verticalalignment='top')
    ax[1].set_xlabel('Real Median Sequence Length')
    ax[1].set_ylabel('Real Indel Events')
    ax[1].set_xlim(0, max([l[0] for l in real_lengths_and_indels]) * 1.1)
    ax[1].set_ylim(0, max([l[1] for l in real_lengths_and_indels]) * 1.1)

    for i in range(2):

        ax[i].tick_params(width = 0.5, length = 2, which = 'both')
        sns.despine(ax=ax[i])
        for spine in ax[i].spines.values():
            spine.set_linewidth(0.5)

    fig.savefig(os.path.join(FIG_OUT, 'sim_vs_real_sequence_lengths_indels.pdf'), bbox_inches='tight')


    #Get evolutionary counts (manually)
    print("Getting all evolutionary event counts from historian output...")

    simulated_evolutionary_counts = get_all_evolutionary_counts_from_historian_output(
        sequences_dir = non_dummy_sequences_dir['output_sequences_dir'],
        tree_dir = prepared_sequences_dirs['output_simulated_tree_dir'],
        families = sim_families,
        num_processes = num_processes,
    )

    print(f"Simulated evolutionary counts: {simulated_evolutionary_counts}")
    real_evolutionary_counts = get_all_evolutionary_counts_from_historian_output(
        sequences_dir = real_historian_output_dirs['output_sequences_dir'],
        tree_dir = prepared_sequences_dirs['output_real_tree_dir'],
        families = sim_families,
        num_processes = num_processes,
    )
    print(f"Real evolutionary counts: {real_evolutionary_counts}")

    #plot indel length distribution
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    aggregated_real = []
    aggregated_sim = []
    for f in sim_families:
        real_file = os.path.join(
            real_evolutionary_counts['output_events_dir'], f"{f}.txt"
        )

        sim_file = os.path.join(
            simulated_evolutionary_counts['output_events_dir'], f"{f}.txt"
        )

        real_df = pd.read_csv(real_file, sep=',')
        sim_df = pd.read_csv(sim_file, sep=',')

        real_indels = real_df[real_df['event_type'].isin(['insertion', 'deletion'])]
        sim_indels = sim_df[sim_df['event_type'].isin(['insertion', 'deletion'])]

        aggregated_real.extend(real_indels.length.values)
        aggregated_sim.extend(sim_indels.length.values)
    
    real_events = pd.DataFrame(aggregated_real, columns=['length'])
    sim_events = pd.DataFrame(aggregated_sim, columns=['length'])

    real_dist = real_events['length'].value_counts().sort_index()
    sim_dist = sim_events['length'].value_counts().sort_index()
    all_lengths = range(1,50)
    total_real = len(real_events)
    total_sim = len(sim_events)

    combined_data = []
    for length in all_lengths:
        real_count = real_dist.get(length, 0)
        sim_count = sim_dist.get(length, 0)
        combined_data.append({
            'length': length,
            'count': real_count,
            'density': real_count / total_real if total_real > 0 else 0,
            'data_type': 'Real (inferred)',
        })

        combined_data.append({
            'length': length,
            'count': sim_count,
            'density': sim_count / total_sim if total_sim > 0 else 0,
            'data_type': 'PEINT Simulated',
        })

    combined_df = pd.DataFrame(combined_data)

    sns.barplot(
        x='length',
        y='density',
        hue='data_type',
        data=combined_df,
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
    )

    ax.set_xlim(-1, 20.5)

    ax.set_xlabel('Indel Length')
    ax.set_ylabel('Density')
    ax.set_title('Indel Length Distribution')
    ax.set_xticks(np.arange(0, 21, step = 3))
    ax.set_xticks(np.arange(0, 21, step = 1), labels=[], minor=True)
    ax.set_yticks(np.arange(0, 0.6, step = 0.1))
    ax.tick_params(width = 0.5, length = 2, which = 'major')
    ax.tick_params(width = 0.5, length = 1, which = 'minor')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.legend(loc='upper right', fontsize=7, title='Data Type', title_fontsize='8', frameon=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.1)

    fig.savefig(os.path.join(FIG_OUT, 'figure3_indel_length_distribution.pdf'), bbox_inches='tight')

    real_expanded = []
    sim_expanded = []

    real_lengths = real_events['length'].value_counts().sort_index()
    sim_lengths = sim_events['length'].value_counts().sort_index()
    all_lengths = range(1,50)
    real_counts = [real_lengths.get(length, 0) for length in all_lengths]
    sim_counts = [sim_lengths.get(length, 0) for length in all_lengths]


    real_expanded = []
    sim_expanded = []
    for length, count in zip(all_lengths, real_counts):
        real_expanded.extend([length] * count)
    for length, count in zip(all_lengths, sim_counts):
        sim_expanded.extend([length] * count)

    real_sorted = np.sort(real_expanded)
    sim_sorted = np.sort(sim_expanded)

    real_cdf = np.arange(1, len(real_sorted) + 1) / len(real_sorted)
    sim_cdf = np.arange(1, len(sim_sorted) + 1) / len(sim_sorted)

    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.05)

    ax.plot(real_sorted, real_cdf, label='Real (inferred)', color='purple', linewidth=1.5)
    ax.plot(sim_sorted, sim_cdf, label='PEINT Simulated', color ='green', linewidth=1.5)

    ax.set_xlabel('Indel Length')
    ax.set_ylabel('Cumulative Density')
    ax.set_title('Indel Length CDF')
    ax.set_xlim(0, 50)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(np.arange(0, 50, step = 10))
    ax.set_xticks(np.arange(0, 50, step = 5), labels=[], minor=True)
    ax.set_yticks(np.arange(0, 1.1, step = 0.1))
    ax.tick_params(width = 0.5, length = 2, which = 'major')
    ax.tick_params(width = 0.5, length = 1, which = 'minor')
    sns.despine(ax=ax)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)
    ax.legend(loc='lower right', fontsize=7, title='Condition', title_fontsize='8', frameon=False)
    ax.grid(True, which='both', linestyle='--', linewidth=0.1)

    fig.savefig(os.path.join(FIG_OUT, 'figure3_indel_length_cdf.pdf'), bbox_inches='tight')
