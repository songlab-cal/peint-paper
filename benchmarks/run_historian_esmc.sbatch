#!/bin/bash
#SBATCH --job-name=hist_esmc
#SBATCH --partition=epurdom
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc/slurm-%x-%j.out

# Historian indel-event reconstruction on the ESM-C PEINT progressive sims (revision 2).
# Progressive-only, real reused from revision 1, eval subtrees recycled; historian recon
# runs -allspan -band 40 (no -refine). Heavy CPU/mem -> epurdom. Every step is cached, so
# it resumes from wherever it stopped.
set -x
PY=/scratch/users/akoehl/conda/envs/protevo-env/bin/python
export PYTHONPATH=/scratch/users/akoehl/peint:/scratch/users/akoehl/peint-paper
export PATH=/accounts/postdoc/akoehl/Software/mafft_bin:$PATH

cd /scratch/users/akoehl/peint-paper
$PY -u -m benchmarks.historian_esmc_progressive --num_processes 32 --band 40
