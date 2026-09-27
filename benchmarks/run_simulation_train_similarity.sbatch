#!/bin/bash
#SBATCH --job-name=sim_sim
#SBATCH --partition=epurdom
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH --time=1-00:00:00
#SBATCH --output=/scratch/users/akoehl/peint-paper/local_data/derived/heldout_similarity/slurm-%x-%j.out

# Searches the simulation's own sequences -- roots and generated leaves -- against the training
# database built by run_heldout_train_similarity.sbatch. Reuses that database, so that job must
# have completed first. Resumable: each query set skips if its hit table already exists.

set -euo pipefail
PY=/scratch/users/akoehl/conda/envs/peint-esmc/bin/python
export PEINT_PAPER_PEINT_REPO=/scratch/users/akoehl/peint
export PYTHONPATH=/scratch/users/akoehl/peint:/scratch/users/akoehl/peint-paper
export PATH=/accounts/postdoc/akoehl/Software:$PATH
cd /scratch/users/akoehl/peint-paper

$PY -m benchmarks.simulation_train_similarity fasta
$PY -m benchmarks.simulation_train_similarity blastp --threads "${SLURM_CPUS_PER_TASK:-64}"
echo "=== done; now run: $PY -m benchmarks.simulation_train_similarity report ==="
