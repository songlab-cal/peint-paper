#!/bin/bash
#SBATCH --job-name=heldout_sim
#SBATCH --partition=epurdom
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/users/akoehl/peint-paper/local_data/derived/heldout_similarity/slurm-%x-%j.out

# Closest training match for every held-out sequence (see benchmarks/heldout_train_similarity.py).
# All three stages are resumable: each skips work whose output already exists, so a requeued or
# extended job picks up where it stopped. `report` is cheap and is run afterwards from the repo root.

set -euo pipefail

PY=/scratch/users/akoehl/conda/envs/peint-esmc/bin/python
export PEINT_PAPER_PEINT_REPO=/scratch/users/akoehl/peint
export PYTHONPATH=/scratch/users/akoehl/peint:/scratch/users/akoehl/peint-paper
export PATH=/accounts/postdoc/akoehl/Software:$PATH

cd /scratch/users/akoehl/peint-paper
mkdir -p local_data/derived/heldout_similarity

THREADS=${SLURM_CPUS_PER_TASK:-64}
BENCH="$PY -m benchmarks.heldout_train_similarity"

echo "=== fasta ==="
$BENCH fasta --workers 32

echo "=== makedb ==="
$BENCH makedb --threads "$THREADS"

# -b6 -c1 trades memory for speed: roughly 36 GB resident against the 320 G requested, and a
# single index chunk means one pass over the database per query block.
echo "=== blastp ==="
$BENCH blastp --threads "$THREADS" --sensitivity very-sensitive --block 6 --index-chunks 1

echo "=== done; now run: $BENCH report ==="
