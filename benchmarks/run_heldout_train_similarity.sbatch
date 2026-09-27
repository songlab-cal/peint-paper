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

# --hit-membuf holds DIAMOND's intermediate seed hits in RAM. They are the dominant cost here and
# have nothing to do with the output size: a first attempt spilled 158 GB of them to scratch and
# died on the 500 G quota, against a 3 GB database and a final table of a few hundred MB. RAM is
# the resource we actually have, so we spend that instead.
#
# Block size stays at the DIAMOND default rather than the 6 the first attempt used. At -b6 -c1 that
# run already peaked at 191 GB resident with the hits going to disk; the buffer has to fit beside
# that, so a smaller reference block is what keeps the total inside the 320 G requested.
echo "=== blastp ==="
$BENCH blastp --threads "$THREADS" --sensitivity very-sensitive --block 2 --index-chunks 1

echo "=== done; now run: $BENCH report ==="
