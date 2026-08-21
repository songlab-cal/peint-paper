#!/bin/bash
#SBATCH --job-name=esmc_bench
#SBATCH --partition=yss
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc/slurm-%x-%j.out

# One shard of the ESM-C benchmark: sims + alignment + WAG/LG classical for the families in
# $1 (a JSON with {"families":[...]}). Two shards run concurrently on two GPUs into the same
# out_path (families are disjoint -> per-family outputs never collide). Conservation is NOT
# scored here; the aggregate JSD is computed in one final pass over all 545 families, because
# a per-shard aggregate would only see its own half.
FAMILIES="$1"

set -x
PY=/scratch/users/akoehl/conda/envs/peint-esmc/bin/python
export HF_HOME=/scratch/users/akoehl/hf_cache
export PYTHONPATH=/scratch/users/akoehl/peint:/scratch/users/akoehl/peint-paper

SIM_ROOT=/scratch/users/akoehl/old/protein-evolution/local_data/simulation/final_simulation_512_leaves/ratio_0-1_nucleus_1-0
CKPT=/scratch/users/yufan.cao/protevo_ablations/esmc/20260729-5e5d20h960d-esmc-14498fams-esmc/epoch=4-step=60000.ckpt
OUT=/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc

cd /scratch/users/akoehl/peint-paper

$PY -m benchmarks.generate_all_results \
  --families_path "$FAMILIES" \
  --real_sequences_dir "$SIM_ROOT/empirical_msas" \
  --tree_dir "$SIM_ROOT/trees_newick" \
  --root_sequences_dir "$SIM_ROOT/root_sequences" \
  --peint_checkpoint_path "$CKPT" \
  --out_path "$OUT" \
  --alisim_models WAG LG \
  --num_processes 1 \
  --peint_max_batch_size 64 \
  --peint_family_bucket_size 32
