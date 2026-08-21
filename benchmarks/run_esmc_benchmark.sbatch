#!/bin/bash
#SBATCH --job-name=esmc_bench
#SBATCH --partition=yss
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc/slurm-%j.out

# ESM-C PEINT benchmark: simulate the held-out families with the ESM-C 60k checkpoint,
# align, simulate WAG/LG classical baselines in that frame, and score conservation (JSD).
# Reuses nothing from the ESM2 results (separate out_path); sims are length-bucketed and run
# single-process on one GPU (num_processes=1, per the GPU-multiprocessing caveat).

set -x
PY=/scratch/users/akoehl/conda/envs/peint-esmc/bin/python
export HF_HOME=/scratch/users/akoehl/hf_cache
export PYTHONPATH=/scratch/users/akoehl/peint:/scratch/users/akoehl/peint-paper

SIM_ROOT=/scratch/users/akoehl/old/protein-evolution/local_data/simulation/final_simulation_512_leaves/ratio_0-1_nucleus_1-0
CKPT=/scratch/users/yufan.cao/protevo_ablations/esmc/20260729-5e5d20h960d-esmc-14498fams-esmc/epoch=4-step=60000.ckpt
OUT=/scratch/users/akoehl/protein-evolution/local_data/results_revision2_esmc

cd /scratch/users/akoehl/peint-paper

$PY -m benchmarks.generate_all_results \
  --families_path /scratch/users/akoehl/protein-evolution/local_data/final_sim_held_out_family.json \
  --real_sequences_dir "$SIM_ROOT/empirical_msas" \
  --tree_dir "$SIM_ROOT/trees_newick" \
  --root_sequences_dir "$SIM_ROOT/root_sequences" \
  --peint_checkpoint_path "$CKPT" \
  --out_path "$OUT" \
  --alisim_models WAG LG \
  --num_processes 1 \
  --peint_max_batch_size 64 \
  --peint_family_bucket_size 32 \
  --include_conservation
