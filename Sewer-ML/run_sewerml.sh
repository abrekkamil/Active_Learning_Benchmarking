#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --job-name=RAL_101_random
#SBATCH --output=logs/RAL_101_%j.out
#SBATCH --error=logs/RAL_101_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
# ==============================================================
# Sewer-ML RL Active Learning Job
# ==============================================================

set -e

# ------ Paths ------
DATAROOT="../../Datasets/"
RESULTS_DIR="./results"
CHECKPOINT_DIR="./checkpoints"
LOG_DIR="./logs"
SCRIPT="RAL_sewerml.py"

# ------ Create output directories ------
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

# ------ Print job info ------
echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Script     : $SCRIPT"
echo "============================================"

# ------ Print GPU info ------
nvidia-smi

# ------ Run ------

CLIPIQA_JSON="IQA/clip_iqa_train_all.json"   # adjust path if needed
CLIPIQA_THRESHOLD="0.5"
SECONDARY_STRATEGY="stratified"         # strategy applied after IQA filter


python "$SCRIPT" \
    --dataroot          "$DATAROOT" \
    --scale_size        224 \
    --batch_size        32 \
    --test_batch_size   32 \
    --workers           8 \
    --train_known_labels 1.0 \
    --test_known_labels  1.0 \
    \
    --cold_start_strategy "clipiqa" \
    --al_cycles         10 \
    --al_budget         2000 \
    --initial_percentage 0.01 \
    --candidate_ratio   0.02 \

    --clipiqa_json_path     "IQA/clip_iqa_train_all.json" \
    --clipiqa_threshold     "$CLIPIQA_THRESHOLD" \
    --secondary_strategy    "$SECONDARY_STRATEGY" \
    \
    --oracle_epochs     30 \
    --initial_epochs    10 \
    --cycle_epochs      10 \
    --lr                1e-4 \
    \
    --policy_lr         1e-4 \
    --policy_hidden     256 \
    --policy_temp_start 1.0 \
    --policy_temp_end   0.5 \
    --entropy_beta      1e-3 \
    \
    --results_dir       "$RESULTS_DIR" \
    --experiment_name   "sewerml_rl" \
    --dataset_type          "sewerml" \
    --query_strategy        "reinforce" \
    \
    --save_model \
    --model_save_path   "$CHECKPOINT_DIR" \
    --gpu               0

EXIT_CODE=$?

echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
echo "Memory usage:"
sstat -j $SLURM_JOB_ID.batch --format=MaxRSS
echo "============================================"

exit $EXIT_CODE