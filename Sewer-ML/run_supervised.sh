#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --job-name=supervised_101
#SBATCH --output=logs/supervised_%j.out
#SBATCH --error=logs/supervised_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
set -e

DATAROOT="../../Datasets/"
RESULTS_DIR="./results"
CHECKPOINT_DIR="./checkpoints_supervised"
LOG_DIR="./logs"
SCRIPT="supervised_sewerml.py"

mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Start time : $(date)"
echo "Script     : $SCRIPT"
echo "============================================"

python "$SCRIPT" \
    --dataroot          "$DATAROOT" \
    --scale_size        224 \
    --batch_size        32 \
    --test_batch_size   32 \
    --workers           8 \
    --epochs            50 \
    --lr                1e-4 \
    --weight_decay      1e-4 \
    --eval_freq         1 \
    --use_weighted_loss \
    --results_dir       "$RESULTS_DIR" \
    --experiment_name   "sewerml_supervised" \
    --model_save_path   "$CHECKPOINT_DIR" \

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
 