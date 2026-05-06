#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --job-name=RB_sup_subset
#SBATCH --output=/nobackup/projects/bddur59/Code/Python_files/RB/logs/RB_sup_full_%j.out
#SBATCH --error=/nobackup/projects/bddur59/Code/Python_files/RB/logs/RB_sup_full_%j.err

#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
# ==============================================================
# Sewer-ML Binary RB - Supervised on matched random subset
# ==============================================================
# This is the "no-AL" baseline. The train_fraction MUST match the
# total label budget of the AL run:
#
#   AL total labels = initial_percentage + al_cycles * al_budget
#                   = (0.01 * N_train) + (10 * 2000)
#
# If N_train ~= 1,040,000 then AL total = 10,400 + 20,000 = ~30,400
# -> train_fraction = 30400 / 1040000 = 0.0292  (~0.03)
#
# Adjust TRAIN_FRACTION below to match YOUR AL run exactly.
# ==============================================================
set -e

cd "$(dirname "$0")"
# ------ Paths ------
DATAROOT="../../../Datasets/"
RESULTS_DIR="./results_rb"
CHECKPOINT_DIR="./checkpoints_rb"
LOG_DIR="./logs"
SCRIPT="rb_supervised.py"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

# ------ Subset configuration ------
TRAIN_FRACTION="0.03"        # match AL total-label budget (see note above)
POS_RATIO="0.5"              # target positive fraction when stratified
USE_STRATIFIED="true"        # "true" to force balanced pos/neg, "false" for pure random

# ------ Print job info ------
echo "============================================"
echo "Job ID         : $SLURM_JOB_ID"
echo "Node           : $SLURMD_NODENAME"
echo "Start time     : $(date)"
echo "Script         : $SCRIPT"
echo "Mode           : Supervised on random subset"
echo "Train fraction : $TRAIN_FRACTION"
echo "Stratified     : $USE_STRATIFIED (pos_ratio=$POS_RATIO)"
echo "============================================"
# ------ Print GPU info ------
nvidia-smi
# ------ Build stratified flag ------
STRAT_FLAG=""
EXP_NAME="rb_random_subset"
if [ "$USE_STRATIFIED" = "true" ]; then
    STRAT_FLAG="--stratified"
    EXP_NAME="rb_random_subset_balanced"
fi
# ------ Run ------
python "$SCRIPT" \
    --dataroot          "$DATAROOT" \
    --scale_size        224 \
    --batch_size        64 \
    --workers           8 \
    --target_class      "RB" \
    --train_fraction    "$TRAIN_FRACTION" \
    $STRAT_FLAG \
    --pos_ratio         "$POS_RATIO" \
    --arch              "resnet50" \
    --epochs            30 \
    --lr                1e-4 \
    --weight_decay      1e-4 \
    --seed              42 \
    --results_dir       "$RESULTS_DIR" \
    --experiment_name   "$EXP_NAME" \
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
