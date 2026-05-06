#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --job-name=RB_AL_10_03Q
#SBATCH --output=/nobackup/projects/bddur59/Code/Python_files/RB/logs/RB_AL_%j.out
#SBATCH --error=/nobackup/projects/bddur59/Code/Python_files/RB/logs/RB_AL_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

# Diagnostic — figure out where SLURM thinks we are
echo "===== DIAGNOSTIC ====="
echo "PWD at start    : $(pwd)"
echo "Script path \$0  : $0"
echo "dirname of \$0   : $(dirname "$0")"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "======================"

set -e

# Use ABSOLUTE paths — don't trust relative paths under SLURM
SCRIPT_DIR="/nobackup/projects/bddur59/Code/Python_files/RB"
cd "$SCRIPT_DIR"

DATAROOT="$SCRIPT_DIR/../../../Datasets/"
RESULTS_DIR="$SCRIPT_DIR/results_rb"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints_rb"
LOG_DIR="$SCRIPT_DIR/logs"
SCRIPT="$SCRIPT_DIR/rb_active_learning.py"

CLIPIQA_JSON="$SCRIPT_DIR/../IQA/clip_iqa_train_all.json"
CLIPIQA_THRESHOLD="0.3"
SECONDARY_STRATEGY="balanced"
POS_RATIO="0.5"

echo "After cd: $(pwd)"
echo "RESULTS_DIR: $RESULTS_DIR"
mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"
ls -ld "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

echo "============================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Start time : $(date)"
echo "============================================"

command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi unavailable"


python "$SCRIPT" \
    --dataroot            "$DATAROOT" \
    --scale_size          224 \
    --batch_size          32 \
    --workers             8 \
    --target_class        "RB" \
    --main_arch           "resnet101" \
    --al_cycles           10 \
    --al_budget           2000 \
    --candidate_ratio     0.02 \
    --cold_start_strategy "clipiqa" \
    --initial_percentage  0.1 \
    --secondary_strategy  "$SECONDARY_STRATEGY" \
    --pos_ratio           "$POS_RATIO" \
    --clipiqa_json_path   "$CLIPIQA_JSON" \
    --clipiqa_threshold   "$CLIPIQA_THRESHOLD" \
    --oracle_epochs       15 \
    --initial_epochs      10 \
    --cycle_epochs        10 \
    --lr                  1e-4 \
    --weight_decay        1e-4 \
    --policy_lr           1e-4 \
    --policy_hidden       256 \
    --policy_temp_start   1.0 \
    --policy_temp_end     0.5 \
    --entropy_beta        1e-3 \
    --seed                42 \
    --results_dir         "$RESULTS_DIR" \
    --experiment_name     "rb_al_clipiqa" \
    --dataset_type        "sewerml" \
    --gpu                 0



EXIT_CODE=$?
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
exit $EXIT_CODE
