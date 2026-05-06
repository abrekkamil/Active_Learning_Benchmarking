#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --job-name=FS_AL
#SBATCH --output=logs/FS_AL_%j.out
#SBATCH --error=logs/FS_AL_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
# ==============================================================
# Sewer-ML Binary FS - REINFORCE Active Learning
# ==============================================================
# Total label budget:
#   = (initial_percentage * N_train) + al_cycles * al_budget
#   = (0.01 * ~1.04M)         + 10 * 2000
#   = ~10,400 + 20,000 = ~30,400 labels
#
# Note on FS: positive rate is ~0.6% (vs ~18% for RB), so we use
# pos_ratio=0.3 in the cold start to avoid burning through too
# many of the rare FS positives in one shot.
# ==============================================================
set -e
cd "$(dirname "$0")"

# ------ Paths ------
DATAROOT="../../../Datasets/"
RESULTS_DIR="./results_rb"
CHECKPOINT_DIR="./checkpoints_rb"
LOG_DIR="./logs"
SCRIPT="rb_active_learning.py"
# ------ Cold start ------
CLIPIQA_JSON="../IQA/clip_iqa_train_all.json"
CLIPIQA_THRESHOLD="0.5"
SECONDARY_STRATEGY="balanced"
POS_RATIO="0.3"
# ------ Create output directories ------
mkdir -p "$RESULTS_DIR"
mkdir -p "$CHECKPOINT_DIR"
mkdir -p "$LOG_DIR"
# ------ Print job info ------
echo "============================================"
echo "Job ID       : $SLURM_JOB_ID"
echo "Node         : $SLURMD_NODENAME"
echo "Start time   : $(date)"
echo "Script       : $SCRIPT"
echo "Target class : FS"
echo "Mode         : REINFORCE AL on binary FS"
echo "Cold start   : clipiqa + $SECONDARY_STRATEGY (pos_ratio=$POS_RATIO)"
echo "============================================"
# ------ Print GPU info ------
nvidia-smi
# ------ Run ------
python "$SCRIPT" \
    --dataroot          "$DATAROOT" \
    --scale_size        224 \
    --batch_size        32 \
    --workers           8 \
    --target_class      "FS" \
    \
    --main_arch         "resnet50" \
    --al_cycles         10 \
    --al_budget         2000 \
    --candidate_ratio   0.02 \
    \
    --cold_start_strategy "clipiqa" \
    --initial_percentage  0.01 \
    --secondary_strategy  "$SECONDARY_STRATEGY" \
    --pos_ratio           "$POS_RATIO" \
    --clipiqa_json_path   "$CLIPIQA_JSON" \
    --clipiqa_threshold   "$CLIPIQA_THRESHOLD" \
    \
    --oracle_epochs     15 \
    --initial_epochs    10 \
    --cycle_epochs      10 \
    --lr                1e-4 \
    --weight_decay      1e-4 \
    \
    --policy_lr         1e-4 \
    --policy_hidden     256 \
    --policy_temp_start 1.0 \
    --policy_temp_end   0.5 \
    --entropy_beta      1e-3 \
    \
    --seed              42 \
    --results_dir       "$RESULTS_DIR" \
    --experiment_name   "fs_al_clipiqa" \
    --dataset_type      "sewerml" \
    --gpu               0
EXIT_CODE=$?
echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
echo "Memory usage:"
sstat -j $SLURM_JOB_ID.batch --format=MaxRSS
echo "============================================"
exit $EXIT_CODE