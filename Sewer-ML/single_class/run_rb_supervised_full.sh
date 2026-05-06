#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --job-name=RB_sup_full
#SBATCH --output=/nobackup/projects/bddur59/Code/Python_files/RB/logs/RB_sup_full_%j.out
#SBATCH --error=/nobackup/projects/bddur59/Code/Python_files/RB/logs/RB_sup_full_%j.err
#SBATCH --time=2-00:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu

set -e

# Hardcoded absolute path — don't trust relative paths under SLURM
SCRIPT_DIR="/nobackup/projects/bddur59/Code/Python_files/RB"
cd "$SCRIPT_DIR"

DATAROOT="$SCRIPT_DIR/../../../Datasets/"
RESULTS_DIR="$SCRIPT_DIR/results_rb"
CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints_rb"
LOG_DIR="$SCRIPT_DIR/logs"
SCRIPT="$SCRIPT_DIR/rb_supervised.py"

mkdir -p "$RESULTS_DIR" "$CHECKPOINT_DIR" "$LOG_DIR"

echo "============================================"
echo "Job ID     : ${SLURM_JOB_ID:-local}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "Start time : $(date)"
echo "Script     : $SCRIPT"
echo "Mode       : Supervised ceiling (full data)"
echo "============================================"

command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi unavailable"

python "$SCRIPT" \
    --dataroot          "$DATAROOT" \
    --scale_size        224 \
    --batch_size        64 \
    --workers           8 \
    --target_class      "RB" \
    --train_fraction    1.0 \
    --arch              "resnet50" \
    --epochs            30 \
    --lr                1e-4 \
    --weight_decay      1e-4 \
    --seed              42 \
    --results_dir       "$RESULTS_DIR" \
    --experiment_name   "rb_full_ceiling" \
    --save_model \
    --model_save_path   "$CHECKPOINT_DIR" \
    --gpu               0

EXIT_CODE=$?
echo "============================================"
echo "End time   : $(date)"
echo "Exit code  : $EXIT_CODE"
[ -n "$SLURM_JOB_ID" ] && sstat -j "${SLURM_JOB_ID}.batch" --format=MaxRSS || true
echo "============================================"
exit $EXIT_CODE