#!/bin/bash
#SBATCH --account=bddur59
#SBATCH --partition=infer
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00

set -euo pipefail

CONFIG="${1:?Usage: run_domain_adaptation_joint.sh CONFIG_PATH [--smoke-test]}"
MODE="${2:-}"

REPO=/nobackup/projects/bddur59/Code/Active_Learning_Benchmarking
CONDA_SH=/nobackup/projects/bddur59/kaltinay/ppc64le/miniconda/etc/profile.d/conda.sh

source "$CONDA_SH"
conda activate PyTorch_for_AL

unset LD_PRELOAD || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

cd "$REPO"

if [[ ! -f "$CONFIG" ]]; then
    echo "ERROR: configuration not found: $CONFIG" >&2
    exit 1
fi

echo "Job ID: ${SLURM_JOB_ID:-interactive}"
echo "Node: $(hostname)"
echo "Configuration: $CONFIG"
echo "Mode: ${MODE:-full}"
echo "Started: $(date)"

python -m src.domain_adaptation_joint --config "$CONFIG" $MODE

echo "Finished: $(date)"
