#!/bin/bash

#SBATCH --job-name=DA_SegFormer
#SBATCH --account=bddur59
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/domain_adaptation_%j.out
#SBATCH --error=logs/domain_adaptation_%j.err

set -euo pipefail

echo "============================================================"
echo "Standalone SegFormer domain-adaptation experiment"
echo "Job ID: ${SLURM_JOB_ID:-not-set}"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "Submission directory: ${SLURM_SUBMIT_DIR:-not-set}"
echo "============================================================"

CONDA_SH="/nobackup/projects/bddur59/kaltinay/ppc64le/miniconda/etc/profile.d/conda.sh"

if [[ ! -f "$CONDA_SH" ]]; then
    echo "ERROR: Conda initialization file not found:"
    echo "$CONDA_SH"
    exit 1
fi

source "$CONDA_SH"
conda activate PyTorch_for_AL
unset LD_PRELOAD || true

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_MAX_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Submit the job from the repository root.
if [[ -z "${SLURM_SUBMIT_DIR:-}" ]]; then
    echo "ERROR: SLURM_SUBMIT_DIR is not defined."
    exit 1
fi

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f "src/domain_adaptation.py" ]]; then
    echo "ERROR: src/domain_adaptation.py was not found."
    echo "Submit this script from the repository root."
    exit 1
fi

mkdir -p results/domain_adaptation

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

MODE="${1:-full}"

case "$MODE" in
    smoke)
        EXTRA_ARGS=(--smoke-test)
        ;;
    full)
        EXTRA_ARGS=()
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'."
        echo "Usage:"
        echo "  sbatch scripts/run_domain_adaptation.sh smoke"
        echo "  sbatch scripts/run_domain_adaptation.sh full"
        exit 2
        ;;
esac

echo
echo "Mode: $MODE"
echo "Python: $(which python)"
python --version

echo
echo "PyTorch and CUDA information:"
python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("CUDA device:", torch.cuda.get_device_name(0))
PY

echo
nvidia-smi || true

echo
echo "Launching experiment..."

python -m src.domain_adaptation \
    --config config/domain_adaptation_segformer.yaml \
    "${EXTRA_ARGS[@]}"

echo
echo "============================================================"
echo "Completed: $(date)"
echo "============================================================"
