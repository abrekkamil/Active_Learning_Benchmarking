#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:turing:1
#SBATCH -p res-gpu-small
#SBATCH --job-name=AL_masonry_pool
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00

source /etc/profile

python Active_learning_result_comparison_masonry.py