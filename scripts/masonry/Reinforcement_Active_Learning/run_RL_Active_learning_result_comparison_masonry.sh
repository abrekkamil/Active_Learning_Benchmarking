#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH -p res-gpu-small
#SBATCH --job-name=RL_AL_masonry
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=2-00:00:00

source /etc/profile


python RL_Active_learning_result_comparison_masonry.py