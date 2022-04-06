#!/bin/bash

#SBATCH --job-name=orb
#SBATCH --output=%x.out
#SBATCH --error=%x.err
#SBATCH --time=10-00:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=10240
#SBATCH --partition=nodes
#SBATCH --gres=gpu:a100:0
#SBATCH --chdir=/cluster/raid/home/cyril.vallez/Project

# Verify working directory
echo $(pwd)

# Print gpu configuration for this job
#nvidia-smi

# Verify gpu allocation (should be 1 GPU)
echo "Indices of visible GPU(s) before job : $CUDA_VISIBLE_DEVICES"

# Initialize the shell to use local conda
eval "$(conda shell.bash hook)"

# Activate (local) env
conda activate base

python3 main.py $1
