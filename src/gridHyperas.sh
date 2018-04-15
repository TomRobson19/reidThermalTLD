#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p gpu-large

#SBATCH --qos=long-high-prio
#SBATCH --job-name=gridHyp 

# SBATCH -t 48:00:00
# SBATCH --mem=28g

module load cuda/8.0-cudnn6
module load opencv/3.4-py3

nvidia-smi

python3 newGrid.py
