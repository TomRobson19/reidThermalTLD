#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:pascal:1
#SBATCH -p gpu-large

#SBATCH --qos=long-high-prio
#SBATCH --job-name=gridHyp 

#SBATCH --mem=28g
#SBATCH -t 04-00:00:00

module load cuda/8.0-cudnn6
module load opencv/3.4-py3

nvidia-smi

python3 newGrid.py
