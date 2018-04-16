#!/bin/bash
#SBATCH -N 1
#SBATCH -c 2
#SBATCH --gres=gpu:pascal:1

#SBATCH -p gpu-small
#SBATCH --qos=debug

#SBATCH --job-name=eval

#SBATCH --mem=12g
# SBATCH -t 01:00:00

module load cuda/8.0-cudnn6
module load opencv/3.4-py3

nvidia-smi

python3 evaluate.py
