#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:pascal:1

#SBATCH -p gpu-small
#SBATCH --qos=debug

#SBATCH --job-name=eval

#SBATCH -e stderr-filename
#SBATCH -o stdout-filename

#SBATCH --mem=10g
# SBATCH -t 01:00:00

module load cuda/8.0-cudnn6
module load opencv/3.4-py3

nvidia-smi

python3 evaluate.py
