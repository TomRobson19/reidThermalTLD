#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:pascal:1
#SBATCH -p short
#SBATCH --job-name=transfer

#SBATCH -e stderr-transfer
#SBATCH -o stdout-transfer

#SBATCH --mem=8g
#SBATCH -t 24:00:00

module load cuda/8.0-cudnn6
module load opencv/3.4-py3

python3 transfer.py
