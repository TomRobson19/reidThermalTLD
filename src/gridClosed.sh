#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:pascal:1
#SBATCH -p long
#SBATCH --qos=long-high-prio
#SBATCH --job-name=gridClosed

#SBATCH -e stderr-grid
#SBATCH -o stdout-grid

#SBATCH --mem=8g

module load cuda/8.0-cudnn6
module load opencv/3.4-py3

python3 closedGrid.py
