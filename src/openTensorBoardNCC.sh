#!/bin/bash
#
# Open Tensorboard in output directory
#
module load cuda/8.0-cudnn6

python3 /home/hzwr87/.local/lib/python3.5/site-packages/tensorboard/main.py --logdir=output/ --port=6007