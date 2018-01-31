#!/bin/bash
#
# Open Tensorboard in output directory
#
module load cuda/8.0-cudnn6

xdg-open http://127.0.0.1:6006
python3 /home/hzwr87/.local/lib/python3.5/site-packages/tensorboard/main.py --logdir=output/
