#!/bin/bash
#
# Open Tensorboard in output directory
#
xdg-open http://127.0.0.1:6006
python3 /usr/local/lib/python3.5/dist-packages/tensorboard/main.py --logdir=output/
