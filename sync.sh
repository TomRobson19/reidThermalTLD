#!/bin/bash

#Syncs local files with ncc

#SETUP
# 1. Gen a new ssh key:
#	ssh-keygen -t rsa
#	(Make sure you name it something sensible e.g. keyForNCC)
#
# 2. Copy public key to server:
#	Copy the .pub file to ~/ssh/authorized_keys
#	(You may have to mkdir the .ssh dir)
#
#	You can now use ssh without entering your password
#
# 3. Edit paths for directories
# 4. Run this file

while true; do  
  rsync -azr --exclude "*.out" --exclude "*.csv" --exclude "*.swp" --exclude 'main' --exclude '/oldData' --exclude '/data' --exclude '/classificationsCNN' --exclude '/classificationsREID' --exclude 'dataExtraction' --exclude '__pycache__' --exclude 'people/' --exclude 'newPeople/' --exclude '/output' --exclude '/saved_models' --exclude '/env' --exclude 'stdout-grid' --exclude 'stderr-grid' --exclude 'stdout-filename' --exclude 'stderr-filename' src/ hzwr87@ncc.clients.dur.ac.uk:~/reidSource --delete
  inotifywait -r -e modify,attrib,close_write,move,create,delete src
done