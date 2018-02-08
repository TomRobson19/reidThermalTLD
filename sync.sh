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
  inotifywait -r -e modify,attrib,close_write,move,create,delete src
  rsync -azr --exclude "*.swp" --exclude 'main' --exclude 'dataExtraction' --exclude '__pycache__' --exclude 'people/' --exclude 'output' --exclude 'stderr-filename'--exclude 'stdout-filename' src/ hzwr87@ncc.clients.dur.ac.uk:~/reidSource --delete
done