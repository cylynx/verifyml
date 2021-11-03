#!/usr/bin/env bash
# script to set up a python virtual env in vercel

amazon-linux-extras install python3.8

# docs.sh uses python3
alias python3=python3.8

# install requirements
pip3.8 install -r requirements.txt