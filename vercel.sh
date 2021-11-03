#!/usr/bin/env bash
# script to set up and activate a python virtual env in vercel

amazon-linux-extras install python3.8
alias python3=python3.8

pip3.8 install virtualenv
virtualenv venv
source venv/bin/activate
python3 --version
pip3.8 install -r requirements.txt