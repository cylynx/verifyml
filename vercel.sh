#!/usr/bin/env bash
# script to set up and activate a python virtual env in vercel

amazon-linux-extras install python3.8
pip install virtualenv
virtualenv venv
source venv/bin/activate
python --version
pip install -r requirements.txt