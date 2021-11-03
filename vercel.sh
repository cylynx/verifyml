#!/usr/bin/env bash
# script to set up and activate a python virtual env in vercel

amazon-linux-extras install python3.8
python3.8 -m venv venv
source venv/bin/activate
pip3.8 install -r requirements.txt