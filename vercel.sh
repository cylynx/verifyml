#!/usr/bin/env bash
# install script to set up a python virtual env in vercel

amazon-linux-extras install python3.8

# docs.sh uses python3
alias python3=python3.8

# install and activate virtual env
pip3.8 install virtualenv
virtualenv venv
source venv/bin/activate

# install requirements
pip3.8 install -r requirements.txt

# venv and packages do not persist across vercel's install and build commands?
# use `source venv/bin/activate && ./docs.sh build` for build to ensure same env