#! /usr/bin/env bash

virtualenv -p python3.6 venv
# /usr/bin/python3 venv
source venv/bin/activate

# install everything
pip install -r requirements.txt
pip install -r requirements2.txt

deactivate
