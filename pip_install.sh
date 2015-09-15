#!/bin/bash

# Travis doesn't seem to be able to handle keeping the python paths straight
# when it installs modules from both apt and pip. Consequently, this file
# downloads pip dependencies to a temporary directory and imports them from
# there.

mkdir -p pip_externals
pip install -t pip_externals -r requirements.txt

# Set the Python Path correctly.
export PYTHONPATH="${PYTHONPATH}:pip_externals/"
