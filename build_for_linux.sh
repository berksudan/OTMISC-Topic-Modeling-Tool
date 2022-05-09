#!/bin/bash

# Change current directory to project directory.
cd "$(dirname "$0")" || exit

# Install necessary apt packages for development
sudo apt install python3.8-venv python3-pip

# Create virtual environment directory
python3 -m venv venv/

# Activate virtual environment
source venv/bin/activate

# Upgrade Python
python3 -m pip install --upgrade pip

# Check version of pip
# Version must be below 18.XX and compatible with Python 3.5+
pip --version

# Install dependencies
pip install -r "requirements.txt"
