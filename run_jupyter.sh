#!/bin/bash

CURRENT_PATH="$( cd "$(dirname "$0")" ; pwd -P )"

cd "$CURRENT_PATH" # Change directory to where bash script resides.

"$CURRENT_PATH/venv/bin/jupyter-notebook"

