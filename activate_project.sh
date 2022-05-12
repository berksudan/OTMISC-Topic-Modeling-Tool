#!/bin/bash

CURRENT_PATH="$(
  cd "$(dirname "${BASH_SOURCE[0]}")" || exit
  pwd -P
)"
cd "$CURRENT_PATH" || exit # Change directory to where bash script resides.

PROMPT_DIRTRIM=1

source "./venv/bin/activate"
