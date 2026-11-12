#!/usr/bin/env bash
# The one command we run:  ./run.sh <input_dir> <output.json>
# Change what it calls, not how it is called. It must finish in 30 minutes.
set -euo pipefail
if [ "$#" -ne 2 ]; then
  echo "usage: ./run.sh <input_dir> <output.json>" >&2
  exit 2
fi
if [ -x "$(dirname "$0")/.venv/Scripts/python.exe" ]; then
    PYTHON="$(dirname "$0")/.venv/Scripts/python.exe"
elif [ -x "$(dirname "$0")/.venv/bin/python" ]; then
    PYTHON="$(dirname "$0")/.venv/bin/python"
else
    PYTHON="python3"
fi

"$PYTHON" "$(dirname "$0")/solution.py" "$1" "$2" 
