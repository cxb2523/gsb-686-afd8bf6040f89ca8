#!/usr/bin/env bash
# setup.sh - gets Changeling ready: the psdenv virtual environment (Python 3.9 - 3.12),
# the packages in requirements.txt and the models. Safe to run again: it only redoes what's
# missing or changed. run.sh calls it for you.
set -euo pipefail
cd "$(dirname "$0")"

VENV_DIR=psdenv
VENV_PY="$VENV_DIR/bin/python"
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) VENV_PY="$VENV_DIR/Scripts/python.exe" ;; esac

# 1) The virtual environment (TensorFlow and PyTorch need Python 3.9 - 3.12).
if [ ! -x "$VENV_PY" ]; then
    echo "Creating the virtual environment..."
    BASE_PY=""
    for candidate in python3.11 python3.12 python3.10 python3.9 python3 python; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c \
            'import sys; sys.exit(0 if (3, 9) <= sys.version_info[:2] <= (3, 12) else 1)' 2>/dev/null; then
            BASE_PY="$candidate"
            break
        fi
    done
    if [ -z "$BASE_PY" ]; then
        echo "Python 3.9 - 3.12 is needed (3.11 is best)." >&2
        exit 1
    fi
    "$BASE_PY" -m venv "$VENV_DIR"
fi

# 2) The packages, whenever requirements.txt changed since the last install.
#    The first time downloads several GB (PyTorch, TensorFlow) - give it a while.
if ! cmp -s requirements.txt "$VENV_DIR/.requirements"; then
    echo "Installing the packages - the first time takes a while..."
    "$VENV_PY" -m pip install --upgrade pip
    "$VENV_PY" -m pip install -r requirements.txt
    cp requirements.txt "$VENV_DIR/.requirements"
fi

# 3) Any missing models.
"$VENV_PY" scripts/download_models.py

# 4) Which GPU, and a warning if its driver is too old.
"$VENV_PY" scripts/check_gpu.py || true

echo "Changeling Imago is ready. Start it with ./run.sh"
