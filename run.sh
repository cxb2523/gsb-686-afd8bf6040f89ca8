#!/usr/bin/env bash
# run.sh - starts Changeling Imago. Sets it up first if it needs it (setup.sh).
#   ./run.sh                                       opens the app
#   ./run.sh -s face.png -t clip.mp4 -o out.mp4    swaps without the app
# Uses your NVIDIA GPU when there is one; add --execution-provider cpu to force the CPU.
set -euo pipefail
cd "$(dirname "$0")"

bash ./setup.sh

VENV_PY=psdenv/bin/python
case "$(uname -s)" in MINGW*|MSYS*|CYGWIN*) VENV_PY=psdenv/Scripts/python.exe ;; esac

PROVIDER=()
case " $* " in
    *" --execution-provider"*) ;;
    *) if command -v nvidia-smi >/dev/null 2>&1; then PROVIDER=(--execution-provider cuda); fi ;;
esac
exec "$VENV_PY" run.py ${PROVIDER[@]+"${PROVIDER[@]}"} "$@"
