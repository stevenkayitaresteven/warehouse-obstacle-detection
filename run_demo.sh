#!/usr/bin/env bash
# One-command demo launcher for macOS and Linux.
#   ./run_demo.sh            start the demo
#   ./run_demo.sh --full     also install the real-image detection tab
#   ./run_demo.sh --check    install and run a self-check
cd "$(dirname "$0")" || exit 1

PY=""
for c in python3.12 python3.11 python3.13 python3.10 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && \
     "$c" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' >/dev/null 2>&1; then
    PY="$c"
    break
  fi
done

if [ -z "$PY" ]; then
  echo "Python 3.10 or newer was not found."
  echo "  macOS:          install Python 3.12 from https://www.python.org/downloads/"
  echo "                  (or: brew install python@3.12)"
  echo "  Ubuntu/Debian:  sudo apt install python3 python3-venv"
  exit 1
fi

echo "Using: $(command -v "$PY")"
exec "$PY" run_demo.py "$@"
