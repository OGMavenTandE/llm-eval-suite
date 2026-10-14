#!/bin/sh
# Build the offline bundle, install it with no package index, and run doctor
# plus a stub quant-compare. A network attempt fails the script.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

exec "$PY" -m llm_eval verify-offline "$@"
