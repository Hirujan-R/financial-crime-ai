#!/usr/bin/env bash
# Launch the analyst dashboard.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then
  PY=python3
fi

exec "$PY" -m streamlit run src/financial_crime_ai/dashboard/app.py "$@"
