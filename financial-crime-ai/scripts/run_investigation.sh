#!/usr/bin/env bash
# Investigate a single flagged transaction from the CLI.
# Usage: scripts/run_investigation.sh T00194810
set -euo pipefail
cd "$(dirname "$0")/.."

PY=./.venv/bin/python
if [ ! -x "$PY" ]; then
  PY=python3
fi

if [ -z "${1:-}" ]; then
  echo "Usage: $0 <txn_id>" >&2
  exit 1
fi

"$PY" - <<EOF
import json
from financial_crime_ai.agent.context import InvestigatorContext
from financial_crime_ai.agent.investigator import CaseInvestigator

ctx = InvestigatorContext.from_project_dir(".")
report = CaseInvestigator(ctx).investigate("$1")
print(json.dumps(report.to_dict(), indent=2, default=str))
EOF
