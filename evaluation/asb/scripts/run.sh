#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE=1
cd "$(dirname "$0")/.."
python scripts/agent_attack.py "$@"
