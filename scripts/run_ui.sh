#!/usr/bin/env bash
# Start the lightweight comparison UI, bound to localhost only.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
export RAG_CONFIG_FILE="${RAG_CONFIG_FILE:-configs/mock-demo.json}"
python ui/app.py
