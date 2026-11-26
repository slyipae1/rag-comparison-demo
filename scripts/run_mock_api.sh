#!/usr/bin/env bash
# Start both offline mock pipelines on localhost. No credentials or services needed.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="src:${PYTHONPATH:-}"
python -m rag_comparison.api.app --config-file configs/mock-demo.json --all
