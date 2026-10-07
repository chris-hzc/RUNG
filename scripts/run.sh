#!/usr/bin/env bash
# Train RUNG on clean graphs, then evaluate it under adaptive global PGD attacks.
#
#   bash scripts/run.sh [dataset] [gamma] [norm]
#   bash scripts/run.sh cora 36 MCP
set -euo pipefail

DATA=${1:-cora}
GAMMA=${2:-36}
NORM=${3:-MCP}

python clean.py  --model RUNG --norm "$NORM" --gamma "$GAMMA" --data "$DATA"
python attack.py --model RUNG --norm "$NORM" --gamma "$GAMMA" --data "$DATA"
