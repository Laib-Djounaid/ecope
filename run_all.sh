#!/usr/bin/env bash
# End-to-end pipeline. Every step is resumable (finished jobs are skipped).
# Usage: bash run_all.sh [scheme ...]      (default: spatial)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/Scripts/python}   # Windows venv; use .venv/bin/python on Linux/macOS
SCHEMES=${@:-spatial}
mkdir -p logs

$PY src/01_build_dataset.py   > logs/01_build_dataset.log 2>&1
$PY src/01b_data_report.py    > logs/01b_data_report.log 2>&1
$PY src/02_splits.py          > logs/02_splits.log 2>&1
for s in $SCHEMES; do
  $PY -W ignore src/03_tabular.py --scheme "$s" --n-jobs 3 --threads 3 > "logs/03_tabular_$s.log" 2>&1
  $PY -W ignore src/04_graph.py   --scheme "$s" --n-jobs 3 --threads 3 > "logs/04_graph_$s.log" 2>&1
done
echo "done: $SCHEMES"
