#!/usr/bin/env bash
set -euo pipefail

results_root="${1:-outputs}"
output_dir="${2:-evaluation_results}"

python -m rare26_nct_tso.evaluate \
    --results-root "$results_root" \
    --split-csv data/splits/5fold_cv.csv \
    --output-dir "$output_dir" \
    --iterations 1000 \
    --seed 42

