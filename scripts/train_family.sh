#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
    echo "Usage: $0 {linear|eqc4} DATA_DIR PRETRAINED_CHECKPOINT [OUTPUT_ROOT]" >&2
    exit 2
fi

family="$1"
data_dir="$2"
pretrained_checkpoint="$3"
output_root="${4:-outputs}"

if [[ "$family" != "linear" && "$family" != "eqc4" ]]; then
    echo "Family must be linear or eqc4" >&2
    exit 2
fi

python -m rare26_nct_tso.train \
    --family "$family" \
    --data-dir "$data_dir" \
    --pretrained-path "$pretrained_checkpoint" \
    --split-csv data/splits/5fold_cv.csv \
    --output-root "$output_root"

