#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "Usage: $0 DATA_DIR PRETRAINED_CHECKPOINT [OUTPUT_ROOT]" >&2
    exit 2
fi

data_dir="$1"
pretrained_checkpoint="$2"
output_root="${3:-outputs}"

CUDA_VISIBLE_DEVICES=0 python -m rare26_nct_tso.train \
    --family linear \
    --data-dir "$data_dir" \
    --pretrained-path "$pretrained_checkpoint" \
    --split-csv data/splits/5fold_cv.csv \
    --output-root "$output_root" &
linear_pid=$!

CUDA_VISIBLE_DEVICES=1 python -m rare26_nct_tso.train \
    --family eqc4 \
    --data-dir "$data_dir" \
    --pretrained-path "$pretrained_checkpoint" \
    --split-csv data/splits/5fold_cv.csv \
    --output-root "$output_root" &
eqc4_pid=$!

status=0
wait "$linear_pid" || status=$?
wait "$eqc4_pid" || status=$?
exit "$status"

