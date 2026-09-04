# Reproducibility notes

## Data layout

The image root is not distributed by this repository. It must contain paths
matching the split file:

```text
DATA_DIR/
├── center_1/
│   ├── ndbe/*.png
│   └── neo/*.png
└── center_2/
    ├── ndbe/*.png
    └── neo/*.png
```

The exact split is committed because regenerating a stratified split from an
unordered filesystem traversal is not guaranteed to reproduce the same fold
assignment.

## External model artifact

Download or otherwise obtain the GastroNet-5M DINOv1 ResNet-50 checkpoint
under its original license. Verify its digest using
`scripts/verify_artifacts.py`; the weights must not be substituted silently.

## Reproduction levels

- **Method reproduction:** matching data, split, architecture, preprocessing,
  optimizer, epoch counts, and seeds. This repository supports this level.
- **Numerical reproduction:** matching every floating-point checkpoint bit.
  This is not guaranteed across CUDA/cuDNN versions and GPU architectures.
- **Historical environment reproduction:** requires the missing `req_c.txt`
  export. Add it under `environment/` when it becomes available.

## Evaluation interpretation

`rare26_nct_tso.selection_metric` reproduces the analytic metric used to pick
historical checkpoints. `rare26_nct_tso.challenge_metric` implements the
organizer's publicly described resampling procedure. They are intentionally
separate so an implementation detail used during training is not confused
with the challenge leaderboard metric.

