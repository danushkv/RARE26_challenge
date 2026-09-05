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

Download the GastroNet-5M DINOv1 ResNet-50 checkpoint from the
[Theta Vision Cortex listing](https://cortex.thetavision.nl/dataset-provider/listing/2/)
under its original access and license terms. Verify its digest using
`scripts/verify_artifacts.py`; the weights must not be substituted silently.

This GastroNet artifact is the initialization used before fine-tuning. It is
not one of the ten final challenge-submission checkpoints.

## Final submission checkpoints

The final five linear-head and five eqC4 fold checkpoints, together with the
ensemble calibration configuration, are hosted in
[`danushkv/RARE26`](https://huggingface.co/danushkv/RARE26). Reproduction and
auditing should use commit
`7dd88ce6b82f1591e5dd21c5890401c68bccfc69`, rather than an unpinned `main`
branch. File-level SHA-256 digests are recorded in
`artifacts/checksums.md`.

## Reproduction levels

- **Method reproduction:** matching data, split, architecture, preprocessing,
  optimizer, epoch counts, and seeds. This repository supports this level.
- **Numerical reproduction:** matching every floating-point checkpoint bit.
  This is not guaranteed across CUDA/cuDNN versions and GPU architectures.
- **Historical environment reproduction:** the available `req_x.txt` records
  only `pip`, so it cannot reconstruct every transitive package from the
  original job. The pinned repository requirements provide a clean compatible
  environment, but should not be described as a bit-for-bit historical export.

## Evaluation interpretation

`rare26_nct_tso.selection_metric` reproduces the analytic metric used to pick
historical checkpoints. `rare26_nct_tso.challenge_metric` implements the
organizer's publicly described resampling procedure. They are intentionally
separate so an implementation detail used during training is not confused
with the challenge leaderboard metric.
