# RARE26 NCT-TSO

Reproducible training and internal validation code for the RARE26 submission
that combines five ResNet-50 linear-head models with five ResNet-50 models
using a C4-equivariant classifier head.

This repository contains training code, the exact five-fold assignment,
challenge-style evaluation, artifact checksums, and a reusable description of
the training method. It does **not** redistribute the challenge images, the
GastroNet pretrained checkpoint, trained weights, or Grand Challenge test
data.

## Repository layout

```text
rare26_nct_tso/
├── artifacts/                 # checksums for external inputs
├── data/splits/               # exact labeled-data fold assignment
├── docs/                      # method and reproducibility documentation
├── environment/               # environment provenance / future req_c.txt
├── scripts/                   # training, evaluation, and verification entry points
├── src/rare26_nct_tso/        # models, training, and metrics
├── tests/                     # lightweight regression tests
├── pyproject.toml
└── requirements.txt
```

## Setup

Python 3.12.3 was used for the original jobs.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The complete historical environment export (`req_c.txt`) was not available
when this repository was assembled. See `environment/README.md` before making
an exact-environment claim.

## Required inputs

1. Place the RARE26 images under a directory with `center_1/{ndbe,neo}` and
   `center_2/{ndbe,neo}` subdirectories.
2. Obtain `RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth` under its
   applicable license.
3. Verify the immutable inputs:

```bash
python scripts/verify_artifacts.py \
    /path/to/RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth
```

## Train

Train a family sequentially across all five folds:

```bash
./scripts/train_family.sh linear /path/to/rare_dataset /path/to/pretrained.pth
./scripts/train_family.sh eqc4  /path/to/rare_dataset /path/to/pretrained.pth
```

On a workstation with two GPUs, train both families concurrently:

```bash
./scripts/train_two_gpus.sh /path/to/rare_dataset /path/to/pretrained.pth
```

To resume work manually or schedule folds independently, select one fold:

```bash
python -m rare26_nct_tso.train \
    --family eqc4 \
    --fold 3 \
    --data-dir /path/to/rare_dataset \
    --pretrained-path /path/to/pretrained.pth
```

Outputs are written to `outputs/{linear,eqc4}/fold_N/`. Each directory
contains `best.pth`, `run_config.json`, and `oof_val_predictions.csv`.

## Evaluate the OOF predictions

After all ten runs finish:

```bash
./scripts/evaluate_oof.sh outputs evaluation_results
```

The evaluator validates every prediction against the committed fold CSV and
writes:

- `evaluation_results/metrics.json`
- `evaluation_results/oof_scores.csv`

The evaluation code reports the publicly described 1,000-repeat RARE26
resampling metric. It also preserves within-fold percentile normalization used
for the internal linear+C4 ensemble comparison. See
`docs/training_strategy.md` for the validation limitations and the Open
Validation Phase result.

## Test

Lightweight tests do not require a GPU:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
python -m compileall -q src tests scripts
```

## Method description

The challenge-ready training-method text is in
[`docs/training_strategy.md`](docs/training_strategy.md). Artifact provenance
and exact/non-exact reproducibility boundaries are documented in
[`docs/reproducibility.md`](docs/reproducibility.md). Audited internal and
Open Validation results are separated in
[`docs/validation_results.md`](docs/validation_results.md).

## License

Code in this repository is released under the MIT License. Dataset and model
artifacts retain their own licenses and are not covered by this repository's
license.
