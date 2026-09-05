# RARE26 NCT-TSO

Reproducible training and internal validation code for the RARE26 submission
that combines five ResNet-50 linear-head models with five ResNet-50 models
using a C4-equivariant classifier head.

This repository contains training code, the exact five-fold assignment,
challenge-style evaluation, artifact checksums, and a reusable description of
the training method. The ten trained submission checkpoints are published separately on
[Hugging Face](https://huggingface.co/danushkv/RARE26).

## Repository layout

```text
RARE26_challenge/
├── artifacts/                 # checksums for external inputs
├── data/splits/               # exact labeled-data fold assignment
├── docs/                      # method and reproducibility documentation
├── environment/               # environment and req_x.txt provenance
├── scripts/                   # training, evaluation, and verification entry points
├── src/rare26_nct_tso/        # models, training, and metrics
├── tests/                     # lightweight regression tests
├── pyproject.toml
└── requirements.txt
```

## Setup

Python 3.12.3 was used for the original jobs.

On the original cluster, training used the existing environment:

```bash
source /data/cat/ws/dave995e-my_folder/dave995e-folder-1781485218/envs/rare/bin/activate
```

The full package snapshot exported from that environment is committed as
[`environment/req_x.txt`](environment/req_x.txt). The activation path above is
machine-specific and is included for provenance; it is not expected to work on
another system.

For a portable clean installation, run:

```bash
./scripts/create_environment.sh
source .venv/bin/activate
```

See [`environment/README.md`](environment/README.md) for the distinction
between the captured training environment and the portable dependency set.

## Required inputs

1. Place the RARE26 images under a directory with `center_1/{ndbe,neo}` and
   `center_2/{ndbe,neo}` subdirectories.
2. Download `RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth` from the
   [Theta Vision Cortex GastroNet-5M listing](https://cortex.thetavision.nl/dataset-provider/listing/2/)
   under its applicable access and license terms.
3. Verify the immutable inputs:

```bash
python scripts/verify_artifacts.py \
    /path/to/RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth
```

## Submitted checkpoints

The exact five linear-head and five eqC4 checkpoints used by the submitted
ensemble can be downloaded from
[C4-ensemble chkpts](https://huggingface.co/danushkv/RARE26).

These are the final fine-tuned inference checkpoints. They are distinct from
the GastroNet checkpoint in **Required inputs**, which initializes training.

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

## Results

| Evaluation | Model | PPV at 90% recall | AUROC | Average precision |
|---|---|---:|---:|---:|
| Internal five-fold OOF | Linear ResNet-50 | 0.2288 | 0.9699 | 0.8864 |
| Internal five-fold OOF | C4-head ResNet-50 | 0.3971 | 0.9744 | 0.9099 |
| Internal five-fold OOF | Equal linear+C4 ensemble | 0.4426 | 0.9752 | 0.9233 |
| Official Open Validation | Submitted linear+C4 ensemble | **0.0106** | Not provided | Not provided |

The internal split is image-stratified rather than patient/examination-grouped
and was also used for checkpoint selection, so its results can be optimistic.
The official Open Validation value is the primary external result. Full
interpretation is provided in [`docs/validation_results.md`](docs/validation_results.md).

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

## Citation

If you use this training strategy or its checkpoints, please cite:

> Danush Kumar Venkatesh. *GastroC4 Ensemble: GastroNet-Initialized ResNet-50
> Models with Linear and C4-Equivariant Heads for Rare Neoplasia Detection.*
> RARE26 Challenge submission, 2026.

```bibtex
@misc{venkatesh2026gastroc4,
  author = {Venkatesh, Danush Kumar},
  title = {GastroC4 Ensemble: GastroNet-Initialized ResNet-50 Models with
           Linear and C4-Equivariant Heads for Rare Neoplasia Detection},
  year = {2026},
  note = {RARE26 Challenge submission},
  url = {https://huggingface.co/danushkv/RARE26}
}
```

## License

Code in this repository is released under the [MIT License](LICENSE). PyTorch,
escnn, and other dependencies retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). Dataset and model artifacts
are not covered by this repository's license.
