# Validation results and interpretation

## Internal out-of-fold analysis

The evaluator in this repository was run on the historical predictions from
all five folds of both model families. The table below uses fold-normalized
percentile scores and 1,000 positive-resampling iterations with seed 42.

| OOF score | PPV@90Recall median | 2.5–97.5% resampling range | AUROC | Enriched-set AP |
|---|---:|---:|---:|---:|
| Linear ResNet-50 | 0.2288 | 0.0196–0.5510 | 0.9699 | 0.8864 |
| C4-head ResNet-50 | 0.3971 | 0.0301–0.6750 | 0.9744 | 0.9099 |
| Equal linear+C4 rank ensemble | 0.4426 | 0.0166–0.8438 | 0.9752 | 0.9233 |

Raw pooled logits are included as diagnostics because independently trained
folds have different score scales. Their resampled PPV medians were 0.0675
for the linear family and 0.3375 for the C4 family.

## External Open Validation result

The submitted linear+C4 ResNet configuration received a PPV@90Recall score of
**0.0106** in the RARE26 Open Validation Phase. No additional metric was
available for that submission.

## Interpretation

The private score is the stronger generalization evidence. The OOF split was
stratified per image rather than by patient/examination, contains only 31–33
positive images per fold, and was also used for epoch-level checkpoint
selection. The public training images come from only two centers. These facts
make the OOF values unsuitable as estimates of performance on a separate
multicenter cohort.

The internal results remain useful for reproducing the model-selection path
and demonstrating why the linear+C4 combination was chosen, but they must be
reported alongside this limitation and the external score.

