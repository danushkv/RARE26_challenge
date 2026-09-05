# RARE26 training strategy

## Submitted configuration

The system combines ten ResNet-50 classifiers: five models with a standard
linear classifier and five models with a cyclic C4-equivariant classifier
head. The backbone is a `timm` ResNet-50 initialized from the
[`RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1`](https://cortex.thetavision.nl/dataset-provider/listing/2/)
self-supervised checkpoint. The original checkpoint classifier is discarded
and a new binary head is initialized before fine-tuning.

The C4 component applies equivariant linear layers only to the pooled
2,048-dimensional backbone representation. Features are interpreted as copies
of the C4 regular representation, processed by two hidden layers containing 64
regular fields each, and mapped to two trivial representations that form the
invariant output logits. The convolutional backbone itself is not equivariant;
the C4 head is a classifier-level inductive bias.

## Data and preprocessing

Training uses the provided 3,095 labeled images and the committed
`data/splits/5fold_cv.csv`. The split contains 2,937 NDBE and 158 neoplasia
images, stratified by class and acquisition center across five folds. For each
fold, four folds are used for training and one for validation.

Images are decoded as RGB and resized/cropped to 518×518 pixels. Validation
uses deterministic resizing followed by ImageNet normalization. Every training
fold uses random resized cropping, horizontal and vertical flips, rotation, and
color jitter. Additional fold-specific transformations provide ensemble
diversity:

| Fold | Additional transformations |
|---:|---|
| 0 | None |
| 1 | Affine and color shifts |
| 2 | Blur |
| 3 | Color shifts and blur |
| 4 | Affine |

## Optimization

All backbone and classifier parameters are fine-tuned. Training uses AdamW
with learning rate 1e-4, weight decay 1e-4, and cosine annealing to 1% of the
initial learning rate. Batch size is 32. Both inverse-frequency weighted
sampling and inverse-frequency cross-entropy class weights are enabled. Fold
seeds are `42 + fold`.

Linear-head models are trained for 50 epochs and C4-head models for 75 epochs.
The checkpoint with the highest validation prevalence-corrected PPV at
least 90% recall is retained. This analytic selection metric is kept in the
repository strictly to reproduce the historical runs.

## Validation and ensemble analysis

Each retained checkpoint generates predictions only for its held-out fold.
The positive score is `logit_1 - logit_0`. Because independently trained folds
can have different logit scales, scores are transformed to within-fold
empirical percentile ranks. The linear and C4 percentile scores are then
averaged with equal weight for the internal ensemble comparison.

Reported challenge-style PPV retains all NDBE examples, resamples neoplasia
examples with replacement to approximate a 100:1 negative-to-positive ratio,
computes PPV at a minimum recall of 90%, and reports the median across 1,000
iterations. AUROC and average precision on the enriched OOF set are secondary
diagnostics.

## Important validation limitation

The supplied fold file is image-stratified rather than patient/examination
grouped, and checkpoints were selected on their corresponding validation
folds. Consequently, internal OOF results can be optimistic and must not be
presented as estimates of private multicenter performance. The submitted
ResNet ensemble obtained an Open Validation Phase PPV@90Recall score of
0.0106. This external result supersedes the much higher internal estimates
when characterizing generalization.
