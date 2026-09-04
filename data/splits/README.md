# Split provenance

`5fold_cv.csv` is the exact split used for the ten submitted-model training
runs. Its SHA-256 is recorded in `artifacts/checksums.md`.

This split is derived from the challenge dataset and remains subject to the
dataset/challenge terms; it is not relicensed by the repository's MIT License.

The split was generated with shuffled stratified five-fold cross-validation
(`random_state=42`) over the combined `center + class` key. It is therefore
stratified at image level, not grouped by patient or examination. The public
filenames are anonymized and contain no group identifiers.

Expected totals:

| Class | Images |
|---|---:|
| NDBE (`target=0`) | 2,937 |
| Neoplasia (`target=1`) | 158 |
| Total | 3,095 |

Each fold contains 619 images and 31–33 positive images.
