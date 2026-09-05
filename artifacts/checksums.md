# Required artifact checksums

The model weights and image data are intentionally not committed.

| Artifact | Source | SHA-256 |
|---|---|---|
| `data/splits/5fold_cv.csv` | Included in this repository | `dc07de48a0cb1c94a65069a75bff7903be262d04619b78f48c1eb03903e4e2c4` |
| `RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth` | [Theta Vision Cortex](https://cortex.thetavision.nl/dataset-provider/listing/2/) | `5688929fea4437031604001495fb77fb18cdc2ff92ae24120f93aeceaf5aa16d` |

Verify both before a reproduction run:

```bash
sha256sum data/splits/5fold_cv.csv /path/to/RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth
```

## Final submission ensemble

The trained checkpoints are hosted at
[`danushkv/RARE26`](https://huggingface.co/danushkv/RARE26), pinned to revision
`7dd88ce6b82f1591e5dd21c5890401c68bccfc69`.

| Hugging Face path | SHA-256 |
|---|---|
| `model_chkpts/checkpoints/linear/fold_0_best.pth` | `f93bfdcfb52430d19639196a9271d5c676c12f4a6951e1828ecb62bfff8881bf` |
| `model_chkpts/checkpoints/linear/fold_1_best.pth` | `7dceab2d9c346ec3cab19cc8362c966336a2d09232c9f1cceee7311dcde9aff0` |
| `model_chkpts/checkpoints/linear/fold_2_best.pth` | `0a45e1f85476aff2aa964fbe065140fcd858113278ea48dacf284a760feb3347` |
| `model_chkpts/checkpoints/linear/fold_3_best.pth` | `3eadce5d49d5928062fe9fa65fb03a040df21812d0669b26a20c9d2c258ede6d` |
| `model_chkpts/checkpoints/linear/fold_4_best.pth` | `a8aaf56716dba9440c6b50f2f1a2ebb652b5a5d9e046672b2713207850b98f79` |
| `model_chkpts/checkpoints/eqC4/fold_0_best.pth` | `b101ead579de955130f77daaeb4e82954b246633398c3aefe252884a2ee0aa1e` |
| `model_chkpts/checkpoints/eqC4/fold_1_best.pth` | `1c7d6de112664e7dcdfaeb5083220614de076c6162190d9962d43efbe96a64ce` |
| `model_chkpts/checkpoints/eqC4/fold_2_best.pth` | `1e416dab041eed6fd4e1d5d9129c01d78105a0a5bede2d41c33823b67d9b7bc7` |
| `model_chkpts/checkpoints/eqC4/fold_3_best.pth` | `8a4db5a1f6b0de6c0edee4da410d3cd2da046b8e0b32d4f9b5c89101427a14c6` |
| `model_chkpts/checkpoints/eqC4/fold_4_best.pth` | `fb1a78c716201a91b1b0cf379f597afd1c17ba99d858a9a3c3000d032528d746` |
