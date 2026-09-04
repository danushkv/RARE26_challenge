# Required artifact checksums

The model weights and image data are intentionally not committed.

| Artifact | SHA-256 |
|---|---|
| `data/splits/5fold_cv.csv` | `dc07de48a0cb1c94a65069a75bff7903be262d04619b78f48c1eb03903e4e2c4` |
| `RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth` | `5688929fea4437031604001495fb77fb18cdc2ff92ae24120f93aeceaf5aa16d` |

Verify both before a reproduction run:

```bash
sha256sum data/splits/5fold_cv.csv /path/to/RN50_Billion-Scale-SWSL_2BGastroNet-5M_DINOv1.pth
```

