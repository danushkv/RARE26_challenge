# Environment provenance

The recorded training jobs used Python 3.12.3 and PyTorch 2.7.1+cu126 on an
NVIDIA H100. The original cluster setup loaded:

```text
release/25.06
GCCcore/13.3.0
Python/3.12.3
CUDA
```

The repository-level `requirements.txt` is the smallest pinned environment
reconstructed from the training source and installed package metadata. The user-provided full environment
export was expected at `rare_26/req_c.txt`, but that file was not present when
this repository was assembled. Once available, copy it here as `req_c.txt` and
record its SHA-256 digest in `artifacts/checksums.md`. Do not label the minimal
requirements file as the exact historical environment.
