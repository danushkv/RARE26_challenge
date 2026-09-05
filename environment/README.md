# Environment creation and provenance

Create the environment from the repository root with:

```bash
./scripts/create_environment.sh
source .venv/bin/activate
```

The script requires Python 3.12 and creates `.venv`, installs the pinned
dependencies, installs this package in editable mode, and runs `pip check`.
Use `PYTHON_BIN=/path/to/python3.12` when Python 3.12 is not available as
`python3.12`. A Fortran compiler such as `gfortran` may be required when pip
needs to build the `py3nj` dependency of escnn from source.

The recorded training jobs used Python 3.12.3 and PyTorch 2.7.1+cu126 on an
NVIDIA H100. On the original cluster, load the matching modules first:

```text
release/25.06
GCCcore/13.3.0
Python/3.12.3
CUDA
```

## `req_x.txt` audit

The supplied `rare_26/req_x.txt` was inspected on 5 September 2026. It contains
only this single entry:

```text
pip @ file:///build/pip-25.3-py3-none-any.whl#sha256=9655943313a94722b7774661c21049070f6bbb0a1516bf02f7c8d5d9201514cd
```

It is preserved as `environment/req_x.txt`, but it is not a usable export of
the training environment because it omits PyTorch, escnn, timm, NumPy, and all
other training dependencies.

The repository-level `requirements.txt` was reconstructed from the recorded
training environment and source imports. The observed environment contained
NumPy 1.26.4 alongside OpenCV 4.13.0.92, even though that OpenCV release
declares NumPy 2 or newer on Python 3.12. To make a fresh installation
dependency-resolvable while retaining the recorded NumPy version, the clean
specification uses OpenCV 4.11.0.86. Consequently, this is a compatible method
reproduction environment, not a bit-for-bit historical export.
