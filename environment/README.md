# Environment creation and provenance

## Original training environment

The submitted models were trained after activating this environment:

```bash
source /data/cat/ws/dave995e-my_folder/dave995e-folder-1781485218/envs/rare/bin/activate
```

The environment uses Python 3.12.3 and PyTorch 2.7.1+cu126. Its complete
package set was exported on 5 September 2026 with the environment's own
interpreter:

```bash
/data/cat/ws/dave995e-my_folder/dave995e-folder-1781485218/envs/rare/bin/python \
    -m pip freeze --all
```

The resulting snapshot is committed as [`req_x.txt`](req_x.txt). The virtual
environment directory is not committed because it contains machine-specific
binaries, symlinks, and absolute paths.

The original cluster modules were:

```text
release/25.06
GCCcore/13.3.0
Python/3.12.3
CUDA
```

## Portable recreation

Create a clean environment from the repository root with:

```bash
./scripts/create_environment.sh
source .venv/bin/activate
```

The script requires Python 3.12 and creates `.venv`, installs the pinned
dependencies, installs this package in editable mode, and runs `pip check`.
Use `PYTHON_BIN=/path/to/python3.12` when Python 3.12 is not available as
`python3.12`. A Fortran compiler such as `gfortran` may be required when pip
needs to build the `py3nj` dependency of escnn from source.

The captured environment contains both OpenCV 4.13.0.92 and NumPy 1.26.4.
Those installed packages ran the training jobs, but current OpenCV metadata
declares NumPy 2 or newer and `pip check` reports that mismatch. Therefore the
portable `requirements.txt` uses OpenCV 4.11.0.86, which supports NumPy 1.26.4.
Use `req_x.txt` to audit the original environment and `requirements.txt` to
create a clean dependency-resolvable environment.
