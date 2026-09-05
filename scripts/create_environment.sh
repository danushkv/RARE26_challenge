#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
env_dir="${1:-${repo_dir}/.venv}"
python_bin="${PYTHON_BIN:-python3.12}"

if [[ -e "${env_dir}" ]]; then
    echo "Environment path already exists: ${env_dir}" >&2
    echo "Choose a new path or remove the existing environment explicitly." >&2
    exit 1
fi

"${python_bin}" -m venv "${env_dir}"
"${env_dir}/bin/python" -m pip install --upgrade "pip==25.3"
"${env_dir}/bin/python" -m pip install -r "${repo_dir}/requirements.txt"
"${env_dir}/bin/python" -m pip install -e "${repo_dir}"
"${env_dir}/bin/python" -m pip check

echo "Environment created at ${env_dir}"
echo "Activate it with: source ${env_dir}/bin/activate"
