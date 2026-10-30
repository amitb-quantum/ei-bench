#!/usr/bin/env bash
# Creates an isolated conda env "eibench" (does not touch any existing env).
set -euo pipefail
ENV_NAME="${1:-eibench}"
cd "$(dirname "$0")/.."

if ! command -v conda >/dev/null; then
  echo "conda not found on PATH (expected miniforge3). Activate it first: source ~/miniforge3/bin/activate"; exit 1
fi
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "env '$ENV_NAME' already exists - reusing it"
else
  conda create -y -n "$ENV_NAME" python=3.12
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install -U pip
python -m pip install -r requirements.txt   # PyPI torch wheels ship Blackwell (sm_120) kernels
python -m pip install -e .
python scripts/env_check.py
