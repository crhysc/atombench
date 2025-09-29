#!/usr/bin/env bash
set -euo pipefail

# Optional: fail with a helpful message if a non-conda dir exists at the target prefix
if conda env list | awk '{print $1}' | grep -qx flowmm; then
  : # ok, env name already registered with conda
else
  prefix="$(conda info --base)/envs/flowmm"
  if [[ -d "$prefix" && ! -f "$prefix/conda-meta/history" ]]; then
    echo "Error: Non-conda folder exists at $prefix. Remove or rename it, then re-run." >&2
    exit 1
  fi
fi

cd models/flowmm

mamba env create -f environment.yml -y
set +u
eval "$(conda shell.bash hook)"
conda activate flowmm
set -u

# sanity checks
python -c 'import sys; print(sys.version)'
pip --version || (echo "pip missing after activation"; exit 1)

# install extras
pip install uv
uv pip install "jarvis-tools>=2024.5" "pymatgen>=2024.1" pandas numpy tqdm matminer
uv pip install -e . \
  -e remote/riemannian-fm \
  -e remote/cdvae \
  -e remote/DiffCSP-official

# only create the sentinel on success
touch "${PROJECT_ROOT:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}/flowmm_env.created"
echo "Done"

