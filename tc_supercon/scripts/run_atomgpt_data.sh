#!/bin/bash
set -e
mkdir -p agpt_tc_supercon
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python scripts/id_prop.py \
    --dataset dft_3d \
    --id-key jid \
    --output agpt_tc_supercon \
    --target Tc_supercon \
    --seed 123 \
    --max-size 1000
