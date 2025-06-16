#!/bin/bash
set -e
mkdir -p /lab/mml/kipp/677/jarvis/rhys/benchmarks/models/cdvae/data/alexandria/
uv pip install jarvis-tools pymatgen numpy pandas tqdm
python scripts/generate_data_cdvae.py \
    --csv-files dataset1.csv dataset2.csv \
    --id-key mat_id \
    --target Tc \
    --max-size 1000 \
    --val-ratio 0.1 \
    --test-ratio 0.1 \
    --split-seed 123 \
    --output-dir .
python - <<'PYCODE'
import os
path = "../models/cdvae/data/alexandria"
files = ["train.csv", "test.csv", "val.csv"]
for file in files:
	file_path = os.path.join(path,file)
	if os.path.exists(file_path):
		os.remove(file_path)
	os.rename(file, file_path)
PYCODE

