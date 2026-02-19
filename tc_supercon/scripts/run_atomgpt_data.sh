#!/bin/bash
set -e
mkdir -p atomgpt_data
python scripts/data_preprocess.py atomgpt  --dataset dft_3d --output ./atomgpt_data  \
                                           --target Tc_supercon --seed 123 --max-size 1058
rm -rf JVASP*
