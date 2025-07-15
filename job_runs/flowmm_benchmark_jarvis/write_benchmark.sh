#!/bin/bash
conda activate flowmm

rm -f AI-AtomGen*

python inspect_pt.py \
   --pt_path ./outputs/rfmcsp-conditional-*/????????/checkpoints/inferences/consolidated_reconstruct.pt \
   --output_csv AI-AtomGen-prop-dft_3d-test-rmse.csv

mv ../../models/flowmm/AI-AtomGen-prop-dft_3d-test-rmse.csv .

