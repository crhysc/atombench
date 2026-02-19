#!/bin/bash
set -e
export DEBUG="true"
python scripts/alexandria_preprocess.py atomgpt \
       --csv-files dataset1.csv dataset2.csv \
       --output ./atomgpt_data --seed 123 --max-size 8253
#rm -rf agm*
