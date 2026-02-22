#!/bin/bash


source ~/.bashrc
eval "$(conda shell.bash hook)"
conda activate flowmm

SECONDS=0
start_iso="$(date -Iseconds)"

# remove yamls that might exist from past pipeline runs
rm -f models/flowmm/src/flowmm/rfm/manifolds/stats_supercon*
rm -f models/flowmm/src/flowmm/rfm/manifolds/stats_alex*
python - <<'PYCODE'
import os
files = ["atom_density.yaml", "spd_pLTL_stats.yaml", "spd_std_coef.yaml", "lattice_params_stats.yaml"]
path = "models/flowmm/src/flowmm/rfm/manifolds"
for file in files:
        filepath = os.path.join(path,file)
        if os.path.exists(filepath):
                os.remove(filepath)
PYCODE

# create necessary yamls for training and inference
cd models/flowmm
python -u -m flowmm.rfm.manifolds.spd
python -u -m flowmm.rfm.manifolds.lattice_params
python -u -m flowmm.model.standardize data=supercon
python -u -m flowmm.model.standardize data=alexandria

elapsed="$SECONDS"
end_iso="$(date -Iseconds)"

printf "Start: %s\nEnd:   %s\nElapsed: %ds (%.2f min, %.2f hr)\n" \
  "$start_iso" "$end_iso" "$elapsed" \
  "$(awk "BEGIN{print $elapsed/60}")" \
  "$(awk "BEGIN{print $elapsed/3600}")"
