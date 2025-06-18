#!/usr/bin/env python3
# conda activate /lab/mml/kipp/677/jarvis/Software/microgpt310
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import numpy as np
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from scipy import stats
from matplotlib.gridspec import GridSpec
from pymatgen.analysis.structure_matcher import StructureMatcher
from jarvis.core.lattice import get_2d_lattice
import pandas as pd
from jarvis.io.vasp.inputs import Poscar

the_grid = GridSpec(2, 3)
plt.rcParams.update({"font.size": 18})
fig = plt.figure(figsize=(14, 8))

def emd_distance(p, q, bins=None):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    if bins is None:
        bins = np.arange(len(p))
    return wasserstein_distance(bins, bins, u_weights=p, v_weights=q)

def kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p /= np.sum(p)
    q /= np.sum(q)
    return stats.entropy(p, q)

# ------------------------------------------------------------------
#  Load data
# ------------------------------------------------------------------
df = pd.read_csv("AI-AtomGen-prop-dft_3d-test-rmse.csv")
d = []
for _, row in df.iterrows():
    d.append({
        "target":     Poscar.from_string(row["target"].replace("\\n", "\n")).atoms.to_dict(),
        "predicted":  Poscar.from_string(row["prediction"].replace("\\n", "\n")).atoms.to_dict(),
    })

# ------------------------------------------------------------------
#  Basic lattice / angle arrays
# ------------------------------------------------------------------
x_a      = [i["target"]["abc"][0]   for i in d]
y_a      = [i["predicted"]["abc"][0]for i in d]
x_b      = [i["target"]["abc"][1]   for i in d]
y_b      = [i["predicted"]["abc"][1]for i in d]
x_c      = [i["target"]["abc"][2]   for i in d]
y_c      = [i["predicted"]["abc"][2]for i in d]
x_alpha  = [i["target"]["angles"][0]for i in d]
y_alpha  = [i["predicted"]["angles"][0]for i in d]
x_beta   = [i["target"]["angles"][1]for i in d]
y_beta   = [i["predicted"]["angles"][1]for i in d]
x_gamma  = [i["target"]["angles"][2]for i in d]
y_gamma  = [i["predicted"]["angles"][2]for i in d]

# ------------------------------------------------------------------
#  Angle γ hist (fixed: removed stray path)
# ------------------------------------------------------------------
plt.subplot(the_grid[0, 2])
weights_x = np.ones_like(x_gamma) / len(x_gamma) * 100
weights_y = np.ones_like(y_gamma) / len(y_gamma) * 100
plt.hist(x_gamma, bins=np.arange(30, 150, 10),
         weights=weights_x, label="target_gamma",
         color="tab:blue", alpha=0.6)
plt.hist(y_gamma, bins=np.arange(30, 150, 10),
         weights=weights_y, label="predicted_gamma",
         color="plum", alpha=0.6)
plt.xlabel(r"$\gamma$ (°)")
plt.title("(c)")

# ------------------------------------------------------------------
#  Build space-group arrays
# ------------------------------------------------------------------
comp, spg = [], []
samps_spg, samps_comp = [], []
x_spg, y_spg = [], []
x_Z, y_Z     = [], []
x_lat, y_lat = [], []

for i in d:
    a1 = Atoms.from_dict(i["target"])
    a2 = Atoms.from_dict(i["predicted"])
    comp1 = a1.composition.reduced_formula
    comp2 = a2.composition.reduced_formula

    x_Z.append(a1.composition.weight)
    y_Z.append(a2.composition.weight)

    lat_1 = get_2d_lattice(atoms=i["target"])[1]
    lat_2 = get_2d_lattice(atoms=i["predicted"])[1]
    x_lat.append(lat_1)
    y_lat.append(lat_2)

    if comp1 == comp2:
        comp.append(i)
    samps_comp.append(i)

    try:
        spg1 = a1.get_spacegroup[0]
        spg2 = a2.get_spacegroup[0]
        x_spg.append(spg1)
        y_spg.append(spg2)
        if spg1 == spg2:
            spg.append(i)
        samps_spg.append(i)
    except Exception:
        pass

# ------------------------------------------------------------------
#  Space-group histogram (bins fixed to include <50)
# ------------------------------------------------------------------
plt.subplot(the_grid[1, 0])
weights_x = np.ones_like(x_spg) / len(x_spg) * 100
weights_y = np.ones_like(y_spg) / len(y_spg) * 100
bins_spg = np.arange(1, 231, 10)  # 1–230 inclusive, step 10
plt.hist(x_spg, bins=bins_spg, weights=weights_x,
         label="target_spg", color="tab:blue", alpha=0.6)
plt.hist(y_spg, bins=bins_spg, weights=weights_y,
         label="predicted_spg", color="plum", alpha=0.6)
plt.ylabel("Materials dist.")
plt.xlabel("Space-group number")
plt.title("(d)")
# Uncomment the next line if you only want to *display* 1–50
# plt.xlim(1, 50)

# … (rest of the plotting code stays unchanged) …
