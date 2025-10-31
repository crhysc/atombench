import os
import json
from pathlib import Path
from jarvis.db.jsonutils import loadjson
from jarvis.core.atoms import Atoms
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.special import rel_entr
from scipy.stats import wasserstein_distance
from scipy import stats
import numpy as np
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from jarvis.analysis.structure.spacegroup import Spacegroup3D
import pandas as pd
from jarvis.io.vasp.inputs import Poscar
from functools import lru_cache
from pymatgen.core import Structure

# ── Matplotlib defaults ────────────────────────────────────────────────────
mpl.rcParams['font.family'] = 'serif'
plt.rcParams.update({"font.size": 18})

# ── Figure / GridSpec ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8))
the_grid = GridSpec(2, 3)

# ── Helper functions ──────────────────────────────────────────────────────
@lru_cache(maxsize=4096)
def niggli_params_from_poscar_text(poscar_text: str):
    s = Structure.from_str(poscar_text.replace("\\n", "\n"), fmt="poscar")
    s = s.get_primitive_structure()
    s = s.get_reduced_structure(reduction_algo="niggli")  # Niggli canonical cell
    a, b, c = s.lattice.abc
    alpha, beta, gamma = s.lattice.angles
    return a, b, c, alpha, beta, gamma

def reduced_structure_from_poscar_text(poscar_text: str) -> Structure:
    s = Structure.from_str(poscar_text.replace("\\n", "\n"), fmt="poscar")
    s = s.get_primitive_structure()
    s = s.get_reduced_structure(reduction_algo="niggli")
    return s

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

# NEW: find the directory that contains benchmarks.csv (walk up from CWD)
def find_benchmarks_dir(start: Path) -> Path | None:
    for d in [start, *start.parents]:
        if (d / "benchmarks.csv").is_file():
            return d
    return None

# ── RMSE (StructureMatcher) helper mirroring your snippet ────────────────
def compute_atomgen_rmse(df: pd.DataFrame) -> float:
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rms_vals = []
    for _, mm in df.iterrows():
        try:
            atoms_target = Poscar.from_string((mm["target"].replace("\\n", "\n"))).atoms
            atoms_pred   = Poscar.from_string((mm["prediction"].replace("\\n", "\n"))).atoms
            rms_dist = matcher.get_rms_dist(
                atoms_pred.pymatgen_converter(),
                atoms_target.pymatgen_converter(),
            )
            if rms_dist is not None:
                rms_vals.append(float(rms_dist[0]))  # first element is RMS
        except Exception:
            continue
    if len(rms_vals) == 0:
        return float("nan")
    return round(float(np.mean(rms_vals)), 4)

# ── Load data & extract Niggli-reduced params ─────────────────────────────
df = pd.read_csv("AI-AtomGen-prop-dft_3d-test-rmse.csv")

x_a, x_b, x_c, x_alpha, x_beta, x_gamma = [], [], [], [], [], []
y_a, y_b, y_c, y_alpha, y_beta, y_gamma = [], [], [], [], [], []

for _, row in df.iterrows():
    try:
        ta, tb, tc, tal, tbe, tga = niggli_params_from_poscar_text(row["target"])
        pa, pb, pc, pal, pbe, pga = niggli_params_from_poscar_text(row["prediction"])

        x_a.append(ta);      y_a.append(pa)
        x_b.append(tb);      y_b.append(pb)
        x_c.append(tc);      y_c.append(pc)
        x_alpha.append(tal); y_alpha.append(pal)
        x_beta.append(tbe);  y_beta.append(pbe)
        x_gamma.append(tga); y_gamma.append(pga)
    except Exception:
        continue

# ── Histogram helper (avoid repetition) ───────────────────────────────────
def overlay_hist(ax, x, y, bins, xlabel, title):
    w_x = np.ones_like(x, dtype=float) / max(1, len(x)) * 100
    w_y = np.ones_like(y, dtype=float) / max(1, len(y)) * 100
    ax.hist(x, bins=bins, weights=w_x, alpha=0.6, color="tab:blue", label="target")
    ax.hist(y, bins=bins, weights=w_y, alpha=0.6, color="plum",    label="predicted")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    return ax

# ── (a) a ─────────────────────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[0, 0]),
             x_a, y_a,
             bins=np.arange(2, 7, 0.1),
             xlabel=r"a ($\AA$)",
             title="(a)").set_ylabel("Materials dist.")
plt.legend()

# ── (b) c ─────────────────────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[0, 1]),
             x_c, y_c,
             bins=np.arange(2, 7, 0.1),
             xlabel=r"c ($\AA$)",
             title="(b)")

# ── (c) γ ─────────────────────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[0, 2]),
             x_gamma, y_gamma,
             bins=np.arange(30, 150, 10),
             xlabel=r"$\gamma$ ($^\circ$)",
             title="(c)")

# ── (d)-(f) derive from Niggli-reduced POSCARs directly (no 'records') ───
x_spg, y_spg, x_Z, y_Z = [], [], [], []
x_lat, y_lat = [], []

for _, row in df.iterrows():
    try:
        s_t = reduced_structure_from_poscar_text(row["target"])
        s_p = reduced_structure_from_poscar_text(row["prediction"])

        # molecular weight (amu)
        x_Z.append(s_t.composition.weight)
        y_Z.append(s_p.composition.weight)

        # space group and crystal system via pymatgen
        sga_t = SpacegroupAnalyzer(s_t, symprec=0.1)
        sga_p = SpacegroupAnalyzer(s_p, symprec=0.1)
        x_spg.append(sga_t.get_space_group_number())
        y_spg.append(sga_p.get_space_group_number())
        x_lat.append(sga_t.get_crystal_system())  # "tetragonal", "cubic", ...
        y_lat.append(sga_p.get_crystal_system())
    except Exception:
        continue

# ── (d) Spacegroup histogram (identical bins) ────────────────────────────
ax_spg = plt.subplot(the_grid[1, 0])
bins_spg = np.arange(1, 231, 10)
overlay_hist(ax_spg,
             x_spg, y_spg,
             bins=bins_spg,
             xlabel="Spacegroup number",
             title="(d)").set_ylabel("Materials dist.")

# ── (e) Crystal system counts ────────────────────────────────────────────
lat_order = ["triclinic", "monoclinic", "orthorhombic",
             "tetragonal", "trigonal", "hexagonal", "cubic"]
lat_to_idx = {name: i for i, name in enumerate(lat_order)}
valid_lat = [(lx, ly) for lx, ly in zip(x_lat, y_lat) if lx and ly]
if valid_lat:
    x_lat, y_lat = zip(*valid_lat)
else:
    x_lat, y_lat = [], []
x_lat_counts = np.bincount([lat_to_idx[l] for l in x_lat], minlength=len(lat_order))
y_lat_counts = np.bincount([lat_to_idx[l] for l in y_lat], minlength=len(lat_order))

ax_lat = plt.subplot(the_grid[1, 1])
bar_w = 0.4
pos = np.arange(len(lat_order))

ax_lat.bar(pos, x_lat_counts, width=bar_w, alpha=0.6, label="target",   color="tab:blue")
ax_lat.bar(pos, y_lat_counts, width=bar_w, alpha=0.6, label="predicted", color="plum")

ax_lat.set_xticks(pos)
ax_lat.set_xticklabels((pos + 1).tolist(), rotation=0, ha="center")
ax_lat.set_xlabel("Crystal system number")
ax_lat.set_title("(e)")

# ── (f) Molecular weight ────────────────────────────────────────────────
overlay_hist(plt.subplot(the_grid[1, 2]),
             x_Z, y_Z,
             bins=np.arange(15, 2000, 100),
             xlabel="Weight (AMU)",
             title="(f)")

# ── Final layout & save ──────────────────────────────────────────────────
plt.tight_layout()
bench_lookup = {
    "agpt_benchmark_alex":  "AtomGPT Alexandria",
    "agpt_benchmark_jarvis":"AtomGPT JARVIS",
    "cdvae_benchmark_alex": "CDVAE Alexandria",
    "cdvae_benchmark_jarvis":"CDVAE JARVIS",
    "flowmm_benchmark_alex":"FlowMM Alexandria",
    "flowmm_benchmark_jarvis":"FlowMM JARVIS"
}
fig.subplots_adjust(top=0.88)
plt.suptitle(bench_lookup.get(Path.cwd().name), fontsize=30)

out_png = f"{Path.cwd().name}_distribution.png"
plt.savefig(out_png, format="png")
plt.close()
print(f"✓ saved {out_png}")

# ── Metrics (match reference script exactly) ─────────────────────────────
mae_a     = float(mean_absolute_error(x_a, y_a))
mae_b     = float(mean_absolute_error(x_b, y_b))
mae_c     = float(mean_absolute_error(x_c, y_c))
mae_alpha = float(mean_absolute_error(x_alpha, y_alpha))
mae_beta  = float(mean_absolute_error(x_beta,  y_beta))
mae_gamma = float(mean_absolute_error(x_gamma, y_gamma))

kld_a     = float(kl_divergence(x_a,     y_a))
kld_b     = float(kl_divergence(x_b,     y_b))
kld_c     = float(kl_divergence(x_c,     y_c))
kld_alpha = float(kl_divergence(x_alpha, y_alpha))
kld_beta  = float(kl_divergence(x_beta,  y_beta))
kld_gamma = float(kl_divergence(x_gamma, y_gamma))

rmse_atomgen = compute_atomgen_rmse(df)

metrics = {
    "benchmark_name": Path.cwd().name,
    "KLD": {
        "a":     kld_a,
        "b":     kld_b,
        "c":     kld_c,
        "alpha": kld_alpha,
        "beta":  kld_beta,
        "gamma": kld_gamma,
    },
    "MAE": {
        "average_mae": {
            "a":     mae_a,
            "b":     mae_b,
            "c":     mae_c,
            "alpha": mae_alpha,
            "beta":  mae_beta,
            "gamma": mae_gamma,
        }
    },
    "RMSE": {
        "AtomGen": rmse_atomgen
    }
}

# ── NEW: append metrics_metadata from the directory containing benchmarks.csv ──
benchmarks_dir = find_benchmarks_dir(Path.cwd())
if benchmarks_dir is None:
    print("ℹ️  benchmarks.csv not found in current or parent directories; skipping metrics_metadata.")
else:
    meta_path = benchmarks_dir / "metrics_metadata.json"
    if meta_path.is_file():
        try:
            with open(meta_path, "r") as mf:
                metrics_metadata = json.load(mf)
            # add at the very end of the JSON (insertion-ordered dict)
            metrics["metrics_metadata"] = metrics_metadata
            print(f"✓ appended metrics_metadata from {meta_path}")
        except Exception as e:
            print(f"⚠️  failed to read metrics_metadata.json at {meta_path}: {e}")
    else:
        print(f"ℹ️  {meta_path} not found; skipping metrics_metadata.")

# ── Write metrics.json ────────────────────────────────────────────────────
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("✓ wrote metrics.json")

