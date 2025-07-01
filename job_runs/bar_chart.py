import json
import os
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

ROOT = Path.cwd()                # script is run from job_runs/

rows = []

for subdir in sorted(ROOT.iterdir()):
    if not subdir.is_dir():
        continue

    metrics_fp = subdir / "metrics.json"
    if not metrics_fp.is_file():
        print(f"⚠️  no metrics.json in {subdir.name} – skipped")
        continue

    # ── 1. load & flatten JSON metrics ───────────────────────────────────
    with metrics_fp.open() as fh:
        rec = json.load(fh)

    # fallback benchmark name if field absent / mis-spelled
    rec.setdefault("benchmark_name", subdir.name)

    # pandas.json_normalize gives us a flat dict:
    flat = pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict()
    rows.append(flat)

# ── 2. build DataFrame & column ordering ─────────────────────────────────
if not rows:
    raise SystemExit("No metrics.json files found – nothing to do.")

df = pd.DataFrame(rows)

# move benchmark_name to the first column if it’s present
if "benchmark_name" in df.columns:
    cols = ["benchmark_name"] + [c for c in df.columns if c != "benchmark_name"]
    df = df[cols]

# ── 3. create bar chart ─────────────────────────────────────────────────────
params = ['KLD.a','KLD.b','KLD.c','KLD.alpha','KLD.beta','KLD.gamma']

# set the benchmark names as the row index
plot_df = df.set_index('benchmark_name')[params]

# plot grouped bars: each row → one group, each column → one color
ax = plot_df.plot(
    kind='bar',
    figsize=(10, 6),
    edgecolor='k'    # optional: give each bar a black outline
)

ax.set_xlabel('Benchmark')       # the index name on the x-axis
ax.set_ylabel('Metric Value')
ax.set_title('Comparison of Six Metrics Across Benchmarks')
ax.legend(title='Parameter')

plt.xticks(rotation=30, ha='right')   # tilt long names if needed
plt.tight_layout()
plt.savefig('comparison_bar_chart.png', dpi=300)
plt.show()