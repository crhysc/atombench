import json
import os
from pathlib import Path
import sys

# Force a non‐interactive backend in case you’re headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

ROOT = Path.cwd()  # where we’re running from
print(f"DEBUG: Running script in {ROOT}", file=sys.stderr)

rows = []

for subdir in sorted(ROOT.iterdir()):
    if not subdir.is_dir():
        continue

    metrics_fp = subdir / "metrics.json"
    if not metrics_fp.is_file():
        print(f"⚠️  no metrics.json in {subdir.name} – skipped", file=sys.stderr)
        continue

    with metrics_fp.open() as fh:
        rec = json.load(fh)

    rec.setdefault("benchmark_name", subdir.name)
    flat = pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict()
    rows.append(flat)
    print(f"DEBUG: Loaded metrics for {rec['benchmark_name']}", file=sys.stderr)

if not rows:
    print("ERROR: No metrics.json files found – exiting", file=sys.stderr)
    sys.exit(1)

df = pd.DataFrame(rows)
print("DEBUG: DataFrame shape:", df.shape, file=sys.stderr)
print("DEBUG: DataFrame columns:", df.columns.tolist(), file=sys.stderr)

if "benchmark_name" in df.columns:
    cols = ["benchmark_name"] + [c for c in df.columns if c != "benchmark_name"]
    df = df[cols]
else:
    print("ERROR: 'benchmark_name' column missing", file=sys.stderr)
    sys.exit(1)

params = ['KLD.a','KLD.b','KLD.c','KLD.alpha','KLD.beta','KLD.gamma']
missing = [p for p in params if p not in df.columns]
if missing:
    print("ERROR: Missing metric columns:", missing, file=sys.stderr)
    sys.exit(1)

plot_df = df.set_index('benchmark_name')[params]
print("DEBUG: plot_df shape:", plot_df.shape, file=sys.stderr)
print("DEBUG: plot_df head:\n", plot_df.head(), file=sys.stderr)

ax = plot_df.plot(
    kind='bar',
    figsize=(10, 6),
    edgecolor='k'
)

ax.set_xlabel('Benchmark')
ax.set_ylabel('Metric Value')
ax.set_title('Comparison of Six Metrics Across Benchmarks')
ax.legend(title='Parameter')

plt.xticks(rotation=30, ha='right')
plt.tight_layout()

out_path = ROOT / 'comparison_bar_chart.png'
print(f"DEBUG: Saving figure to {out_path}", file=sys.stderr)
plt.savefig(out_path, dpi=300)
# plt.show()  # removed to avoid hanging
print("DEBUG: Done.", file=sys.stderr)
