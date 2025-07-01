import json
import os
from pathlib import Path

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

    # ── 1. rename the PDF (if present) ────────────────────────────────────
    pdf_fp = subdir / "distribution.pdf"
    if pdf_fp.exists():
        new_name = f"distribution_{subdir.name}.pdf"
        pdf_fp.rename(subdir / new_name)

    # ── 2. load & flatten JSON metrics ───────────────────────────────────
    with metrics_fp.open() as fh:
        rec = json.load(fh)

    # fallback benchmark name if field absent / mis-spelled
    rec.setdefault("benchmark_name", subdir.name)

    # pandas.json_normalize gives us a flat dict:
    flat = pd.json_normalize(rec, sep=".", max_level=3).iloc[0].to_dict()
    rows.append(flat)

# ── 3. build DataFrame & column ordering ─────────────────────────────────
if not rows:
    raise SystemExit("No metrics.json files found – nothing to do.")

df = pd.DataFrame(rows)

# move benchmark_name to the first column if it’s present
if "benchmark_name" in df.columns:
    cols = ["benchmark_name"] + [c for c in df.columns if c != "benchmark_name"]
    df = df[cols]

# ── 4. save tidy CSV ─────────────────────────────────────────────────────
out_path = ROOT / "epic_metrics.csv"
df.to_csv(out_path, index=False)
print(f"✓ consolidated table written to {out_path.resolve()}")
