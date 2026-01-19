#!/usr/bin/env python3
"""
check_split_leakage.py

Data leakage + structural uniqueness checks for crystal-structure train/val/test CSV splits.

Supports two layouts:

(A) Root contains dataset subfolders:
    data/
      alexandria/{train,val,test}.csv
      supercon/{train,val,test}.csv

(B) Root is itself a dataset folder:
    alexandria/{train,val,test}.csv

Checks (per dataset):
  1) Disjointness by material_id
  2) Disjointness by normalized CIF text hash
  3) Disjointness by *structure* hash (parsed CIF -> canonical-ish structure -> robust hash)
  4) Within-split duplicates for the same keys
  5) Optional "near-duplicate" leakage using pymatgen StructureMatcher (pruned by composition buckets)

Outputs:
  - stdout summary
  - CSV reports + JSON summary to --out
  - A rebuttal-friendly markdown summary: DISJOINTNESS_SUMMARY.md

Dependencies:
  - pandas, numpy, tqdm
  - pymatgen (required for structural checks; matcher optional but still needs pymatgen)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm


# ---------- CIF / Structure utilities ----------

def normalize_cif_text(cif_text: str) -> str:
    """
    Normalize CIF text for text-hash comparisons:
      - normalize line endings
      - strip trailing whitespace per line
      - strip leading/trailing whitespace overall
    """
    if cif_text is None:
        return ""
    s = str(cif_text).replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n")).strip()
    return s


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()


@dataclass
class StructRecord:
    split: str
    dataset: str
    row_idx: int
    material_id: str
    cif_sha1: str
    struct_hash: Optional[str]
    reduced_formula: Optional[str]
    nsites: Optional[int]
    volume: Optional[float]
    sg_number: Optional[int]
    parse_ok: bool
    parse_error: Optional[str]
    # only used if --matcher is enabled:
    struct_dict: Optional[dict]


def _safe_import_pymatgen():
    try:
        from pymatgen.core import Structure
        from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
        from pymatgen.analysis.structure_matcher import StructureMatcher
        return Structure, SpacegroupAnalyzer, StructureMatcher
    except Exception as e:
        raise RuntimeError(
            "pymatgen is required for structural uniqueness checks. "
            "Install it (e.g., `pip install pymatgen`) and re-run.\n"
            f"Import error: {e}"
        )


def structure_fingerprint_from_cif(
    cif_text: str,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    keep_struct_dict: bool,
) -> Tuple[
    Optional[str], Optional[str], Optional[int], Optional[float], Optional[int],
    bool, Optional[str], Optional[dict]
]:
    """
    Parse CIF -> reduce/standardize -> deterministic site ordering -> hash.

    Returns:
      (struct_hash, reduced_formula, nsites, volume, sg_number, parse_ok, error, struct_dict)
    """
    Structure, SpacegroupAnalyzer, _ = _safe_import_pymatgen()

    try:
        s = Structure.from_str(cif_text, fmt="cif")

        # Niggli reduction: helps canonicalize lattice representation.
        try:
            s = s.get_reduced_structure(reduction_algo="niggli")
        except Exception:
            pass

        sg_number = None
        # Symmetry standardization (spglib-backed) improves robustness across CIF variants.
        try:
            sga = SpacegroupAnalyzer(s, symprec=symprec, angle_tolerance=angle_tolerance)
            sg_number = int(sga.get_space_group_number())
            s = sga.get_primitive_standard_structure(international_monoclinic=True)
        except Exception:
            pass

        # Deterministic ordering of sites
        frac = np.mod(np.array(s.frac_coords, dtype=float), 1.0)
        Z = np.array([int(getattr(site.specie, "Z", site.specie.number)) for site in s.sites], dtype=np.int32)

        order = np.lexsort((
            np.round(frac[:, 2], decimals),
            np.round(frac[:, 1], decimals),
            np.round(frac[:, 0], decimals),
            Z
        ))
        frac = frac[order]
        Z = Z[order]

        lat = np.array(s.lattice.matrix, dtype=float)
        lat_r = np.round(lat, decimals=decimals)
        frac_r = np.round(frac, decimals=decimals)

        payload = (
            lat_r.astype(np.float64).tobytes()
            + Z.astype(np.int32).tobytes()
            + frac_r.astype(np.float64).tobytes()
        )
        struct_hash = hashlib.sha256(payload).hexdigest()

        reduced_formula = s.composition.reduced_formula
        nsites = int(s.num_sites)
        volume = float(s.lattice.volume)

        struct_dict = s.as_dict() if keep_struct_dict else None
        return struct_hash, reduced_formula, nsites, volume, sg_number, True, None, struct_dict

    except Exception as e:
        return None, None, None, None, None, False, repr(e), None


# ---------- I/O and analysis ----------

def read_split_csv(path: Path) -> pd.DataFrame:
    """
    Robust CSV read for CIF fields that may contain embedded newlines.
    Tries fast engine='c' first, falls back to engine='python' if parsing fails.
    """
    read_kwargs = dict(
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        quoting=csv.QUOTE_MINIMAL,
    )
    try:
        df = pd.read_csv(path, engine="c", **read_kwargs)
    except Exception:
        df = pd.read_csv(path, engine="python", **read_kwargs)

    cols = {c.lower(): c for c in df.columns}
    if "material_id" not in cols or "cif" not in cols:
        raise ValueError(
            f"{path} must contain 'material_id' and 'cif' columns. "
            f"Found columns: {list(df.columns)}"
        )
    df = df.rename(columns={cols["material_id"]: "material_id", cols["cif"]: "cif"})
    return df


def discover_datasets(root: Path, splits: Iterable[str]) -> Dict[str, Dict[str, Path]]:
    """
    Accept either:
      (A) root/<dataset_name>/{split}.csv
      (B) root/{split}.csv   (root is itself a dataset folder)

    Returns: dataset_name -> {split -> csv_path}
    """
    root = root.resolve()
    out: Dict[str, Dict[str, Path]] = {}

    # Case (B): root is a dataset folder
    direct: Dict[str, Path] = {}
    for sp in splits:
        f = root / f"{sp}.csv"
        if f.exists():
            direct[sp] = f
    if "train" in direct and "test" in direct:
        out[root.name] = direct
        return out

    # Case (A): root contains dataset subfolders
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        found: Dict[str, Path] = {}
        for sp in splits:
            f = p / f"{sp}.csv"
            if f.exists():
                found[sp] = f
        if "train" in found and "test" in found:
            out[p.name] = found

    if not out:
        raise FileNotFoundError(
            f"No datasets found under {root}. Expected either:\n"
            f"  - {root}/train.csv and {root}/test.csv (root is dataset folder), OR\n"
            f"  - {root}/<dataset>/train.csv and {root}/<dataset>/test.csv (root contains dataset subfolders)."
        )
    return out


def build_records_for_split(
    dataset: str,
    split: str,
    csv_path: Path,
    symprec: float,
    angle_tolerance: float,
    decimals: int,
    keep_struct_dict: bool,
) -> List[StructRecord]:
    df = read_split_csv(csv_path)

    records: List[StructRecord] = []
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"Parsing {dataset}/{split}", leave=False):
        mid = str(row["material_id"])
        cif_text = normalize_cif_text(row["cif"])
        cif_sha = sha1_hex(cif_text)

        struct_hash, red_formula, nsites, vol, sg, ok, err, sdict = structure_fingerprint_from_cif(
            cif_text=cif_text,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
            decimals=decimals,
            keep_struct_dict=keep_struct_dict,
        )

        records.append(
            StructRecord(
                split=split,
                dataset=dataset,
                row_idx=int(i) if isinstance(i, (int, np.integer)) else int(getattr(i, "item", lambda: 0)()),
                material_id=mid,
                cif_sha1=cif_sha,
                struct_hash=struct_hash,
                reduced_formula=red_formula,
                nsites=nsites,
                volume=vol,
                sg_number=sg,
                parse_ok=ok,
                parse_error=err,
                struct_dict=sdict,
            )
        )
    return records


def df_from_records(recs: List[StructRecord]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in recs])


def duplicates_report(df: pd.DataFrame, key: str) -> pd.DataFrame:
    x = df.copy()
    x = x[~x[key].isna() & (x[key] != "")]
    dup = x[x.duplicated(key, keep=False)].sort_values(by=[key, "split", "material_id"])
    return dup


def overlap_report(dfA: pd.DataFrame, dfB: pd.DataFrame, key: str) -> pd.DataFrame:
    a = dfA[~dfA[key].isna() & (dfA[key] != "")]
    b = dfB[~dfB[key].isna() & (dfB[key] != "")]
    common = set(a[key]).intersection(set(b[key]))
    if not common:
        return pd.DataFrame(columns=list(dfA.columns) + ["_side"])
    outA = a[a[key].isin(common)].copy()
    outA["_side"] = "A"
    outB = b[b[key].isin(common)].copy()
    outB["_side"] = "B"
    out = pd.concat([outA, outB], ignore_index=True).sort_values(by=[key, "_side", "material_id"])
    return out


def run_structurematcher_near_duplicates(
    dfA: pd.DataFrame,
    dfB: pd.DataFrame,
    key_cols: Tuple[str, str],
    ltol: float,
    stol: float,
    angle_tol: float,
    vol_rel_tol: float,
    max_bucket: int,
) -> pd.DataFrame:
    """
    Expensive check: find matches between dfA and dfB using StructureMatcher,
    pruned by buckets based on reduced_formula and nsites.

    Requires struct_dict present (enable by running with --matcher).
    """
    Structure, _, StructureMatcher = _safe_import_pymatgen()
    sm = StructureMatcher(
        ltol=ltol,
        stol=stol,
        angle_tol=angle_tol,
        primitive_cell=True,
        scale=True,
        attempt_supercell=False,
        allow_subset=False,
    )

    A = dfA[dfA["parse_ok"] & dfA["struct_dict"].notna()].copy()
    B = dfB[dfB["parse_ok"] & dfB["struct_dict"].notna()].copy()

    bucketA: Dict[Tuple[str, int], List[int]] = {}
    bucketB: Dict[Tuple[str, int], List[int]] = {}

    for idx, r in A.iterrows():
        if not r["reduced_formula"] or not r["nsites"]:
            continue
        bucketA.setdefault((r["reduced_formula"], int(r["nsites"])), []).append(idx)

    for idx, r in B.iterrows():
        if not r["reduced_formula"] or not r["nsites"]:
            continue
        bucketB.setdefault((r["reduced_formula"], int(r["nsites"])), []).append(idx)

    hits = []
    keys = set(bucketA).intersection(bucketB)

    for k in tqdm(sorted(keys), desc=f"StructureMatcher buckets {key_cols[0]} vs {key_cols[1]}", leave=False):
        idsA = bucketA[k]
        idsB = bucketB[k]
        if len(idsA) * len(idsB) > max_bucket:
            continue

        structsA = []
        metaA = []
        for idx in idsA:
            row = A.loc[idx]
            s = Structure.from_dict(row["struct_dict"])
            structsA.append(s)
            metaA.append(row)

        structsB = []
        metaB = []
        for idx in idsB:
            row = B.loc[idx]
            s = Structure.from_dict(row["struct_dict"])
            structsB.append(s)
            metaB.append(row)

        for sA, rA in zip(structsA, metaA):
            vA = float(rA["volume"]) if rA["volume"] is not None else None
            for sB, rB in zip(structsB, metaB):
                vB = float(rB["volume"]) if rB["volume"] is not None else None
                if vA and vB:
                    rel = abs(vA - vB) / max(vA, vB)
                    if rel > vol_rel_tol:
                        continue

                if sm.fit(sA, sB):
                    rms, max_dist = sm.get_rms_dist(sA, sB)
                    hits.append(
                        {
                            "reduced_formula": k[0],
                            "nsites": k[1],
                            "A_material_id": rA["material_id"],
                            "B_material_id": rB["material_id"],
                            "A_struct_hash": rA["struct_hash"],
                            "B_struct_hash": rB["struct_hash"],
                            "rms_dist": float(rms) if rms is not None else None,
                            "max_dist": float(max_dist) if max_dist is not None else None,
                        }
                    )

    return pd.DataFrame(hits)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Check train/val/test disjointness and structural uniqueness for CIF-in-CSV splits."
    )
    ap.add_argument("--data-root", type=str, required=True, help="Path to dataset root (see docstring for layouts).")
    ap.add_argument("--splits", type=str, default="train,val,test", help="Comma-separated splits to look for.")
    ap.add_argument("--datasets", type=str, default="", help="Comma-separated dataset folder names to include (default: auto-discover).")
    ap.add_argument("--out", type=str, default="leakage_reports", help="Output directory for reports.")
    ap.add_argument("--symprec", type=float, default=1e-2, help="symprec for SpacegroupAnalyzer standardization.")
    ap.add_argument("--angle-tol", type=float, default=5.0, help="angle_tolerance for SpacegroupAnalyzer.")
    ap.add_argument("--decimals", type=int, default=6, help="Rounding decimals for structure hash payload.")
    ap.add_argument("--compare-across-datasets", action="store_true", help="Also compare identical splits across dataset folders.")
    ap.add_argument("--matcher", action="store_true", help="Run StructureMatcher-based near-duplicate checks (slower).")

    ap.add_argument("--ltol", type=float, default=0.2, help="StructureMatcher ltol.")
    ap.add_argument("--stol", type=float, default=0.3, help="StructureMatcher stol.")
    ap.add_argument("--matcher-angle-tol", type=float, default=5.0, help="StructureMatcher angle_tol.")
    ap.add_argument("--vol-rel-tol", type=float, default=0.02, help="Relative volume tolerance pre-filter for matcher.")
    ap.add_argument("--max-bucket", type=int, default=2000, help="Skip matcher buckets with pair count > this.")

    args = ap.parse_args()

    root = Path(args.data_root).expanduser().resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    outdir = Path(args.out).expanduser().resolve()
    ensure_dir(outdir)

    discovered = discover_datasets(root, splits=splits)

    if args.datasets.strip():
        wanted = {d.strip() for d in args.datasets.split(",") if d.strip()}
        discovered = {k: v for k, v in discovered.items() if k in wanted}
        if not discovered:
            raise FileNotFoundError(f"No matching datasets found for --datasets={args.datasets}")

    all_frames: Dict[str, pd.DataFrame] = {}
    summary = {"data_root": str(root), "datasets": {}}

    for dataset, split_map in tqdm(discovered.items(), desc="Datasets"):
        keep_struct_dict = bool(args.matcher)
        recs_all: List[StructRecord] = []

        for split, csv_path in split_map.items():
            recs = build_records_for_split(
                dataset=dataset,
                split=split,
                csv_path=csv_path,
                symprec=args.symprec,
                angle_tolerance=args.angle_tol,
                decimals=args.decimals,
                keep_struct_dict=keep_struct_dict,
            )
            recs_all.extend(recs)

        df = df_from_records(recs_all)
        all_frames[dataset] = df

        n_total = len(df)
        n_ok = int(df["parse_ok"].sum())
        n_fail = n_total - n_ok

        summary["datasets"][dataset] = {
            "rows_total": n_total,
            "rows_parse_ok": n_ok,
            "rows_parse_fail": n_fail,
            "splits_present": sorted(df["split"].unique().tolist()),
        }

        # Write per-dataset records (drop struct_dict to keep CSV clean)
        df_no_struct = df.drop(columns=["struct_dict"], errors="ignore")
        df_no_struct.to_csv(outdir / f"{dataset}__records.csv", index=False)

        # Within-split duplicates
        for split in sorted(df["split"].unique()):
            dsplit = df[df["split"] == split].copy()
            for key in ["material_id", "cif_sha1", "struct_hash"]:
                dup = duplicates_report(dsplit, key=key)
                if len(dup):
                    dup.drop(columns=["struct_dict"], errors="ignore").to_csv(
                        outdir / f"{dataset}__dup_within_{split}__by_{key}.csv", index=False
                    )

        # Pairwise split overlaps
        split_list = sorted(df["split"].unique())
        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                a, b = split_list[i], split_list[j]
                dfA = df[df["split"] == a].copy()
                dfB = df[df["split"] == b].copy()

                for key in ["material_id", "cif_sha1", "struct_hash"]:
                    ov = overlap_report(dfA, dfB, key=key)
                    if len(ov):
                        ov.drop(columns=["struct_dict"], errors="ignore").to_csv(
                            outdir / f"{dataset}__overlap_{a}_vs_{b}__by_{key}.csv", index=False
                        )

                if args.matcher:
                    near = run_structurematcher_near_duplicates(
                        dfA=dfA,
                        dfB=dfB,
                        key_cols=(f"{dataset}/{a}", f"{dataset}/{b}"),
                        ltol=args.ltol,
                        stol=args.stol,
                        angle_tol=args.matcher_angle_tol,
                        vol_rel_tol=args.vol_rel_tol,
                        max_bucket=args.max_bucket,
                    )
                    if len(near):
                        near.to_csv(outdir / f"{dataset}__neardup_{a}_vs_{b}__structurematcher.csv", index=False)

    # Optional: cross-dataset comparisons (same split name)
    if args.compare_across_datasets and len(all_frames) > 1:
        ds_names = sorted(all_frames.keys())
        for i in range(len(ds_names)):
            for j in range(i + 1, len(ds_names)):
                A, B = ds_names[i], ds_names[j]
                dfA = all_frames[A]
                dfB = all_frames[B]

                common_splits = sorted(set(dfA["split"]).intersection(set(dfB["split"])))
                for sp in common_splits:
                    a = dfA[dfA["split"] == sp].copy()
                    b = dfB[dfB["split"] == sp].copy()

                    for key in ["material_id", "cif_sha1", "struct_hash"]:
                        ov = overlap_report(a, b, key=key)
                        if len(ov):
                            ov.drop(columns=["struct_dict"], errors="ignore").to_csv(
                                outdir / f"cross__{A}_vs_{B}__{sp}__overlap_by_{key}.csv", index=False
                            )

                    if args.matcher:
                        near = run_structurematcher_near_duplicates(
                            dfA=a,
                            dfB=b,
                            key_cols=(f"{A}/{sp}", f"{B}/{sp}"),
                            ltol=args.ltol,
                            stol=args.stol,
                            angle_tol=args.matcher_angle_tol,
                            vol_rel_tol=args.vol_rel_tol,
                            max_bucket=args.max_bucket,
                        )
                        if len(near):
                            near.to_csv(
                                outdir / f"cross__{A}_vs_{B}__{sp}__neardup_structurematcher.csv", index=False
                            )

    # Paper-friendly markdown summary
    paper_lines = []
    paper_lines.append("# Split disjointness / leakage audit\n")
    paper_lines.append(f"- Data root: `{root}`")
    paper_lines.append(f"- Datasets scanned: {', '.join(sorted(all_frames.keys()))}")
    paper_lines.append(f"- Structure standardization: SpacegroupAnalyzer(symprec={args.symprec}, angle_tolerance={args.angle_tol})")
    paper_lines.append(f"- Structure hash rounding: {args.decimals} decimals\n")

    for dataset, df in all_frames.items():
        paper_lines.append(f"## {dataset}")
        for split in sorted(df["split"].unique()):
            sub = df[df["split"] == split]
            paper_lines.append(
                f"- {split}: {len(sub)} rows (parse ok: {int(sub['parse_ok'].sum())}, parse fail: {int((~sub['parse_ok']).sum())})"
            )

        split_list = sorted(df["split"].unique())
        for i in range(len(split_list)):
            for j in range(i + 1, len(split_list)):
                a, b = split_list[i], split_list[j]
                Aset = df[df["split"] == a]
                Bset = df[df["split"] == b]

                for key in ["material_id", "cif_sha1", "struct_hash"]:
                    common = set(Aset[key].dropna()) & set(Bset[key].dropna())
                    common.discard("")
                    paper_lines.append(f"- Overlap {a} vs {b} by {key}: {len(common)}")

                if args.matcher:
                    near_path = outdir / f"{dataset}__neardup_{a}_vs_{b}__structurematcher.csv"
                    n_near = 0
                    if near_path.exists():
                        try:
                            # header line excluded
                            n_near = sum(1 for _ in open(near_path, "r", encoding="utf-8")) - 1
                        except Exception:
                            n_near = 0
                    paper_lines.append(f"- Near-duplicate matches {a} vs {b} (StructureMatcher): {max(n_near, 0)}")

        paper_lines.append("")

    (outdir / "DISJOINTNESS_SUMMARY.md").write_text("\n".join(paper_lines) + "\n", encoding="utf-8")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print("Leakage / uniqueness audit complete.")
    print(f"Reports written to: {outdir}")
    print("Key artifact: DISJOINTNESS_SUMMARY.md")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
