#!/usr/bin/env python3
import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
from jarvis.io.vasp.inputs import Poscar

# ---------- POSCAR parsing & canonicalization ----------

def _unescape_poscar_text(s: str) -> str:
    """Turn CSV-literal text into a parseable POSCAR string."""
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n").replace("\\t", " ")
    return s.strip()

def _parse_poscar_atoms(poscar_text: str):
    """Parse POSCAR text -> jarvis.core.atoms.Atoms (or None on failure)."""
    try:
        p = Poscar.from_string(_unescape_poscar_text(poscar_text))
        return p.atoms
    except Exception:
        return None

def _get_species_and_frac(atoms):
    """
    Extract (species_list, frac_coords ndarray) from a JARVIS Atoms object.
    Tries multiple attribute names defensively.
    """
    # species
    species = getattr(atoms, "elements", None)
    if species is None:
        try:
            species = [s.specie for s in atoms]
        except Exception:
            raise ValueError("Could not read species from Atoms.")

    # fractional coordinates
    frac = getattr(atoms, "frac_coords", None)
    if frac is None:
        lat = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
        cart = np.asarray(getattr(atoms, "coords", None), dtype=float)
        if lat is None or cart is None:
            raise ValueError("Could not read coordinates from Atoms.")
        inv_lat_T = np.linalg.inv(lat.T)
        frac = cart @ inv_lat_T

    return list(species), np.asarray(frac, dtype=float)

def _structure_signature(atoms, decimals: int = 1):
    """
    Order-insensitive signature with rounding:
    - Lattice matrix rounded to 'decimals'
    - Fractional coords rounded to 'decimals'
    - Species order ignored (coords grouped per species & sorted)
    Returns a hashable tuple.
    """
    if atoms is None:
        return ("__PARSE_FAILED__",)

    lat = np.asarray(getattr(atoms, "lattice_mat", None), dtype=float)
    if lat is None or lat.shape != (3, 3):
        return ("__BAD_LATTICE__",)
    lat_r = np.round(lat, decimals)

    species, frac = _get_species_and_frac(atoms)
    if len(species) != len(frac):
        return ("__LEN_MISMATCH__",)

    frac_r = np.round(frac, decimals)

    buckets: Dict[str, List[Tuple[float, float, float]]] = {}
    for sp, xyz in zip(species, frac_r):
        tpl = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
        buckets.setdefault(str(sp), []).append(tpl)

    for sp in buckets:
        buckets[sp].sort()

    lat_sig = tuple(lat_r.reshape(-1).tolist())
    species_sig = tuple(sorted((sp, tuple(coords)) for sp, coords in buckets.items()))
    return ("OK", lat_sig, species_sig)

# ---------- File-discovery & grouping ----------

def canonicalize_target_struct(s: Optional[str]) -> Tuple:
    """Convert TARGET string to an order-insensitive rounded structural signature."""
    if s is None or str(s).strip() == "":
        return ("__EMPTY__",)
    atoms = _parse_poscar_atoms(str(s))
    return _structure_signature(atoms, decimals=1)

def find_benchmark_csv(dir_path: Path) -> Optional[Path]:
    """Return the newest CSV in dir_path (or subdirs) that has id/target/prediction columns."""
    candidates: List[Tuple[float, Path]] = []
    for p, _, files in os.walk(dir_path):
        for f in files:
            fl = f.lower()
            if not fl.endswith(".csv"):
                continue
            # Skip misses CSVs here; they are handled separately.
            if fl.endswith(".misses.csv"):
                continue
            path = Path(p) / f
            try:
                with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                    reader = csv.DictReader(fh)
                    fields = [fn.strip().lower() for fn in (reader.fieldnames or [])]
                    if {"id", "target", "prediction"}.issubset(set(fields)):
                        candidates.append((path.stat().st_mtime, path))
            except Exception:
                continue
    if not candidates:
        return None
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]

def find_misses_csv_for_benchmark(benchmark_csv_path: Path) -> Optional[Path]:
    """
    For benchmark CSV .../<name>.csv, look for sibling .../<name>.misses.csv.
    Returns path if it exists, else None.
    """
    candidate = benchmark_csv_path.with_name(
        f"{benchmark_csv_path.stem}.misses{benchmark_csv_path.suffix}"
    )
    return candidate if candidate.exists() and candidate.is_file() else None

def load_id_set(csv_path: Path) -> Set[str]:
    """
    Load a set of IDs from any CSV that has an 'id' column.
    Used for *.misses.csv files.
    """
    ids: Set[str] = set()
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        fieldnames = [fn.strip().lower() for fn in (reader.fieldnames or [])]
        if "id" not in fieldnames:
            raise ValueError(f"No 'id' column found in {csv_path}")
        # Map normalized lowercase headers back to actual row keys
        keymap = {fn.strip().lower(): fn for fn in (reader.fieldnames or [])}
        id_key = keymap["id"]
        for row in reader:
            _id = str(row.get(id_key, "")).strip()
            if _id:
                ids.add(_id)
    return ids

def load_id_to_target(csv_path: Path) -> Dict[str, Tuple]:
    """
    Load mapping id -> structural signature (rounded, species-order-insensitive).
    NOTE: If the same ID appears with differing TARGETs, we WARN and keep the first.
    """
    mapping: Dict[str, Tuple] = {}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            _id = str(row["id"]).strip()
            tgt_sig = canonicalize_target_struct(row["target"])
            if _id in mapping and mapping[_id] != tgt_sig:
                print(f"[WARN] Duplicate ID with differing TARGET in {csv_path.name}: {_id} "
                      f"(keeping first occurrence)", file=sys.stderr)
                continue
            mapping.setdefault(_id, tgt_sig)
    return mapping

def group_benchmarks(root: Path):
    """Group benchmark directories by dataset tag ('jarvis' or 'alex')."""
    groups = {"jarvis": [], "alex": []}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name.lower()
        dataset = None
        if "jarvis" in name:
            dataset = "jarvis"
        elif "alexandria" in name or "alex" in name:
            dataset = "alex"
        if dataset is None:
            continue
        csv_path = find_benchmark_csv(entry)
        if csv_path is None:
            print(f"[WARN] No matching CSV in {entry}", file=sys.stderr)
            continue
        groups[dataset].append((entry.name, csv_path))
    return groups

def compare_group(dataset_key: str, items: List[Tuple[str, Path]], show_diff: int) -> bool:
    """
    Compare ID sets across CSVs for a dataset group.
    Return False **only if ID sets mismatch** (unless all missing IDs are accounted for
    in the corresponding *.misses.csv files).
    Structural TARGET differences are reported as WARNINGS but do not cause failure.
    """
    pretty_name = "alexandria" if dataset_key == "alex" else "jarvis"
    if len(items) < 2:
        print(f"[INFO] Only {len(items)} benchmark found for {pretty_name}; skipping consensus check.")
        return True  # not an ID mismatch

    # Load mappings (+ optional misses IDs)
    loaded = []
    for label, path in items:
        try:
            mapping = load_id_to_target(path)
            misses_path = find_misses_csv_for_benchmark(path)
            misses_ids: Set[str] = set()
            if misses_path is not None:
                try:
                    misses_ids = load_id_set(misses_path)
                except Exception as e:
                    print(f"[WARN] Failed to read misses CSV {misses_path}: {e}", file=sys.stderr)
            loaded.append((label, path, mapping, misses_path, misses_ids))
        except Exception as e:
            # Per request: only fail on ID mismatch; treat read errors as warnings.
            print(f"[WARN] Failed to read {path}: {e}", file=sys.stderr)
            return True

    # Compare ID sets, allowing missing IDs if listed in that benchmark's *.misses.csv
    id_sets = [set(m.keys()) for _, _, m, _, _ in loaded]
    union_ids = set().union(*id_sets)

    any_unaccounted_missing = False
    any_accounted_missing = False

    # First pass: determine whether mismatches are fully accounted for by misses files
    for label, path, mapping, misses_path, misses_ids in loaded:
        present_ids = set(mapping.keys())
        missing = union_ids - present_ids
        if not missing:
            continue
        covered = missing & misses_ids
        uncovered = missing - misses_ids
        if covered:
            any_accounted_missing = True
        if uncovered:
            any_unaccounted_missing = True

    if any_unaccounted_missing:
        print(f"[MISMATCH] ID sets differ for {pretty_name}.")
        for label, path, mapping, misses_path, misses_ids in loaded:
            present_ids = set(mapping.keys())
            missing = union_ids - present_ids
            covered = missing & misses_ids
            uncovered = missing - misses_ids
            extra = present_ids - union_ids  # should be empty

            if uncovered:
                sample = ", ".join(sorted(list(uncovered))[:show_diff])
                print(f"  - {label}: missing {len(uncovered)} IDs NOT accounted for in misses CSV "
                      f"(e.g., {sample})")
            if covered:
                sample = ", ".join(sorted(list(covered))[:show_diff])
                if misses_path is not None:
                    print(f"  - {label}: missing {len(covered)} IDs accounted for by "
                          f"{misses_path.name} (e.g., {sample})")
                else:
                    print(f"  - {label}: missing {len(covered)} IDs accounted for by misses CSV "
                          f"(e.g., {sample})")
            if extra:
                sample = ", ".join(sorted(list(extra))[:show_diff])
                print(f"  - {label}: has {len(extra)} unexpected IDs (e.g., {sample})")
        return False  # <-- only hard failure condition (unaccounted missing IDs)

    # If there were missing IDs but all were accounted for in misses CSVs, warn-only
    if any_accounted_missing:
        print(f"[INFO] ID set differences for {pretty_name} are fully accounted for by *.misses.csv files.")

    # Compare TARGET structures for awareness (WARN only) on common IDs across benchmark CSVs
    common_ids = set.intersection(*id_sets) if id_sets else set()
    base_label, base_path, base_map, _, _ = loaded[0]
    mismatches = []
    for _id in sorted(common_ids):
        base_sig = base_map[_id]
        for label, path, mapping, _, _ in loaded[1:]:
            if mapping.get(_id) != base_sig:
                mismatches.append((_id, base_label, base_path, label, path))
                if len(mismatches) >= show_diff:
                    break
        if len(mismatches) >= show_diff:
            break

    if mismatches:
        print(f"[WARN] TARGET structures differ for {pretty_name} "
              f"(rounded to 1 decimal, species-order-insensitive). Showing up to {show_diff}:",
              file=sys.stderr)
        for _id, b_label, b_path, l_label, l_path in mismatches:
            print(f"  - ID {_id}: {b_label} ({b_path.name}) != {l_label} ({l_path.name})",
                  file=sys.stderr)
    else:
        # Keep original success wording when exact IDs match; otherwise note misses accounting.
        exact_ids_match = all(s == id_sets[0] for s in id_sets[1:]) if id_sets else True
        if exact_ids_match:
            print(f"[OK] IDs and TARGET structures match for {pretty_name} "
                  f"(rounded to 1 decimal, species-order-insensitive).")
        else:
            print(f"[OK] TARGET structures match on common IDs for {pretty_name} "
                  f"(rounded to 1 decimal, species-order-insensitive); "
                  f"ID differences accounted for by *.misses.csv files.")

    return True  # success since IDs matched or were fully accounted for by misses CSVs

def main():
    ap = argparse.ArgumentParser(
        description=("Verify that benchmark CSVs share the same ID set. "
                     "Only ID mismatches cause a non-zero exit. "
                     "TARGET structural differences are reported as warnings.")
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("job_runs"),
        help="Path to the job_runs directory (default: ./job_runs)",
    )
    ap.add_argument(
        "--show-diff",
        type=int,
        default=5,
        help="Max number of example differences to display (default: 5)",
    )
    args = ap.parse_args()

    root = args.root
    if not root.exists() or not root.is_dir():
        print(f"[WARN] Root path does not exist or is not a directory: {root}", file=sys.stderr)
        raise SystemExit(0)  # not an ID mismatch

    groups = group_benchmarks(root)
    id_mismatch = False
    for dataset in ("jarvis", "alex"):
        items = groups.get(dataset, [])
        items.sort(key=lambda t: t[0].lower())
        ok = compare_group(dataset, items, args.show_diff)
        if not ok:
            id_mismatch = True

    raise SystemExit(1 if id_mismatch else 0)

if __name__ == "__main__":
    main()
