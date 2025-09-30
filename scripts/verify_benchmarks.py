#!/usr/bin/env python3
import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def canonicalize_target(s: Optional[str]) -> str:
    """Normalize TARGET strings so trivial formatting differences don't matter."""
    if s is None:
        return ""
    s = str(s)
    # Normalize line endings and convert literal \n and \t escapes
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("\\n", "\n").replace("\\t", " ")
    # Collapse all whitespace runs (spaces, tabs, newlines) to single spaces
    tokens = s.split()
    return " ".join(tokens)

def find_benchmark_csv(dir_path: Path) -> Optional[Path]:
    """Return the CSV in dir_path (or its subdirs) that has the expected header."""
    candidates: List[Tuple[float, Path]] = []
    for p, _, files in os.walk(dir_path):
        for f in files:
            if f.lower().endswith(".csv"):
                path = Path(p) / f
                try:
                    with path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
                        reader = csv.DictReader(fh)
                        fields = [fn.strip().lower() for fn in (reader.fieldnames or [])]
                        if {"id", "target", "prediction"}.issubset(set(fields)):
                            candidates.append((path.stat().st_mtime, path))
                except Exception:
                    # Skip unreadable files
                    continue
    if not candidates:
        return None
    # Prefer most recently modified matching CSV
    candidates.sort(key=lambda t: t[0], reverse=True)
    return candidates[0][1]

def load_id_to_target(csv_path: Path) -> Dict[str, str]:
    """Load mapping id -> canonicalized target from a benchmark CSV."""
    mapping: Dict[str, str] = {}
    with csv_path.open("r", newline="", encoding="utf-8", errors="replace") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            _id = str(row["id"]).strip()
            tgt = canonicalize_target(row["target"])
            if _id in mapping and mapping[_id] != tgt:
                raise ValueError(
                    f"Duplicate ID with differing TARGETs in {csv_path}: {_id}"
                )
            mapping[_id] = tgt
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
            print(f"[WARN] No matching CSV in {entry}")
            continue
        groups[dataset].append((entry.name, csv_path))
    return groups

def compare_group(dataset_key: str, items: List[Tuple[str, Path]], show_diff: int) -> bool:
    """Compare ID sets and TARGETs across CSVs for a dataset group."""
    pretty_name = "alexandria" if dataset_key == "alex" else "jarvis"
    if len(items) < 2:
        print(f"[INFO] Only {len(items)} benchmark found for {pretty_name}; cannot verify consensus.")
        return False

    # Load mappings
    loaded = []
    for label, path in items:
        try:
            mapping = load_id_to_target(path)
            loaded.append((label, path, mapping))
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")
            return False

    # Compare ID sets
    id_sets = [set(m.keys()) for _, _, m in loaded]
    all_ids = id_sets[0]
    ids_match = all(s == all_ids for s in id_sets[1:])
    if not ids_match:
        print(f"[MISMATCH] ID sets differ for {pretty_name}.")
        # Show differences per file
        for label, path, mapping in loaded:
            missing = all_ids - set(mapping.keys())
            extra = set(mapping.keys()) - all_ids
            # Recompute baselined against union to show specifics
        union_ids = set().union(*id_sets)
        for label, path, mapping in loaded:
            missing = union_ids - set(mapping.keys())
            extra = set(mapping.keys()) - union_ids  # should be empty
            if missing:
                sample = ", ".join(sorted(list(missing))[:show_diff])
                print(f"  - {label}: missing {len(missing)} IDs (e.g., {sample})")
            if extra:
                sample = ", ".join(sorted(list(extra))[:show_diff])
                print(f"  - {label}: has {len(extra)} unexpected IDs (e.g., {sample})")
        return False

    # Compare TARGETs for each ID
    base_label, base_path, base_map = loaded[0]
    mismatches = []
    for _id in sorted(all_ids):
        base_tgt = base_map[_id]
        for label, path, mapping in loaded[1:]:
            if mapping[_id] != base_tgt:
                mismatches.append((_id, base_label, base_path, label, path))
                if len(mismatches) >= show_diff:
                    break
        if len(mismatches) >= show_diff:
            break

    if mismatches:
        print(f"[MISMATCH] TARGET rows differ for {pretty_name}. Showing up to {show_diff}:")
        for _id, b_label, b_path, l_label, l_path in mismatches:
            print(f"  - ID {_id}: {b_label} ({b_path.name}) != {l_label} ({l_path.name})")
        return False

    print(f"IDs and test set match for {pretty_name}.")
    return True

def main():
    ap = argparse.ArgumentParser(
        description="Verify that benchmark CSVs share the same ID set and TARGET rows per dataset."
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
        print(f"[ERROR] Root path does not exist or is not a directory: {root}")
        raise SystemExit(2)

    groups = group_benchmarks(root)
    any_fail = False
    for dataset in ("jarvis", "alex"):
        items = groups.get(dataset, [])
        # Sort for stable output
        items.sort(key=lambda t: t[0].lower())
        ok = compare_group(dataset, items, args.show_diff)
        any_fail = any_fail or not ok

    raise SystemExit(1 if any_fail else 0)

if __name__ == "__main__":
    main()

