#!/usr/bin/env python
"""
write_flowmm_benchmark.py

Create a JARVIS-style benchmark CSV with columns
    id,target,prediction
where `target` and `prediction` are POSCAR strings (newline-escaped).

The ground-truth and predicted structures are read with
flowmm.old_eval.core.get_Crystal_obj_lists(…).

Usage
-----
python write_flowmm_benchmark.py \
    --pt_path  PATH/TO/consolidated_reconstruct.pt \
    --output_csv  flowmm_submission.csv \
    --test_csv  PATH/TO/test.csv \
    [--multi_eval] \
    [--ground_truth_pt PATH/TO/gt.pt] \
    [--dump_json  all_splits.json]

If the Crystal objects do **not** contain a usable identifier, the script
now recovers IDs from the provided test.csv (material_id column).
"""
from __future__ import annotations

import os

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms, pmg_to_atoms
from jarvis.io.vasp.inputs import Poscar
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from tqdm import tqdm

from flowmm.old_eval.core import get_Crystal_obj_lists


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #
def crystal_to_atoms(crys) -> Atoms:
    """FlowMM Crystal ➜ JARVIS Atoms."""
    return pmg_to_atoms(crys.structure)


def build_id(crys, fallback: str) -> str:
    """Return a string ID for this Crystal (path stem, name, …)."""
    for attr in ("id", "name", "path"):
        if hasattr(crys, attr):
            val = getattr(crys, attr)
            if isinstance(val, (str, Path)) and val:
                return Path(val).stem
    return fallback


def poscar_string(atoms: Atoms) -> str:
    """POSCAR string with newlines escaped (like the CDVAE script)."""
    return Poscar(atoms).to_string().replace("\n", r"\n")


def rms_check(target_atoms: list[Atoms], pred_atoms: list[Atoms]) -> float | None:
    """Anonymous RMS distance (debug sanity-check)."""
    matcher = StructureMatcher(stol=0.5, angle_tol=10, ltol=0.3)
    rms_vals = []
    for tgt, pred in tqdm(zip(target_atoms, pred_atoms),
                          total=len(target_atoms),
                          desc="RMS-check"):
        try:
            score, _ = matcher.get_rms_anonymous(
                pred.pymatgen_converter(), tgt.pymatgen_converter()
            )
            if score is not None:
                rms_vals.append(score)
        except Exception:
            # ignore mismatches; just keep going
            pass
    if rms_vals:
        return float(np.mean(rms_vals))
    return None


def load_ids_from_test_csv(csv_path: Path) -> list[str]:
    """Load original IDs from test.csv (prefers 'material_id', falls back to 'id')."""
    df = pd.read_csv(csv_path)
    for col in ("material_id", "id"):
        if col in df.columns:
            return df[col].astype(str).tolist()
    raise ValueError(
        f"Could not find an 'material_id' or 'id' column in {csv_path}"
    )

def read_split(csv_path):
    """Read a JARVIS CSV split → (list[Atoms], list[JID])."""
    df = pd.read_csv(csv_path)
    structs, jids = [], []
    for _, row in df.iterrows():
        atoms = pmg_to_atoms(Structure.from_str(row["cif"], fmt="cif"))
        structs.append(atoms)
        jids.append(row["material_id"])
    return structs, jids


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt_path", required=True,
                        help="FlowMM consolidated_reconstruct.pt (or similar)")
    parser.add_argument("--output_csv", required=True,
                        help="Benchmark CSV to write")
    parser.add_argument("--test_csv", required=True,
                        help="Path to test.csv containing the ground-truth rows "
                             "with a 'material_id' column to recover original IDs")
    parser.add_argument("--multi_eval", action="store_true",
                        help="Set if the .pt was generated with --multi_eval")
    parser.add_argument("--ground_truth_pt",
                        help="Optional separate GT .pt "
                             "(only needed if you passed "
                             "--ground_truth_path to FlowMM)")
    parser.add_argument("--dump_json",
                        help="Optional JSON dump of {id: POSCAR} for debugging")
    args = parser.parse_args()

    # --------------------------------------------------------------------- #
    # Load FlowMM prediction / target structures                            #
    # --------------------------------------------------------------------- #
    print("CWD at start:", os.getcwd())
    pred_crys, gt_crys, _ = get_Crystal_obj_lists(
        Path(args.pt_path),
        args.multi_eval,
        Path(args.ground_truth_pt) if args.ground_truth_pt else None,
    )
    print("CWD before write:", os.getcwd())
    del gt_crys
    
    pred_atoms = [crystal_to_atoms(c) for c in pred_crys]
    tgt_atoms, ids  = read_split(Path(args.test_csv))

    if not (len(pred_atoms) == len(tgt_atoms) == len(ids)):
        raise RuntimeError(
            "Prediction / target / id length mismatch: "
            f"{len(pred_atoms)} preds, {len(tgt_atoms)} targets, {len(ids)} ids. "
            "Ensure test.csv corresponds to the same split and ordering."
        )

    # --------------------------------------------------------------------- #
    # Write CSV                                                             #
    # --------------------------------------------------------------------- #
    with open(args.output_csv, "w") as fh:
        fh.write("id,target,prediction\n")
        for jid, tgt, pred in zip(ids, tgt_atoms, pred_atoms):
            fh.write(f"{jid},{poscar_string(tgt)},{poscar_string(pred)}\n")
    print(f"[✓] CSV written to {args.output_csv} "
          f"({len(pred_atoms)} rows)")

    # Optional JSON dump (handy for diff-ing)
    if args.dump_json:
        dump = {
            jid: {
                "target": poscar_string(tgt),
                "prediction": poscar_string(pred),
            }
            for jid, tgt, pred in zip(ids, tgt_atoms, pred_atoms)
        }
        with open(args.dump_json, "w") as jf:
            json.dump(dump, jf)
        print(f"[✓] JSON dump saved to {args.dump_json}")

    # --------------------------------------------------------------------- #
    # Quick RMS sanity-check                                                #
    # --------------------------------------------------------------------- #
    print("\nRunning anonymous RMS sanity-check …")
    rms = rms_check(tgt_atoms, pred_atoms)
    if rms is not None:
        print(f"[✓] Mean anonymous RMS: {rms:.4f} Å over "
              f"{len(tgt_atoms)} structures")
    else:
        print("[!] RMS could not be computed for any structure")

if __name__ == "__main__":
    main()

