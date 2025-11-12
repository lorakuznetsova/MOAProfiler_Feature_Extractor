#!/usr/bin/env python3
"""
prepare_cellpaintssl_dataset.py

Build CellPaintSSL-style dataset.csv from your manifest.csv.

Input (manifest.csv) must contain:
  - imagename      (e.g., images/A01_s5/ch1.tif)
  - moas
  - either broad_sample or perturbation
  - optional: batch, plate, concentration, plate_type

Output columns (fixed order):
  batch, plate, well, well_id, field, perturbation_id, concentration, plate_type, FileName_Merged, target
"""

import argparse
import os
import csv
from pathlib import Path
import re
import pandas as pd
import numpy as np

RE_WELL = re.compile(r'^[A-Pa-p]\d{2}$')
RE_SITE = re.compile(r'^[sS](\d{1,2})$')

def parse_args():
    p = argparse.ArgumentParser(
        prog="prepare_cellpaintssl_dataset.py",
        description="Convert manifest.csv into CellPaintSSL-style dataset.csv"
    )
    p.add_argument("--meta", required=True, help="Path to manifest.csv")
    p.add_argument("--dataset-out", required=True, help="Path to write dataset.csv")
    p.add_argument("--imagename-col", default="imagename", help="Column name for image path (default: imagename)")
    # Defaults applied if columns are absent in the manifest
    p.add_argument("--default-batch", default="BATCH_1")
    p.add_argument("--default-plate", default="PLATE_1")
    p.add_argument("--default-plate-type", default="JUMP_MOA_Plate")
    p.add_argument("--default-concentration", default="0.01")
    return p.parse_args()

def parse_well_field_from_imagename(imagename: str):
    """
    Take last folder and split 'WELL_site':
      images/.../A01_s5/ch1.tif -> A01, s5 -> field f05
    If site is missing, assume s1 -> f01.
    """
    p = Path(str(imagename))
    last = p.parts[-2] if len(p.parts) >= 2 else ""
    if "_" in last:
        well_raw, site_raw = last.split("_", 1)
    else:
        well_raw, site_raw = last, "s1"

    if not RE_WELL.match(well_raw):
        raise ValueError(f"Invalid well code '{well_raw}' parsed from {imagename}")
    m = RE_SITE.match(site_raw)
    if not m:
        raise ValueError(f"Invalid site '{site_raw}' parsed from {imagename} (expected like s5)")
    site_num = int(m.group(1))
    well = well_raw.upper()
    field = f"f{site_num:02d}"
    return well, field

def well_to_well_id(well: str) -> str:
    # 'M11' -> r13c11 (A..P -> 1..16; columns 01..24)
    row_letter = well[0].upper()
    col_num = int(well[1:3])
    row_num = ord(row_letter) - ord('A') + 1
    if row_num < 1 or row_num > 16 or col_num < 1 or col_num > 24:
        raise ValueError(f"Out-of-range well '{well}' (row A-P, col 01-24 expected)")
    return f"r{row_num:02d}c{col_num:02d}"

def main():
    args = parse_args()

    df = pd.read_csv(args.meta, na_values=["NA","NaN","nan","None",""])
    img_col = args.imagename_col

    # Required columns
    must_have = [img_col, "moas"]
    missing = [c for c in must_have if c not in df.columns]
    if missing:
        raise SystemExit(f"manifest missing required columns: {missing}")

    if ("broad_sample" not in df.columns) and ("perturbation" not in df.columns):
        raise SystemExit("manifest must contain either 'broad_sample' or 'perturbation'")

    # Optional columns with defaults
    batch_series = df["batch"].astype(str) if "batch" in df.columns else pd.Series([args.default_batch]*len(df))
    plate_series = df["plate"].astype(str) if "plate" in df.columns else pd.Series([args.default_plate]*len(df))
    conc_series  = df["concentration"].astype(str) if "concentration" in df.columns else pd.Series([args.default_concentration]*len(df))
    ptype_series = df["plate_type"].astype(str) if "plate_type" in df.columns else pd.Series([args.default_plate_type]*len(df))

    header = ["batch","plate","well","well_id","field",
              "perturbation_id","concentration","plate_type","FileName_Merged","target"]

    rows = []
    for idx, row in df.iterrows():
        imagename = str(row[img_col])

        well, field = parse_well_field_from_imagename(imagename)
        well_id = well_to_well_id(well)

        # perturbation_id: prefer broad_sample else perturbation
        if "broad_sample" in df.columns and pd.notna(row.get("broad_sample", np.nan)):
            perturbation_id = str(row["broad_sample"])
        else:
            perturbation_id = str(row.get("perturbation", ""))

        target = str(row["moas"])

        out_row = [
            str(batch_series.iloc[idx]),
            str(plate_series.iloc[idx]),
            well,
            well_id,
            field,
            perturbation_id,
            str(conc_series.iloc[idx]),
            str(ptype_series.iloc[idx]),
            imagename,
            target
        ]
        rows.append(out_row)

    os.makedirs(os.path.dirname(args.dataset_out), exist_ok=True)
    with open(args.dataset_out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    print(f"[OK] Saved {args.dataset_out} with {len(rows)} rows.")

if __name__ == "__main__":
    main()
