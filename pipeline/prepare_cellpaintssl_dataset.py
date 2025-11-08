import os
import csv
import re
import pandas as pd
from pathlib import Path

# Paths (edit as needed)
# TODO: UPDATE EXPERIMENT NAME
input_csv = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Results_MOAProfiler/Pre_Process/compound_images_with_MOA_no_polypharm_well_excluded.csv"
output_csv = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Results_MOAProfiler/Output/dataset.csv"

# Output header (fixed order)
header = [
    "batch", "plate", "well", "well_id", "field",
    "perturbation_id", "concentration", "plate_type", "FileName_Merged", "target"
]

# Defaults (used only if the corresponding columns are absent in the CSV)
default_batch = "BATCH_1"
default_plate = "PLATE_1"
default_plate_type = "JUMP_MOA_Plate"
default_concentration = "0.01"

# Helpers
RE_WELL = re.compile(r'^[A-Pa-p]\d{2}$')
RE_SITE = re.compile(r'^[sS](\d{1,2})$')

def parse_well_field_from_imagename(imagename: str):
    """
    imagename example:
      /.../MOAProfiler_IN/images/M11_s5/ch1.tif
    Take the last folder 'M11_s5' → well='M11', site='s5' → field='f05'
    """
    p = Path(str(imagename))
    last_folder = p.parts[-2] if len(p.parts) >= 2 else ""
    parts = last_folder.split("_")
    if len(parts) != 2:
        raise ValueError(f"Expected 'WELL_site' in last folder, got '{last_folder}' from {imagename}")
    well_raw, site_raw = parts[0], parts[1]

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

# Load MOA Profiler CSV
df = pd.read_csv(input_csv, na_values=["NA", "NaN", "nan", "None", ""]).reset_index(drop=True)
required_cols = [
    "imagename", "broad_sample", "perturbation_t", "cell_type",
    "gene_targets", "control_type", "perturbation", "moas"
]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise SystemExit(f"Input CSV missing required columns: {missing}")

# Pull optional columns if present; otherwise use defaults
batch_series = df["batch"].astype(str) if "batch" in df.columns else pd.Series([default_batch] * len(df))
plate_series = df["plate"].astype(str) if "plate" in df.columns else pd.Series([default_plate] * len(df))
conc_series = df["concentration"].astype(str) if "concentration" in df.columns else pd.Series([default_concentration] * len(df))
ptype_series = df["plate_type"].astype(str) if "plate_type" in df.columns else pd.Series([default_plate_type] * len(df))

# Get output data
rows = []
for idx, row in df.iterrows():
    imagename = str(row["imagename"])

    # Parse well / field from imagename
    well, field = parse_well_field_from_imagename(imagename)
    well_id = well_to_well_id(well)

    # perturbation_id: prefer broad_sample, fallback to perturbation if broad_sample is NA
    perturbation_id = row["broad_sample"] if pd.notna(row["broad_sample"]) else row["perturbation"]
    perturbation_id = str(perturbation_id)

    # target from moas
    target = str(row["moas"])

    # Prepare dataset row
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

# Write output CSV
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Saved {output_csv} with {len(rows)} rows.")
