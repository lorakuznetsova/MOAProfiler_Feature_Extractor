import os
import pandas as pd
import docx
import pickle
import re

# --- Input paths ---
# TODO: UPDATE BEFORE EVERY RUN!!!
image_base_dir = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Results_MOAProfiler/Pre_Process/Images"
docx_path = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Source_Data/MOA_JUMP_Platemap.docx"

# --- Output base paths ---
# (1) User working folder
# TODO: UPDATE BEFORE EVERY RUN!!!
user_out_base = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Results_MOAProfiler/Pre_Process"

# (2) Repo folders
repo_csv_out = "/home/usr/Cell_Painting_Tools/MOAProfiler/Repo/csvs/JUMP1"
repo_pkl_out = "/home/usr/Cell_Painting_Tools/MOAProfiler/Repo/pickles/JUMP1"

# --- Final output filenames ---
filename_csv = "compound_images_with_MOA_no_polypharm_well_excluded.csv"
filename_pkl = "label_index_map_from_compound_images_with_MOA_no_polypharm.csv.pkl"

# --- Ensure repo folders exist ---
os.makedirs(repo_csv_out, exist_ok=True)
os.makedirs(repo_pkl_out, exist_ok=True)

# --- Helper to load DOCX table ---
def read_platemap_from_docx(path):
    doc = docx.Document(path)
    rows = []
    for table in doc.tables:
        for i, row in enumerate(table.rows):
            cells = [cell.text.strip() for cell in row.cells]
            if i == 0:
                header = cells
            else:
                rows.append(dict(zip(header, cells)))
    return pd.DataFrame(rows)

# --- Load and clean platemap ---
platemap = read_platemap_from_docx(docx_path)
platemap['well'] = platemap['well'].str.upper().str.strip()

print("Platemap columns:", platemap.columns.tolist())  # For manual check

# --- Generate metadata records ---
records = []
for folder in os.listdir(image_base_dir):
    folder_path = os.path.join(image_base_dir, folder)
    if not os.path.isdir(folder_path):
        continue

    m = re.match(r"([A-P]\d\d)_s(\d+)", folder)
    if not m:
        print(f"⚠️ Skipping malformed folder: {folder}")
        continue

    well, site = m.groups()
    site = int(site)

    row = platemap[platemap['well'] == well]
    if row.empty:
        print(f"⚠️ No metadata for well: {well} (folder: {folder})")
        continue

    compound = row.iloc[0]['compound']
    moa = row.iloc[0]['moa']
    ch1_path = os.path.join(image_base_dir, folder, "ch1.tif")
    control_type = "negative" if compound.strip().lower() == "dmso" else "treatment"

    records.append({
        'imagename': ch1_path,
        'broad_sample': compound,
        'perturbation_t': "NA",
        'cell_type': "U2OS",
        'gene_targets': "NA",
        'control_type': control_type,
        'perturbation': compound,
        'moas': moa
    })

# --- Convert to DataFrame ---
df = pd.DataFrame(records)

# --- Save to both locations ---
for out_base in [user_out_base, repo_csv_out]:
    csv_path = os.path.join(out_base, filename_csv)
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved metadata CSV: {csv_path} ({len(df)} entries)")

# --- Create and save label index map ---
moas = sorted(df["moas"].dropna().unique())
label_index_map = {moa: i for i, moa in enumerate(moas)}

for out_base in [user_out_base, repo_pkl_out]:
    pkl_path = os.path.join(out_base, filename_pkl)
    with open(pkl_path, "wb") as f:
        pickle.dump(label_index_map, f)
    print(f"✅ Saved label index map: {pkl_path} ({len(label_index_map)} MOAs)")

