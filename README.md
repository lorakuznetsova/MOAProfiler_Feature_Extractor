cat > "$HOME/Paper_Validation/MOAProfiler_Feature_Extractor/README.md" <<'EOF'
# MOAProfiler Feature Extractor (reproducible run)

This repo provides a minimal, reproducible way to extract **MOAProfiler** embeddings from Cell Painting images using a small manifest and a code-only submodule. It includes:

- `pipeline/embed_runner.py` — embedding runner (produces per-image embeddings)
- `pipeline/1_convert_moa_profiler_cellpaintssl.py` — per-image → per-well (CellPaintSSL-style)
- `pipeline/2_prepare_cellpaintssl_dataset.py` — builds `dataset.csv` next to the manifest
- `pipeline/3_postprocess_cellpaintssl.py` — post-processing & aggregation (PyCytominer)
- `assets/moaprofiler/model_best.dat` — JUMP1 EfficientNet-B0 checkpoint
- `assets/moaprofiler/channel_stats_compounds_raw.pkl` — global channel stats for normalization
- `submodules/moa-profiler` — code-only fork of Pfizer’s MOAProfiler (no heavy CSVs/LFS)
- `LICENSES/` — third-party licenses (MOAProfiler, CellPaintSSL); keep this folder in the repo

> Images + `manifest.csv` are provided separately (e.g., Zenodo). The manifest must reference images with **relative** paths like `images/<well>_<site>/ch1.tif`.

---

## 1) Environment

Create the conda env (Python 3.8):

```bash
mamba env create -f env2.yml || conda env create -f env2.yml
conda activate env2
2) Project scaffold
We assume this layout:

csharp
Copy code
~/Paper_Validation/
  ├─ MOAProfiler_Feature_Extractor/   <- this repo (with submodule + assets)
  └─ MOAProfiler_Data/
      ├─ MOAProfiler_IN/              <- images/ + manifest.csv
      └─ MOAProfiler_OUT/             <- results go here
Create it:

bash
Copy code
PROJECT=~/Paper_Validation
mkdir -p "$PROJECT/MOAProfiler_Data"/{MOAProfiler_IN,MOAProfiler_OUT}
3) Clone this repo
Public (HTTPS) — recommended for reviewers:

bash
Copy code
PROJECT=~/Paper_Validation
git clone https://github.com/lorakuznetsova/MOAProfiler_Feature_Extractor.git \
  "$PROJECT/MOAProfiler_Feature_Extractor"

cd "$PROJECT/MOAProfiler_Feature_Extractor"
git submodule update --init --recursive   # pulls the code-only submodule
Author (private via SSH):

bash
Copy code
PROJECT=~/Paper_Validation
git clone git@github.com:lorakuznetsova/MOAProfiler_Feature_Extractor.git \
  "$PROJECT/MOAProfiler_Feature_Extractor"

cd "$PROJECT/MOAProfiler_Feature_Extractor"
git -c submodule.submodules/moa-profiler.url=git@github.com:lorakuznetsova/moa-profiler.git \
    submodule update --init --recursive
Quick sanity:

bash
Copy code
test -f submodules/moa-profiler/efficientnet_pytorch/__init__.py && echo "OK effnet"
test -f submodules/moa-profiler/classification.py && echo "OK classification"
test -f assets/moaprofiler/model_best.dat && echo "OK checkpoint"
test -f assets/moaprofiler/channel_stats_compounds_raw.pkl && echo "OK channel stats"
4) Get the dataset (images + manifest)
Download and extract the archive (e.g., Zenodo) so you end up with:

javascript
Copy code
~/Paper_Validation/MOAProfiler_Data/MOAProfiler_IN/
  ├─ images/
  └─ manifest.csv
Manifest rules

Column name: imagename

Values are relative paths from MOAProfiler_IN, e.g.:

bash
Copy code
images/M11_s5/ch1.tif
images/K10_s2/ch1.tif
Each row corresponds to ch1 for a single site; the runner constructs ch2–ch5 by filename replacement.

Accepted channel order (MOAProfiler default): DNA, ER, RNA, AGP, Mito.

All channels for a site must have the same XY size (we used 1080×1080 after center-crop/pad from Operetta 1360×1024).

5) Run the embedder (per-image embeddings)
bash
Copy code
PROJECT=~/Paper_Validation
REPO="$PROJECT/MOAProfiler_Feature_Extractor"
IN="$PROJECT/MOAProfiler_Data/MOAProfiler_IN"
OUT="$PROJECT/MOAProfiler_Data/MOAProfiler_OUT"

conda activate env2
export PYTHONPATH="$REPO:$REPO/submodules/moa-profiler"

# optional quick checks
test -f "$IN/manifest.csv" && echo "manifest OK" && head -n 3 "$IN/manifest.csv"

# run from IN so relative paths like images/... resolve
cd "$IN"
python "$REPO/pipeline/embed_runner.py" \
  --csv    manifest.csv \
  --ckpt   "$REPO/assets/moaprofiler/model_best.dat" \
  --stats  "$REPO/assets/moaprofiler/channel_stats_compounds_raw.pkl" \
  --outdir "$OUT"
What you should see: a line like >> loading checkpoint: ...model_best.dat, then iteration over images/sites.
Result: $OUT/per_image_embeddings_with_filenames.csv.

6) Convert + dataset (scripts 1 & 2)
Script 1: per-image → per-well (CellPaintSSL-style)

bash
Copy code
python "$REPO/pipeline/1_convert_moa_profiler_cellpaintssl.py" \
  --emb  "$OUT/per_image_embeddings_with_filenames.csv" \
  --meta "$IN/manifest.csv" \
  --well-out "$OUT/well_features.csv" \
  --agg median
Script 2: create dataset.csv

bash
Copy code
python "$REPO/pipeline/2_prepare_cellpaintssl_dataset.py" \
  --meta "$IN/manifest.csv" \
  --dataset-out "$OUT/dataset.csv"
Outputs:

$OUT/well_features.csv

$OUT/dataset.csv

7) Post-process & aggregate (script 3, PyCytominer)
Available normalization methods (single or chained combos):

nginx
Copy code
standardize
mad_robustize
spherize
spherize_mad_robustize
mad_robustize_spherize
spherize_standardize
standardize_spherize
no_post_proc
Default is spherize_mad_robustize. Run:

bash
Copy code
python "$REPO/pipeline/3_postprocess_cellpaintssl.py" \
  --embedding_csv "$OUT/well_features.csv" \
  --val_csv       "$OUT/dataset.csv" \
  --operation     median \
  --norm_method   spherize_mad_robustize \
  --out_well_csv  "$OUT/well_features_normalized.csv" \
  --out_agg_csv   "$OUT/agg_features_from_normalized.csv"
Outputs:

$OUT/well_features_normalized.csv — post-processed per-well features

$OUT/agg_features_from_normalized.csv — aggregated (by perturbation_id) features

Note: We do not use any helper shell script. Run the three Python commands directly.

8) One-shot pipeline (all steps after embeddings)
If you already have $OUT/per_image_embeddings_with_filenames.csv, do:

bash
Copy code
# 1) per-image -> per-well
python "$REPO/pipeline/1_convert_moa_profiler_cellpaintssl.py" \
  --emb  "$OUT/per_image_embeddings_with_filenames.csv" \
  --meta "$IN/manifest.csv" \
  --well-out "$OUT/well_features.csv" \
  --agg median

# 2) dataset.csv
python "$REPO/pipeline/2_prepare_cellpaintssl_dataset.py" \
  --meta "$IN/manifest.csv" \
  --dataset-out "$OUT/dataset.csv"

# 3) post-proc + aggregate
python "$REPO/pipeline/3_postprocess_cellpaintssl.py" \
  --embedding_csv "$OUT/well_features.csv" \
  --val_csv       "$OUT/dataset.csv" \
  --operation     median \
  --norm_method   spherize_mad_robustize \
  --out_well_csv  "$OUT/well_features_normalized.csv" \
  --out_agg_csv   "$OUT/agg_features_from_normalized.csv"
9) Versions & exact commits (reproducibility)
Environment (conda env: env2):

torch 1.10.2

torchvision 0.11.3 (CPU build; warning about libc10_cuda.so is harmless on CPU-only machines)

numpy 1.22.2

pandas 1.1.3

Git:

Repo commit (HEAD): d10ad21a0686529779f7cbc46a4c76288095aa41

Submodule submodules/moa-profiler: 20e3ec88353c5daa8619d39eb5c6f5aebe20dd0e (branch code-only)

The submodule is pinned. git submodule update --init --recursive checks out the exact commit above.

10) Troubleshooting
ModuleNotFoundError: efficientnet_pytorch
Export PYTHONPATH in the same shell where you run Python:

bash
Copy code
export PYTHONPATH="$REPO:$REPO/submodules/moa-profiler"
FileNotFoundError: manifest.csv or missing images
Run from the input folder and keep relative paths in the manifest:

bash
Copy code
cd "$IN"
TorchVision CUDA warning (libc10_cuda.so not found)
Expected on CPU-only machines; the run proceeds on CPU.

11) Freeze the environment (optional but recommended for reviewers)
bash
Copy code
# Activate env
conda activate env2

# Exact conda spec (bit-for-bit on conda/mamba)
conda list --explicit > "$REPO/env2-conda-spec.txt"

# Human-readable export (no build strings)
conda env export --no-builds > "$REPO/env2.lock.yml"

# If any pip installs inside env2:
pip freeze > "$REPO/requirements-env2.txt"
Commit these files if you want:

bash
Copy code
git -C "$REPO" add env2-conda-spec.txt env2.lock.yml requirements-env2.txt 2>/dev/null || true
git -C "$REPO" commit -m "Add env2 locks for reproducibility" || true
git -C "$REPO" push || true
12) Licenses & citation
Third-party licenses are stored in LICENSES/. Keep this folder in the repo (reviewers need it).

Original MOAProfiler: Pfizer Open Source (see submodules/moa-profiler/LICENSE and their README).

Please cite MOAProfiler as in their repo and in our manuscript.

This repo provides a minimal wrapper to reproduce embeddings (and downstream post-processing) on our dataset.

EOF
