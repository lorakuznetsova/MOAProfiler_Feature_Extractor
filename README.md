# MOAProfiler Feature Extractor — Reproducible Inference & Post-processing

## Contents
- [Image preprocessing](#image-preprocessing-what-the-embedder-expects)
- [`manifest.csv` rules](#manifestcsv-rules)
- [Environment for embedding (`env2`)](#environment-for-embedding-env2)
- [Clone the repo + submodule](#clone-the-repo--submodule)
- [Run the embedder](#run-the-embedder)
- [Post-processing / normalization](#post-processing--normalization-cellpaintssl-style)
- [Data: MOAProfiler_IN](https://github.com/lorakuznetsova/MOAProfiler_Feature_Extractor/blob/main/README.md#data-moaprofiler_in-zenodo-29-gb)
- Licensing and attribution

This repo provides:
- a **reproducible embedding runtime** for the MOAProfiler model (Pfizer) to extract well-level features from Cell Painting images; and
- a **post-processing pipeline** (CellPaintSSL-style options) to normalize profiles.

---

## Image preprocessing (what the embedder expects)

**Per-site inputs (inference):**
- 5 single-plane **16-bit TIFFs** named `ch1.tif` … `ch5.tif`
- All five are the same size: **1080×1080 px**
- Channel semantic order: `ch1=DNA`, `ch2=ER`, `ch3=RNA`, `ch4=AGP`, `ch5=Mito`
- No color tables, no multi-page TIFFs, no JPEG compression

**Folder layout (example):**
```
MOAProfiler_Data/
└─ MOAProfiler_IN/
   ├─ images/
   │  ├─ A01_s1/
   │  │  ├─ ch1.tif
   │  │  ├─ ch2.tif
   │  │  ├─ ch3.tif
   │  │  ├─ ch4.tif
   │  │  └─ ch5.tif
   │  └─ A01_s2/
   │     ├─ ch1.tif
   │     └─ ...
   └─ manifest.csv
```

## `manifest.csv` (rules)

One row **per site**, pointing to **`ch1.tif`**. Other channels are inferred automatically.

**Required columns (minimum):**
- `imagename` — path to **ch1** (prefer **relative** paths like `images/A01_s1/ch1.tif`)
- `moas` — any string label (used downstream)
- `perturbation` — identifier shown in outputs (e.g., compound name/ID)

If you currently have absolute paths, place `manifest.csv` next to `images/` and rewrite `imagename` to start with `images/.../ch1.tif`.

---

## Environment for embedding (`env2`)

We track a **frozen pip lock file** at `envs/env2/embed_requirements.txt`. Use it to recreate the exact working stack (Torch 1.9.0 **CPU** + deps) without fighting a large YAML.

```bash
# fresh minimal conda env (one-time)
conda create -y -n env2 python=3.8 pip
conda activate env2

# install the frozen stack (includes torch==1.9.0+cpu etc.)
REPO=~/Paper_Validation/MOAProfiler_Feature_Extractor
python -m pip install --no-cache-dir \
  -f https://download.pytorch.org/whl/torch_stable.html \
  -r "$REPO/envs/env2/embed_requirements.txt"

# sanity
python - <<'PY'
import torch, torchvision, timm, numpy, pandas, sklearn
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("torchvision", torchvision.__version__)
print("timm", getattr(__import__('timm'), '__version__', 'n/a'))
print("numpy/pandas/sklearn", numpy.__version__, pandas.__version__, sklearn.__version__)
PY
```
You can keep env2.yml for documentation, but for reproducibility the freeze file is simpler and stable across machines.

## Clone the repo + submodule

```bash
PROJECT=~/Paper_Validation
mkdir -p "$PROJECT/MOAProfiler_Data"/{MOAProfiler_IN,MOAProfiler_OUT}

git clone https://github.com/lorakuznetsova/MOAProfiler_Feature_Extractor.git \
  "$PROJECT/MOAProfiler_Feature_Extractor"

cd "$PROJECT/MOAProfiler_Feature_Extractor"
git submodule update --init --recursive
```
Quick sanity:

```bash
test -f submodules/moa-profiler/efficientnet_pytorch/__init__.py && echo "OK: effnet"
test -f submodules/moa-profiler/classification.py && echo "OK: classification"
test -f assets/moaprofiler/model_best.dat && echo "OK: checkpoint"
test -f assets/moaprofiler/channel_stats_compounds_raw.pkl && echo "OK: channel stats"
```
## Run the embedder

Define these paths once per shell and reuse them.

```bash
# paths
PROJECT=~/Paper_Validation
REPO="$PROJECT/MOAProfiler_Feature_Extractor"
IN="$PROJECT/MOAProfiler_Data/MOAProfiler_IN"
OUT="$PROJECT/MOAProfiler_Data/MOAProfiler_OUT"

# quick checks
test -f "$REPO/submodules/moa-profiler/efficientnet_pytorch/__init__.py" && echo "effnet OK"
test -f "$REPO/submodules/moa-profiler/classification.py" && echo "clf OK"
test -f "$REPO/assets/moaprofiler/model_best.dat" && echo "ckpt OK"
test -f "$REPO/assets/moaprofiler/channel_stats_compounds_raw.pkl" && echo "stats OK"
test -f "$IN/manifest.csv" && echo "manifest OK"
head -n 3 "$IN/manifest.csv"
```
Run:

```bash
conda activate env2
export PYTHONPATH="$REPO:$REPO/submodules/moa-profiler"

cd "$IN"   # manifest uses relative paths like images/.../ch1.tif

python "$REPO/pipeline/embed_runner.py" \
  --csv    manifest.csv \
  --ckpt   "$REPO/assets/moaprofiler/model_best.dat" \
  --stats  "$REPO/assets/moaprofiler/channel_stats_compounds_raw.pkl" \
  --outdir "$OUT"
```
What to expect

- You’ll see >> loading checkpoint: ... and sites iterating.

- Outputs go under $OUT (e.g., per_image_embeddings_with_filenames.csv).

If something barks

- ModuleNotFoundError: efficientnet_pytorch → ensure
export PYTHONPATH="$REPO:$REPO/submodules/moa-profiler" is set in the same shell that runs Python.

- FileNotFoundError: manifest.csv → run from "$IN" and ensure imagename is relative (starts with images/).

- CUDA not required: this stack is CPU-only and consistent across machines.
## Post-processing / normalization (CellPaintSSL-style)

We provide 3 scripts to convert per-image → per-well, build dataset.csv, and apply post-processing.

### Environment: cellpaintssl_env (one-time)

```bash
# create + activate venv
python3 -m venv ~/cellpaintssl_env
source ~/cellpaintssl_env/bin/activate

# install pinned requirements
cd ~/Paper_Validation/MOAProfiler_Feature_Extractor
pip install --upgrade pip
# adjust case if your folder name differs
pip install -r envs/cellpaintssl_env/requirements.txt
```
### Run the normalization scripts

```bash
# paths
PROJECT=~/Paper_Validation
REPO="$PROJECT/MOAProfiler_Feature_Extractor"
IN="$PROJECT/MOAProfiler_Data/MOAProfiler_IN"
OUT="$PROJECT/MOAProfiler_Data/MOAProfiler_OUT"

source ~/cellpaintssl_env/bin/activate

# 1) per-image -> per-well (median aggregate)
python "$REPO/pipeline/1_convert_moa_profiler_cellpaintssl.py" \
  --emb  "$OUT/per_image_embeddings_with_filenames.csv" \
  --meta "$IN/manifest.csv" \
  --well-out "$OUT/well_features.csv" \
  --agg median

# 2) dataset.csv (labels/metadata)
python "$REPO/pipeline/2_prepare_cellpaintssl_dataset.py" \
  --meta "$IN/manifest.csv" \
  --dataset-out "$OUT/dataset.csv"

# 3) postprocess (change --norm_method if you prefer)
python "$REPO/pipeline/3_postprocess_cellpaintssl.py" \
  --embedding_csv "$OUT/well_features.csv" \
  --val_csv "$OUT/dataset.csv" \
  --operation median \
  --norm_method spherize_mad_robustize \
  --out_well_csv "$OUT/well_features_normalized.csv" \
  --out_agg_csv  "$OUT/agg_features_from_normalized.csv"
```
Available --norm_method:
```
- standardize
- mad_robustize
- spherize
- spherize_mad_robustize        (default)
- mad_robustize_spherize
- spherize_standardize
- standardize_spherize
- no_post_proc
```
## Data: MOAProfiler_IN (Zenodo, ~29 GB)

The preprocessed input data used in this work are publicly available on Zenodo:

**Zenodo dataset:** https://doi.org/10.5281/zenodo.17617785

Download the archive `MOAProfiler_IN_v0.1.tar.gz` (~29 GB) from the Zenodo record and extract it, e.g.:

```bash
mkdir -p ~/Paper_Validation/MOAProfiler_Data
tar -xvzf MOAProfiler_IN_v0.1.tar.gz -C ~/Paper_Validation/MOAProfiler_Data
```
After extraction, the directory structure should look like:

~/Paper_Validation/MOAProfiler_Data/MOAProfiler_IN/...

Use this MOAProfiler_IN folder as the input data directory when running the
embedding pipeline (e.g. pipeline/embed_runner.py) as described above.
## Licensing and attribution

The code in this repository is distributed under the terms of the license(s)
included in the `LICENSE` file(s) in the repository root.

This project reuses and adapts components from the following open-source projects:

- **MOAProfiler** (Pfizer)  
  Repository: https://github.com/pfizer-opensource/moa-profiler  
  License: Apache License 2.0  
  Parts of the embedding / inference pipeline (e.g. `pipeline/embed_runner.py`
  and related utilities) are derived from or inspired by the MOAProfiler codebase.

- **CellPaintSSL** (Bayer)  
  Repository: https://github.com/Bayer-Group/CellPaintSSL  
  License: BSD 3-Clause License  
  Parts of the post-processing pipeline (e.g. `pipeline/3_postprocess_cellpaintssl.py`)
  are derived from or inspired by the CellPaintSSL codebase.

For full details, please refer to the `LICENSE` and `NOTICE` files shipped with this repository.


