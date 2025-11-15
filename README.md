# MOAProfiler Feature Extractor — Reproducible Inference & Post-processing

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
MOAProfiler_Data/
MOAProfiler_IN/
images/
A01_s1/
ch1.tif ch2.tif ch3.tif ch4.tif ch5.tif
A01_s2/
ch1.tif ...
manifest.csv

---

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

Clone the repo + submodule

HTTPS (public):

```bash
PROJECT=~/Paper_Validation
mkdir -p "$PROJECT/MOAProfiler_Data"/{MOAProfiler_IN,MOAProfiler_OUT}

git clone https://github.com/lorakuznetsova/MOAProfiler_Feature_Extractor.git \
  "$PROJECT/MOAProfiler_Feature_Extractor"

cd "$PROJECT/MOAProfiler_Feature_Extractor"
git submodule update --init --recursive
```
SSH (private fork) — optional:

```bash
PROJECT=~/Paper_Validation
mkdir -p "$PROJECT/MOAProfiler_Data"/{MOAProfiler_IN,MOAProfiler_OUT}

git clone git@github.com:lorakuznetsova/MOAProfiler_Feature_Extractor.git \
  "$PROJECT/MOAProfiler_Feature_Extractor"

cd "$PROJECT/MOAProfiler_Feature_Extractor"
git -c submodule.submodules/moa-profiler.url=git@github.com:lorakuznetsova/moa-profiler.git \
    submodule update --init --recursive
```
Quick sanity:

```bash
test -f submodules/moa-profiler/efficientnet_pytorch/__init__.py && echo "OK: effnet"
test -f submodules/moa-profiler/classification.py && echo "OK: classification"
test -f assets/moaprofiler/model_best.dat && echo "OK: checkpoint"
test -f assets/moaprofiler/channel_stats_compounds_raw.pkl && echo "OK: channel stats"
```


