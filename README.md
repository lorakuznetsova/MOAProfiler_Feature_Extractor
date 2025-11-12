cat > README.md <<'EOF'
# MOAProfiler Feature Extractor (reproducible run)

This repo provides a minimal, reproducible way to extract **MOAProfiler** embeddings from Cell Painting images using a small manifest and a code-only submodule. It includes:
- `pipeline/embed_runner.py` — embedding runner
- `assets/moaprofiler/model_best.dat` — JUMP1 EfficientNet-B0 checkpoint
- `assets/moaprofiler/channel_stats_compounds_raw.pkl` — global channel stats for normalization
- `submodules/moa-profiler` — code-only fork of Pfizer’s MOAProfiler (no heavy CSVs/LFS)

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
If the repo is public (HTTPS) (recommended for reviewers):
bash
Copy code
PROJECT=~/Paper_Validation
git clone https://github.com/lorakuznetsova/MOAProfiler_Feature_Extractor.git \
  "$PROJECT/MOAProfiler_Feature_Extractor"

cd "$PROJECT/MOAProfiler_Feature_Extractor"
git submodule update --init --recursive   # pulls the code-only submodule
If you are the author (private via SSH):
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
...
Each row corresponds to ch1 for a single site; the runner constructs ch2–ch5 by filename replacement.

Accepted channel order: DNA, ER, RNA, AGP, Mito (MOAProfiler’s default).

All channels for a site must have the same XY size (we used 1080×1080 after crop/pad from Operetta 1360×1024).

5) Run the embedder
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
What you should see:

ruby
Copy code
>> loading checkpoint: ...model_best.dat
…and then iteration over images/sites. Results are saved under $OUT.

6) Troubleshooting
ModuleNotFoundError: efficientnet_pytorch
Ensure you exported PYTHONPATH in the same shell where you run Python:

bash
Copy code
export PYTHONPATH="$REPO:$REPO/submodules/moa-profiler"
FileNotFoundError: manifest.csv or missing images
Make sure you cd "$IN" before running, and the manifest uses relative paths like images/....

TorchVision CUDA warning (e.g., libc10_cuda.so not found)
Harmless on CPU-only machines; the run proceeds on CPU.

7) Preprocessing summary (what we used)
Operetta source frames (1360×1024) were center-cropped and/or padded to 1080×1080 (no resizing) to ensure same XY across channels.

Per-channel min–max scaling to [0,1] is performed inside the dataset; global channel stats from assets/.../channel_stats_compounds_raw.pkl are used by the runner to normalize intensities consistently with MOAProfiler JUMP1 settings.

Channels: DNA, ER, RNA, AGP, Mito.

8) Citation & licenses
Original MOAProfiler: Pfizer Open Source (see submodules/moa-profiler/LICENSE and README).

Please cite MOAProfiler as in their repo and our manuscript.

This repo provides a minimal wrapper to reproduce embeddings on our dataset.
EOF
