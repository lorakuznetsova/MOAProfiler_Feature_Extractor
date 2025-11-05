# embed_runner.py
import os, pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Reuse the dataset class from the repo:
from classification import JUMPMOADataset

# --- Output path ---
# TODO: UPDATE BEFORE EVERY RUN!!!
extra_out_dir = "/home/usr/Desktop/Platform_Validation/MOA_JUMP/Results_MOAProfiler/Output"

# -------- CONFIG (edit if you changed these in the repo) --------
# LABEL_TYPE = "moa_targets_compounds_polycompound"  # only used to pick the "full" CSV path
FULL_CSV   = "csvs/JUMP1/compound_images_with_MOA_no_polypharm_well_excluded.csv"

CKPT_PATH  = "save_dir/JUMP1/multiclass_classification/Jun-22-2022-08:49:11/models/model_best.dat"
STATS_PKL  = "pickles/JUMP1/channel_stats_compounds_raw.pkl"
OUT_EMB_NPY = "embeddings/per_image_embeddings.npy"
OUT_META_CSV = "embeddings/per_image_metadata.csv"

BATCH_SIZE = 16
NUM_WORKERS = 0  # safe on most setups

# -------- Helpers --------
def build_backbone_no_head(ckpt_path, in_channels=5):
    # Build EfficientNet-B0 with dummy head, then strip head and load weights (except head)
    model = EfficientNet.from_name('efficientnet-b0', in_channels=in_channels, num_classes=1)
    model._fc = nn.Identity()  # forward() now outputs 1280-d penultimate vector

    print(">> DEBUG loading:", ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')

    # clean DataParallel/Distributed prefixes
    from collections import OrderedDict
    state = OrderedDict((k.replace('module.', ''), v) for k, v in state.items())

    # drop classifier weights
    state = {k: v for k, v in state.items() if not k.startswith('_fc.')}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print("⚠️ Unexpected keys in checkpoint:", unexpected)
    if missing:
        only_head = [k for k in missing if k.startswith('_fc.')]
        if only_head:
            print("ℹ️ Missing keys (expected head):", only_head)
        else:
            print("ℹ️ Missing non-head keys:", missing)

    model.eval()
    return model

def main():
    os.makedirs(os.path.dirname(OUT_EMB_NPY), exist_ok=True)

    # Load global channel stats (same ones the repo uses)
    with open(STATS_PKL, 'rb') as f:
        norm = pickle.load(f)
    means = [norm["mean"][i] for i in range(5)]
    stds  = [norm["std"][i]  for i in range(5)]
    data_transforms = transforms.Compose([transforms.Normalize(means, stds)])

    # Minimal label map so JUMPMOADataset can index something.
    # We'll build it from the CSV on the fly; classes don't matter for embeddings.
    df = pd.read_csv(FULL_CSV)
    moas = df['moas'].astype(str).str.strip().unique().tolist()
    label_index_map = {m:i for i,m in enumerate(moas)}

    # Dataset / loader
    ds = JUMPMOADataset(FULL_CSV, transform=data_transforms, jitter=False,
                        label_index_map=label_index_map, augment=False, reverse=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                                     pin_memory=torch.cuda.is_available(),
                                     num_workers=NUM_WORKERS)

    # Model w/ head removed
    model = build_backbone_no_head(CKPT_PATH, in_channels=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Inference: collect per-image embeddings + minimal metadata
    all_embs = []
    all_names = []
    meta_rows = []
    with torch.no_grad():
        for (imagenames, x, target, perturbations) in dl:
            x = x.to(device)
            emb = model(x).cpu().numpy()  # (B, 1280)
            all_embs.append(emb)
            all_names.extend(list(imagenames)) # added for definitively matching embeddings with metadata

            # record metadata row-wise
            for i, im in enumerate(imagenames):
                meta_rows.append({
                    "imagename": im,
                    "perturbation": perturbations[i],
                    "target_index": int(target[i])   # index into label_index_map; not used further here
                })

    if len(all_embs) == 0:
        print("No images processed. Check your CSV paths.")
        return

    embs = np.vstack(all_embs)  # (N, 1280)
    np.save(OUT_EMB_NPY, embs)
    pd.DataFrame(meta_rows).to_csv(OUT_META_CSV, index=False)
    
    # --- also save to MOAProfiler_OUT ---
    os.makedirs(extra_out_dir, exist_ok=True)

    np.save(os.path.join(extra_out_dir, "per_image_embeddings.npy"), embs)

    # Save embeddings with filenames for matching to metadata
    assert len(all_names) == embs.shape[0], "ERROR: filenames vs embeddings length mismatch"
    cols = [f"dim{i}" for i in range(embs.shape[1])]
    df_emb = pd.DataFrame(embs, columns=cols)
    df_emb.insert(0, "imagename", all_names)
    df_emb.to_csv(os.path.join(extra_out_dir, "per_image_embeddings_with_filenames.csv"), index=False)

    pd.DataFrame(meta_rows).to_csv(os.path.join(extra_out_dir, "per_image_metadata.csv"), index=False)

    print("✅ Done.")
    print("Embeddings:", embs.shape, "->", OUT_EMB_NPY)
    print("Metadata  ->", OUT_META_CSV)

if __name__ == "__main__":
    main()

