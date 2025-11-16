# pipeline/embed_runner.py
# SPDX-License-Identifier: Apache-2.0
#
# This file is adapted from the MOAProfiler project:
#   https://github.com/pfizer-opensource/moa-profiler
# which is licensed under the Apache License, Version 2.0.
#
# Modifications Copyright (c) 2025, Xenia Kuznetsova and Larisa Kuznetsova.
# See the accompanying LICENSE files in this repository root
# for the full license text.
import os
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from efficientnet_pytorch import EfficientNet

# Reuse the dataset class from your repo
from classification import JUMPMOADataset


def build_backbone_no_head(ckpt_path, in_channels=5):
    """EfficientNet-B0 backbone without the classification head."""
    model = EfficientNet.from_name('efficientnet-b0', in_channels=in_channels, num_classes=1)
    model._fc = nn.Identity()  # penultimate 1280-d vector

    print(">> loading checkpoint:", ckpt_path)
    state = torch.load(ckpt_path, map_location='cpu')

    # Remove DataParallel prefixes and classifier head weights
    state = {k.replace('module.', ''): v for k, v in state.items()}
    state = {k: v for k, v in state.items() if not k.startswith('_fc.')}

    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        print("⚠️ Unexpected keys:", unexpected)
    if missing:
        non_head_missing = [k for k in missing if not k.startswith('_fc.')]
        if non_head_missing:
            print("ℹ️ Missing non-head keys:", non_head_missing)

    model.eval()
    return model


def parse_args():
    p = argparse.ArgumentParser(description="MOAProfiler feature extractor (embeddings only)")
    p.add_argument("--csv",   required=True, help="Path to manifest CSV (e.g., your well_excluded.csv)")
    p.add_argument("--ckpt",  required=True, help="Path to weights file (model_best.dat)")
    p.add_argument("--stats", required=True, help="Path to channel stats pickle (channel_stats_compounds_raw.pkl)")
    p.add_argument("--outdir",required=True, help="Output directory for embeddings and metadata")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers",    type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ---- Load global channel stats ----
    with open(args.stats, 'rb') as f:
        norm = pickle.load(f)
    means = [norm["mean"][i] for i in range(5)]
    stds  = [norm["std"][i]  for i in range(5)]
    data_transforms = transforms.Compose([transforms.Normalize(means, stds)])

    # ---- Minimal label map (classes don't matter for embeddings) ----
    df = pd.read_csv(args.csv)
    moas = df['moas'].astype(str).str.strip().unique().tolist()
    label_index_map = {m: i for i, m in enumerate(moas)}

    # ---- Dataset / loader ----
    ds = JUMPMOADataset(
        args.csv,
        transform=data_transforms,
        jitter=False,
        label_index_map=label_index_map,
        augment=False,
        reverse=False,
    )
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.workers,
    )

    # ---- Model ----
    model = build_backbone_no_head(args.ckpt, in_channels=5)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ---- Inference loop ----
    all_embs, all_names, meta_rows = [], [], []
    with torch.no_grad():
        for (imagenames, x, target, perturbations) in dl:
            x = x.to(device)
            emb = model(x).cpu().numpy()  # (B, 1280)
            all_embs.append(emb)
            all_names.extend(list(imagenames))
            for i, im in enumerate(imagenames):
                meta_rows.append({
                    "imagename": im,
                    "perturbation": perturbations[i],
                    "target_index": int(target[i]),
                })

    if not all_embs:
        print("No images processed. Check your CSV paths.")
        return

    embs = np.vstack(all_embs)  # (N, 1280)

    # ---- Outputs (single folder) ----
    np.save(os.path.join(args.outdir, "per_image_embeddings.npy"), embs)

    cols = [f"dim{i}" for i in range(embs.shape[1])]
    df_emb = pd.DataFrame(embs, columns=cols)
    df_emb.insert(0, "imagename", all_names)
    df_emb.to_csv(os.path.join(args.outdir, "per_image_embeddings_with_filenames.csv"), index=False)

    pd.DataFrame(meta_rows).to_csv(os.path.join(args.outdir, "per_image_metadata.csv"), index=False)

    print("✅ Done.")
    print("Embeddings:", embs.shape)
    print("Saved to:", args.outdir)


if __name__ == "__main__":
    main()

