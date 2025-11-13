#!/usr/bin/env python3
"""
convert_moa_profiler_cellpaintssl.py  (dependency-free)

Convert MOA Profiler per-image embeddings into CellPaintSSL-style WELL features.

Inputs
  --emb  : per_image_embeddings_with_filenames.csv (imagename, dim0..dimN)
  --meta : metadata CSV (imagename, perturbation, moas, [batch], [plate])
Outputs
  --well-out : well_features.csv  (batch, plate, well, perturbation_id, target, emb0..embN)

Aggregation is well-level median (default) or mean (--agg mean).
"""

import argparse
import os
from typing import List
import numpy as np
import pandas as pd

DEFAULT_BATCH = "BATCH_1"
DEFAULT_PLATE = "PLATE_1"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="convert_moa_profiler_cellpaintssl.py",
        description="Convert MOA Profiler per-image outputs to CellPaintSSL-style WELL features (no external deps)."
    )
    p.add_argument("--emb", required=True, help="Path to per_image_embeddings_with_filenames.csv")
    p.add_argument("--meta", required=True, help="Path to metadata CSV (imagename, perturbation, moas, [batch], [plate])")
    p.add_argument("--well-out", required=True, help="Output well_features.csv")
    p.add_argument("--agg", default="median", choices=["median", "mean"], help="Well aggregation (default: median)")
    p.add_argument("--chunk-size", type=int, default=1024, help="Unused (kept for CLI compatibility)")
    return p.parse_args()

def extract_well_from_path(path: str) -> str:
    parts = os.path.normpath(str(path)).split(os.sep)
    folder = parts[-2] if len(parts) >= 2 else ""
    return folder.split("_")[0] if folder else ""

def assert_required_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {where}: {missing}. Expected at least {cols}.")

def get_dim_columns(df: pd.DataFrame) -> List[str]:
    dim_cols = [c for c in df.columns if c.startswith("dim")]
    if not dim_cols:
        raise ValueError("No embedding columns found; expected dim0..dimN in embeddings CSV.")
    try:
        return sorted(dim_cols, key=lambda x: int(x.replace("dim", "")))
    except Exception as e:
        raise ValueError("Embedding columns must be named dim0..dimN with integer suffixes.") from e

def build_well_features(emb_csv: str, meta_csv: str, agg: str = "median") -> pd.DataFrame:
    emb_df = pd.read_csv(emb_csv)
    meta_df = pd.read_csv(meta_csv)

    assert_required_columns(emb_df, ["imagename"], "embeddings")
    assert_required_columns(meta_df, ["imagename", "perturbation", "moas"], "metadata")

    dim_cols = get_dim_columns(emb_df)

    meta_df = meta_df.copy()
    meta_df["well"] = meta_df["imagename"].apply(extract_well_from_path)
    if "batch" not in meta_df.columns:
        meta_df["batch"] = DEFAULT_BATCH
    if "plate" not in meta_df.columns:
        meta_df["plate"] = DEFAULT_PLATE
    meta_df["perturbation_id"] = meta_df["perturbation"].astype(str)
    meta_df["target"] = meta_df["moas"]

    joined = pd.merge(
        emb_df[["imagename"] + dim_cols],
        meta_df[["imagename", "batch", "plate", "well", "perturbation_id", "target"]],
        on="imagename",
        how="inner",
        validate="one_to_one",
    )

    # Strict: require 1:1 coverage of metadata by embeddings
    if len(joined) != len(meta_df):
        missing = len(meta_df) - len(joined)
        raise RuntimeError(
            f"[ERROR] {missing} metadata rows had no matching embeddings by 'imagename'. "
            "Ensure both CSVs contain identical 'imagename' values."
        )

    keys = ["batch", "plate", "well", "perturbation_id", "target"]
    g = joined.groupby(keys, sort=False)[dim_cols]
    agg_df = (g.median() if agg == "median" else g.mean()).reset_index()

    # Rename dim* -> emb* and order columns
    rename_map = {c: "emb" + c[3:] for c in dim_cols}
    agg_df = agg_df.rename(columns=rename_map)
    emb_cols_out = sorted([c for c in agg_df.columns if c.startswith("emb")],
                          key=lambda x: int(x.replace("emb", "")))
    agg_df = agg_df[keys + emb_cols_out]

    # (Optional) smaller file + numerical stability
    for c in emb_cols_out:
        agg_df[c] = agg_df[c].astype(np.float32)

    return agg_df

def main():
    args = parse_args()
    for p in (args.emb, args.meta):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    out_df = build_well_features(args.emb, args.meta, args.agg)
    os.makedirs(os.path.dirname(args.well_out), exist_ok=True)
    out_df.to_csv(args.well_out, index=False)
    print(f"[OK] Well-level features saved: {args.well_out}")
    print(f"[INFO] Shape: {out_df.shape[0]} wells x {out_df.shape[1]} cols")
    print(f"[INFO] First columns: {list(out_df.columns[:5])} | Last emb col: {out_df.columns[-1]}")

if __name__ == "__main__":
    main()

