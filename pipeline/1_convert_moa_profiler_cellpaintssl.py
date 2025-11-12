#!/usr/bin/env python3
"""
convert_moa_profiler_cellpaintssl.py

Convert MOA Profiler outputs to CellPaintSSL-style well-level features

Inputs:
  - per_image_embeddings_with_filenames.csv                     (embeddings)
      * imagename      (string; like /path/to/A01_s5/ch1.tif, must match metadata)
      * dim0..dimN     (float; embedding columns from MOA Profiler, kept as-is)
  - compound_images_with_MOA_no_polypharm_well_excluded.csv     (metadata)
      * imagename      (string; must match embeddings)
      * perturbation   (string; compound name)   -> mapped to 'perturbation_id' in output
      * moas           (string; MOA name)        -> mapped to 'target' in output
      * [optional] batch (string; default 'BATCH_1' if absent)
      * [optional] plate (string; default 'PLATE_1' if absent)

Output (CellPaintSSL-style):
  - well_features.csv with columns:
      batch, plate, well, perturbation_id, target, emb0..embN
      aggregated to well-level by median (default) or mean (--agg mean)

Refs:
  - MOA Profiler repo – per-image embeddings:
    https://github.com/pfizer-opensource/moa-profiler
  - CellPaintSSL repo – well-level feature format for downstream profiling:
    https://github.com/Bayer-Group/CellPaintSSL

Usage:
  python convert_moa_profiler_cellpaintssl.py \
    --emb /path/to/per_image_embeddings_with_filenames.csv \
    --meta /path/to/compound_images_with_MOA_no_polypharm_well_excluded.csv \
    --well-out /path/to/well_features.csv \
    --agg median
"""

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# Import CellPaintSSL aggregation helper from source/inference_utils.py
CPSSL_ROOT = "/home/usr/Cell_Painting_Tools/CellPaintSSL/CellPaintSSL/Repo/CellPaintSSL"
CPSSL_SOURCE = os.path.join(CPSSL_ROOT, "source")

if not os.path.isdir(CPSSL_SOURCE):
    raise ImportError(
        f"Expected CellPaintSSL 'source' dir at: {CPSSL_SOURCE}\n"
        "Please verify the path to the CellPaintSSL repo."
    )

# Add the repo root so 'source' is importable as a package
if CPSSL_ROOT not in sys.path:
    sys.path.insert(0, CPSSL_ROOT)

try:
    # IMPORTANT: import as a package module, not a bare file
    from source.inference_utils import aggregate_embeddings_plate
except Exception as e:
    raise ImportError(
        "Could not import 'aggregate_embeddings_plate' from "
        f"'{os.path.join(CPSSL_SOURCE, 'inference_utils.py')}'.\n"
        "Make sure 'source/__init__.py' exists (it does in CellPaintSSL) "
        "and that the repo root was added to sys.path."
    ) from e

DEFAULT_BATCH = "BATCH_1"
DEFAULT_PLATE = "PLATE_1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="convert_moa_profiler_cellpaintssl.py",
        description="Convert MOA Profiler per-image outputs to CellPaintSSL-style WELL features (strict filename merge, aggregation via aggregate_embeddings_plate)."
    )
    # Order: --emb before --meta
    parser.add_argument("--emb", required=True, help="Path to per_image_embeddings_with_filenames.csv (must have 'imagename' and dim0..dimN)")
    parser.add_argument("--meta", required=True, help="Path to compound_images_with_MOA_no_polypharm_well_excluded.csv (must have 'imagename','perturbation','moas')")
    parser.add_argument("--well-out", required=True, help="Path to output well_features.csv")
    parser.add_argument(
        "--agg",
        default="median",
        choices=["median", "mean"],
        help="Aggregation for well features (default: median)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1024,
        help="Minibatch size to simulate for aggregate_embeddings_plate input (default: 1024)"
    )
    return parser.parse_args()


def extract_well_from_path(path: str) -> str:
    """
    Derive well ID from path by taking the parent folder of the TIFF and splitting on '_',
    then taking the first token:
      .../A01_s5/ch1.tif  -> A01
      .../H12_s3/ch1.tif  -> H12
      .../B03/ch1.tif     -> B03
    """
    parts = os.path.normpath(str(path)).split(os.sep)
    folder = parts[-2] if len(parts) >= 2 else ""
    return folder.split("_")[0] if folder else ""


def assert_required_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {where}: {missing}. "
            f"Expected exactly {cols} (plus optional 'batch'/'plate' in metadata)."
        )


def get_dim_columns(df: pd.DataFrame) -> List[str]:
    """
    Return embedding columns strictly matching dim0..dimN, sorted by N.
    """
    dim_cols = [c for c in df.columns if c.startswith("dim")]
    if not dim_cols:
        raise ValueError("No embedding columns found; expected dim0..dimN in embeddings CSV.")
    try:
        dim_cols_sorted = sorted(dim_cols, key=lambda x: int(x.replace("dim", "")))
    except Exception as e:
        raise ValueError("Embedding columns must be named dim0..dimN with integer suffixes.") from e
    return dim_cols_sorted


def build_well_features(
    emb_csv: str,
    meta_csv: str,
    agg: str = "median",
    chunk_size: int = 1024,
) -> pd.DataFrame:

    # Load inputs (embeddings and metadata)
    emb_df = pd.read_csv(emb_csv)
    meta_df = pd.read_csv(meta_csv)

    # Verify all required column names exist
    assert_required_columns(emb_df, ["imagename"], "embeddings")
    assert_required_columns(meta_df, ["imagename", "perturbation", "moas"], "metadata")

    # Get embedding columns (dim0..dimN) from input
    dim_cols = get_dim_columns(emb_df)

    # Calculate wells based on image names
    meta_df = meta_df.copy()
    meta_df["well"] = meta_df["imagename"].apply(extract_well_from_path)

    # Use batch / plate defaults if absent in metadata
    if "batch" not in meta_df.columns:
        meta_df["batch"] = DEFAULT_BATCH
    if "plate" not in meta_df.columns:
        meta_df["plate"] = DEFAULT_PLATE

    # Set perturbation_id in output = perturbation from input
    meta_df["perturbation_id"] = meta_df["perturbation"].astype(str)

    # Set target in output = moas from input
    meta_df["target"] = meta_df["moas"]

    # Merge embeddings with metadata on image names
    joined = pd.merge(
        emb_df[["imagename"] + dim_cols],
        meta_df[["imagename", "batch", "plate", "well", "perturbation_id", "target"]],
        on="imagename",
        how="inner",
        validate="one_to_one",
    )

    # ERROR and abort if any rows don't match
    if len(joined) != len(meta_df):
        missing = len(meta_df) - len(joined)
        raise RuntimeError(
            f"[ERROR] {missing} metadata rows had no matching embeddings by 'imagename'. "
            "Aborting. Ensure both CSVs contain identical 'imagename' values."
        )

    # Prepare inputs as expected for aggregate_embeddings_plate():
    # plate_dfr: pandas.DataFrame with columns ['batch','plate','well','perturbation_id','target'],
    #            one row per image; length N.
    plate_dfr = joined[["batch", "plate", "well", "perturbation_id", "target"]].reset_index(drop=True)

    # em_mat: NumPy array of embeddings with shape (N, D),
    #         where N = number of images and D = number of dim* columns.
    em_mat = joined[dim_cols].to_numpy(dtype=float)              # (N, D)
    N, D = em_mat.shape

    # reshape: add a singleton Ncrops axis so each per-image embedding has shape (1, D),
    #          yielding an array of shape (N, 1, D).
    em_mat = em_mat.reshape(N, 1, D)

    # plate_embs: list of minibatches; each element has shape (batch_size, 1, D).
    #             The last chunk may be smaller than chunk_size.
    plate_embs = [em_mat[i:i + chunk_size] for i in range(0, N, chunk_size)]

    # Call CellPaintSSL aggregate_embeddings_plate() utility function to aggregate by well
    # NOTE: columns dim0..dimN will be replaced with emb0..embN
    agg_df = aggregate_embeddings_plate(
        plate_dfr=plate_dfr,
        plate_embs=plate_embs,
        my_cols=['batch', 'plate', 'well', 'perturbation_id', 'target'],
        operation=agg
    )

    # Clean up final column order: metadata -> embeddings
    emb_cols_out = [c for c in agg_df.columns if c.startswith("emb")]
    # Sort embeddings by numeric suffix so emb0..embN are in order
    emb_cols_out = sorted(emb_cols_out, key=lambda x: int(x.replace("emb", "")))

    # Define required metadata key order for well_features schema
    keys = ["batch", "plate", "well", "perturbation_id", "target"]
    # Reorder columns: metadata keys first, then embedding columns (emb0..embN)
    agg_df = agg_df[keys + emb_cols_out]

    return agg_df


def moa_to_cpssl(
    emb_csv: str,
    meta_csv: str,
    well_out: str,
    agg: str = "median",
    chunk_size: int = 1024,
) -> pd.DataFrame:
    """
    Convert MOA Profiler per-image outputs to CellPaintSSL-style WELL features.
    Writes to `well_out` and returns the aggregated DataFrame.
    """
    # File existence checks
    for p in (emb_csv, meta_csv):
        if not os.path.exists(p):
            print(f"[ERROR] File not found: {p}", file=sys.stderr)
            raise FileNotFoundError(p)

    # Build the well-level features
    well_df = build_well_features(
        emb_csv=emb_csv,
        meta_csv=meta_csv,
        agg=agg,
        chunk_size=chunk_size,
    )

    # Save and report
    well_df.to_csv(well_out, index=False)
    print(f"[OK] Well-level features saved: {well_out}")
    print(f"[INFO] Shape: {well_df.shape[0]} wells x {well_df.shape[1]} cols")
    print(f"[INFO] First columns: {list(well_df.columns[:5])} | Last emb col: {well_df.columns[-1]}")

    return well_df


def main():
    # Get input arguments
    args = parse_args()

    # Convert MOA Profiler output to CellPaintSSL style
    moa_to_cpssl(
        emb_csv=args.emb,
        meta_csv=args.meta,
        well_out=args.well_out,
        agg=args.agg,
        chunk_size=args.chunk_size,
    )


if __name__ == "__main__":
    main()
