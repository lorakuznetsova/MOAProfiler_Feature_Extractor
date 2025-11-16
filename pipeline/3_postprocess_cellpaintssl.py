#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
#
# This file is adapted from the CellPaintSSL project:
#   https://github.com/Bayer-Group/CellPaintSSL
# which is licensed under the BSD 3-Clause License.
#
# Modifications Copyright (c) 2025, Xenia Kuznetsova and Larisa Kuznetsova.
# See the accompanying LICENSE files in this repository root
# for the full license text.
"""
3_postprocess_cellpaintssl.py

Standalone post-processing of per-well embeddings (CellPaintSSL-style).
Implements the same normalization combos as inference.py, using pycytominer.

Inputs
------
--embedding_csv : CSV with per-well embeddings BEFORE post-processing.
                  Must include columns like: batch, plate, well, perturbation_id, target, emb*
--val_csv       : The same "validation" CSV used at inference time (to merge target metadata).
--operation     : Aggregation operation for consensus/agg output: mean|median
--norm_method   : Normalization method or combo. Default = spherize_mad_robustize
                  Available (for reference):
                    - standardize
                    - mad_robustize
                    - spherize
                    - spherize_mad_robustize
                    - mad_robustize_spherize
                    - spherize_standardize
                    - standardize_spherize
                    - no_post_proc
--l2norm        : If provided, apply L2 normalization (per feature) before the method.
--out_well_csv  : Output path for per-well post-processed embeddings.
--out_agg_csv   : Output path for aggregated (by perturbation_id) profiles.

Example
-------
python 3_postprocess_cellpaintssl.py \
  --embedding_csv /path/to/MOAProfiler_OUT/well_features.csv \
  --val_csv       /path/to/MOAProfiler_IN/dataset.csv \
  --operation     median \
  --norm_method   spherize_mad_robustize \
  --out_well_csv  /path/to/MOAProfiler_OUT/well_features_normalized.csv \
  --out_agg_csv   /path/to/MOAProfiler_OUT/agg_features_from_normalized.csv
"""

import os
import argparse
import pandas as pd
import numpy as np
import pycytominer  # used exactly like in inference_utils.py


# ---- Exact copies of functions from inference_utils.py (semantics preserved) ----

def postprocess_embeddings(
    profiles_df,
    var_thresh=1e-5,
    trt_column='perturbation_id',
    norm_method='spherize',
    l2_norm=False
):
    emb_features = [c for c in profiles_df.columns if c.startswith('emb')]
    emb_meta_features = [c for c in profiles_df.columns if c not in emb_features]

    # Optional L2 normalization (per feature/column)
    if l2_norm:
        from sklearn.preprocessing import normalize as sk_normalize
        profiles_df[emb_features] = sk_normalize(profiles_df[emb_features], axis=0, norm='l2')

    if norm_method == "no_post_proc":
        return profiles_df

    emb_df = profiles_df.loc[:, emb_features]
    meta_df = profiles_df.loc[:, emb_meta_features]

    # Filter low-variance features prior to normalization
    emb_vars_samples = np.var(emb_df[meta_df[trt_column] != 'DMSO'], axis=0)
    emb_vars_dmso = np.var(emb_df[meta_df[trt_column] == 'DMSO'], axis=0)
    my_cond = np.logical_and(emb_vars_samples > var_thresh, emb_vars_dmso > var_thresh)

    emb_df_filt = emb_df.loc[:, my_cond]
    # NOTE: keep reset_index() as in your original (index becomes metadata, harmless)
    profiles_df = pd.concat([meta_df, emb_df_filt], axis=1).reset_index()
    emb_features = [c for c in profiles_df.columns if c.startswith('emb')]

    # Per-plate normalization methods (fit/transform per plate)
    if norm_method in ["standardize", "mad_robustize", "robustize"]:
        norm_prof = []
        samples = (trt_column + " == 'DMSO'") if norm_method == "standardize" else ("~" + trt_column + ".isnull()")
        for plate in profiles_df['plate'].unique():
            norm_df = profiles_df.query('plate == @plate').reset_index(drop=True).copy()
            norm_df = pycytominer.normalize(
                profiles=norm_df,
                features=emb_features,
                meta_features=[c for c in norm_df.columns if c not in emb_features],
                samples=samples,
                method=norm_method,
                output_file='none'
            )
            norm_prof.append(norm_df)
        norm_prof = pd.concat(norm_prof).reset_index(drop=True)
        return norm_prof

    # Sphering is fit on all negative controls (across plates) by design here
    elif norm_method == "spherize":
        profiles_df = pycytominer.normalize(
            profiles=profiles_df,
            features=emb_features,
            meta_features=emb_meta_features + (['index'] if 'index' in profiles_df.columns else []),
            samples=(trt_column + " == 'DMSO'"),
            method=norm_method,
            output_file='none'
        )
    else:
        raise ValueError("Invalid norm_method specified")

    return profiles_df


def post_proc(
    embedding_df,
    val_df,
    trt_column='perturbation_id',
    cols=['perturbation_id', 'target'],
    operation='mean',
    norm_method='spherize',
    l2_norm=False
):
    embeddings_proc_well = postprocess_embeddings(
        embedding_df,
        var_thresh=1e-5,
        trt_column=trt_column,
        norm_method=norm_method,
        l2_norm=l2_norm
    )

    emb_features = [c for c in embeddings_proc_well.columns if c.startswith('emb')]

    # Aggregate to consensus profiles by perturbation_id
    embeddings_proc_agg = pycytominer.aggregate(
        embeddings_proc_well,
        strata=[trt_column],
        features=emb_features,
        operation=operation
    )
    embeddings_proc_agg = pd.merge(
        left=embeddings_proc_agg,
        right=val_df.loc[:, cols].drop_duplicates(),
        how='left'
    )

    # (Note: DMSO rows are NOT dropped, mirroring inference_utils.py)
    return embeddings_proc_well, embeddings_proc_agg


# ---- CLI wrapper that mirrors inference.py combo handling ----

def main():
    parser = argparse.ArgumentParser(
        description="Standalone post-processing identical to inference.py post_proc"
    )
    parser.add_argument(
        "--embedding_csv", required=True,
        help="Per-well embeddings CSV produced BEFORE post-processing (columns: plate, perturbation_id, target, emb*)"
    )
    parser.add_argument(
        "--val_csv", required=True,
        help="The same validation CSV used at inference time (to merge target metadata)"
    )
    parser.add_argument(
        "--operation", default="mean", choices=["mean", "median"],
        help="Aggregation op for the consensus/agg output"
    )
    parser.add_argument(
        "--norm_method", default="spherize_mad_robustize",
        choices=[
            "standardize", "mad_robustize",
            "spherize", "spherize_mad_robustize",
            "mad_robustize_spherize", "spherize_standardize",
            "standardize_spherize", "no_post_proc"
        ],
        help="Normalization method or combo (identical to inference.py options). Default: spherize_mad_robustize"
    )
    parser.add_argument(
        "--l2norm", action="store_true",
        help="Apply L2 normalization before the method (exactly as in inference_utils)"
    )
    parser.add_argument(
        "--out_well_csv", required=True,
        help="Path to write the per-well post-processed embeddings CSV"
    )
    parser.add_argument(
        "--out_agg_csv", required=True,
        help="Path to write the aggregated (by perturbation_id) CSV"
    )
    args = parser.parse_args()

    # Load inputs
    embedding_df = pd.read_csv(args.embedding_csv)
    val_df = pd.read_csv(args.val_csv)

    # Combo handling (same semantics as your original)
    nm = args.norm_method
    if 'spherize_' in nm:
        # first spherize, then the trailing method
        well1, _ = post_proc(
            embedding_df, val_df,
            operation=args.operation,
            norm_method='spherize',
            l2_norm=args.l2norm
        )
        well2, agg2 = post_proc(
            well1, val_df,
            operation=args.operation,
            norm_method=nm.replace('spherize_', ''),
            l2_norm=args.l2norm
        )
        well_out, agg_out = well2, agg2

    elif '_spherize' in nm:
        # first the leading method, then spherize
        leading = nm.replace('_spherize', '')
        well1, _ = post_proc(
            embedding_df, val_df,
            operation=args.operation,
            norm_method=leading,
            l2_norm=args.l2norm
        )
        well2, agg2 = post_proc(
            well1, val_df,
            operation=args.operation,
            norm_method='spherize',
            l2_norm=args.l2norm
        )
        well_out, agg_out = well2, agg2

    else:
        # single method (or no_post_proc)
        well_out, agg_out = post_proc(
            embedding_df, val_df,
            operation=args.operation,
            norm_method=nm,
            l2_norm=args.l2norm
        )

    # Save outputs (no subfolders; write exactly where requested)
    os.makedirs(os.path.dirname(args.out_well_csv), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_agg_csv), exist_ok=True)
    well_out.to_csv(args.out_well_csv, index=False)
    agg_out.to_csv(args.out_agg_csv, index=False)
    print(f"✅ Wrote:\n  per-well -> {args.out_well_csv}\n  aggregate -> {args.out_agg_csv}")


if __name__ == "__main__":
    main()

