#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC full-256 healthy latent-space source/position control.

Purpose
-------
After removing padded edge patches, check whether remaining latent-space structure
is explained by:
1) spatial patch position (r0c0, r0c1, r1c0, r1c1, other)
2) original source image resolution (512x512, 369x369, other)

No model inference, no retraining, no dimensionality reduction rerun.
This script only re-colors the already computed FULL-256 coordinate CSV files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_coords(path: Path):
    df = pd.read_csv(path)
    required = [
        "row_idx", "col_idx",
        "original_width", "original_height"
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    return df


def add_groups(df: pd.DataFrame):
    df = df.copy()

    df["position_group"] = (
        "r" + df["row_idx"].astype(int).astype(str)
        + "c" + df["col_idx"].astype(int).astype(str)
    )
    main_positions = {"r0c0", "r0c1", "r1c0", "r1c1"}
    df.loc[~df["position_group"].isin(main_positions), "position_group"] = "other"

    res = (
        df["original_width"].astype(int).astype(str)
        + "x"
        + df["original_height"].astype(int).astype(str)
    )
    df["resolution_group"] = np.where(
        res == "512x512",
        "512x512",
        np.where(res == "369x369", "369x369", "other"),
    )

    df["combined_group"] = (
        df["resolution_group"] + " / " + df["position_group"]
    )
    return df


def plot_grouped(df, xcol, ycol, group_col, title, out_png):
    fig, ax = plt.subplots(figsize=(11, 8.5))

    counts = df[group_col].value_counts()
    ordered = list(counts.index)

    for group in ordered:
        sub = df[df[group_col] == group]
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=14,
            alpha=0.68,
            label=f"{group} (n={len(sub)})",
        )

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def centroid_table(df, xcol, ycol, group_col):
    rows = []
    for group, sub in df.groupby(group_col):
        rows.append({
            "group": group,
            "n": int(len(sub)),
            f"mean_{xcol}": float(sub[xcol].mean()),
            f"std_{xcol}": float(sub[xcol].std(ddof=0)),
            f"mean_{ycol}": float(sub[ycol].mean()),
            f"std_{ycol}": float(sub[ycol].std(ddof=0)),
        })
    return pd.DataFrame(rows).sort_values("n", ascending=False)


def process_method(
    csv_path: Path,
    xcol: str,
    ycol: str,
    method_name: str,
    out: Path,
):
    df = add_groups(load_coords(csv_path))

    plot_grouped(
        df, xcol, ycol, "position_group",
        f"Healthy FULL 256x256 latent {method_name} — patch position",
        out / f"{method_name.lower()}_full256_by_position.png",
    )

    plot_grouped(
        df, xcol, ycol, "resolution_group",
        f"Healthy FULL 256x256 latent {method_name} — source resolution",
        out / f"{method_name.lower()}_full256_by_resolution.png",
    )

    plot_grouped(
        df, xcol, ycol, "combined_group",
        f"Healthy FULL 256x256 latent {method_name} — resolution / position",
        out / f"{method_name.lower()}_full256_by_resolution_position.png",
    )

    pos_cent = centroid_table(df, xcol, ycol, "position_group")
    res_cent = centroid_table(df, xcol, ycol, "resolution_group")
    comb_cent = centroid_table(df, xcol, ycol, "combined_group")

    pos_cent.to_csv(out / f"{method_name.lower()}_position_centroids.csv", index=False)
    res_cent.to_csv(out / f"{method_name.lower()}_resolution_centroids.csv", index=False)
    comb_cent.to_csv(out / f"{method_name.lower()}_resolution_position_centroids.csv", index=False)

    return df, pos_cent, res_cent, comb_cent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pca_csv", required=True)
    ap.add_argument("--lle_csv", required=True)
    ap.add_argument("--tsne_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 94)
    print("BRISC FULL-256 HEALTHY LATENT SOURCE / POSITION CONTROL")
    print("=" * 94)

    pca_df, pca_pos, pca_res, pca_comb = process_method(
        Path(args.pca_csv), "pc1", "pc2", "PCA", out
    )
    print("[1/3] PCA controls created.")

    lle_df, lle_pos, lle_res, lle_comb = process_method(
        Path(args.lle_csv), "lle1", "lle2", "LLE", out
    )
    print("[2/3] LLE controls created.")

    tsne_df, tsne_pos, tsne_res, tsne_comb = process_method(
        Path(args.tsne_csv), "tsne1", "tsne2", "t-SNE", out
    )
    print("[3/3] t-SNE controls created.")

    # Compact numeric summary based on t-SNE because its islands are visually strongest.
    summary = {
        "n_full256": int(len(tsne_df)),
        "position_counts": {
            k: int(v)
            for k, v in tsne_df["position_group"].value_counts().to_dict().items()
        },
        "resolution_counts": {
            k: int(v)
            for k, v in tsne_df["resolution_group"].value_counts().to_dict().items()
        },
        "tsne_resolution_position_centroids": (
            tsne_comb.to_dict(orient="records")
        ),
        "note": (
            "This is a control analysis only. Separation by position/resolution "
            "indicates acquisition/spatial-content structure in the latent space; "
            "it must not be interpreted as clinical class separation."
        ),
    }

    (out / "source_position_control_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Key counts:")
    print(tsne_df["resolution_group"].value_counts())
    print()
    print(tsne_df["position_group"].value_counts())
    print()
    print("t-SNE combined centroids:")
    print(tsne_comb.to_string(index=False))
    print()
    print("Saved to:", out)
    print("Most useful first plot:")
    print(" ", out / "t-sne_full256_by_resolution_position.png")


if __name__ == "__main__":
    main()
