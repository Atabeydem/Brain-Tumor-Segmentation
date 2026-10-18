#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC healthy latent-space padding control.

Two goals
---------
A) Re-color the EXISTING all-healthy 2D PCA/LLE/t-SNE coordinates by patch geometry:
   - full_256x256
   - padded_edge
   This directly tests whether the previously observed structure follows padding.

B) Re-run PCA/LLE/t-SNE using ONLY true full 256x256 healthy patches
   (valid_width=256 and valid_height=256), producing:
   - PCA 2D + interactive PCA 3D
   - LLE 2D + interactive LLE 3D
   - t-SNE 2D + interactive t-SNE 3D

No model training. No embedding averaging. No resizing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE

try:
    import plotly.graph_objects as go
except Exception as e:
    raise RuntimeError(
        "Plotly is required. Install once with: pip install plotly\n"
        f"Original import error: {e}"
    )


def load_npz(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)

    required = [
        "embeddings",
        "image",
        "source_image",
        "patch_id",
        "row_idx",
        "col_idx",
        "valid_width",
        "valid_height",
        "original_width",
        "original_height",
    ]
    missing = [k for k in required if k not in d.files]
    if missing:
        raise RuntimeError(f"Missing NPZ keys: {missing}")

    x = d["embeddings"].astype(np.float32)

    meta = pd.DataFrame({
        "image": d["image"].astype(str),
        "source_image": d["source_image"].astype(str),
        "patch_id": d["patch_id"].astype(str),
        "row_idx": d["row_idx"].astype(int),
        "col_idx": d["col_idx"].astype(int),
        "valid_width": d["valid_width"].astype(int),
        "valid_height": d["valid_height"].astype(int),
        "original_width": d["original_width"].astype(int),
        "original_height": d["original_height"].astype(int),
    })

    if len(x) != len(meta):
        raise RuntimeError("Embedding/meta row count mismatch.")

    return x, meta


def geometry_label(df):
    full = (df["valid_width"] == 256) & (df["valid_height"] == 256)
    return np.where(full, "full_256x256", "padded_edge")


def recolor_existing(csv_path: Path, xcol: str, ycol: str, title: str, out_png: Path):
    df = pd.read_csv(csv_path)
    df["geometry"] = geometry_label(df)

    fig, ax = plt.subplots(figsize=(10, 8))

    for group in ["full_256x256", "padded_edge"]:
        sub = df[df["geometry"] == group]
        ax.scatter(
            sub[xcol],
            sub[ycol],
            s=12,
            alpha=0.65,
            label=f"{group} (n={len(sub)})",
        )

    ax.set_title(title)
    ax.set_xlabel(xcol)
    ax.set_ylabel(ycol)
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return {
        "full_256x256": int((df["geometry"] == "full_256x256").sum()),
        "padded_edge": int((df["geometry"] == "padded_edge").sum()),
    }


def save_2d(coords, title, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:, 0], coords[:, 1], s=12, alpha=0.65)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def hover_text(meta):
    out = []
    for _, r in meta.iterrows():
        out.append(
            f"image={r['image']}<br>"
            f"patch={r['patch_id']}<br>"
            f"row={r['row_idx']}, col={r['col_idx']}<br>"
            f"valid={r['valid_width']}x{r['valid_height']}<br>"
            f"original={r['original_width']}x{r['original_height']}"
        )
    return out


def save_3d(coords, meta, title, out_html):
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=dict(size=3, opacity=0.65),
                text=hover_text(meta),
                hovertemplate=(
                    "%{text}<br>"
                    "C1=%{x:.5f}<br>"
                    "C2=%{y:.5f}<br>"
                    "C3=%{z:.5f}<extra></extra>"
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            zaxis_title="Component 3",
        ),
        margin=dict(l=0, r=0, b=0, t=50),
    )
    fig.write_html(
        str(out_html),
        include_plotlyjs=True,
        full_html=True,
    )


def save_coords(meta, coords, cols, path):
    df = meta.copy()
    for i, c in enumerate(cols):
        df[c] = coords[:, i]
    df.to_csv(path, index=False)


def make_tsne(x, n_components, seed, perplexity, iterations):
    kwargs = dict(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        method="barnes_hut",
        angle=0.5,
    )
    try:
        model = TSNE(max_iter=iterations, **kwargs)
    except TypeError:
        model = TSNE(n_iter=iterations, **kwargs)
    return model.fit_transform(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--existing_vis_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lle_neighbors", type=int, default=20)
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_iterations", type=int, default=1500)
    args = ap.parse_args()

    emb_path = Path(args.embeddings)
    existing = Path(args.existing_vis_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x, meta = load_npz(emb_path)

    full_mask = (
        (meta["valid_width"].to_numpy() == 256)
        & (meta["valid_height"].to_numpy() == 256)
    )

    x_full = x[full_mask]
    meta_full = meta.loc[full_mask].reset_index(drop=True)

    n_all = len(x)
    n_full = len(x_full)
    n_pad = n_all - n_full

    print("=" * 94)
    print("HEALTHY LATENT-SPACE PADDING CONTROL")
    print("=" * 94)
    print(f"All healthy patch vectors : {n_all}")
    print(f"Full 256x256 patches      : {n_full}")
    print(f"Padded edge patches       : {n_pad}")
    print(f"Latent dimensions         : {x.shape[1]}")
    print("Embedding averaging       : NONE")
    print()

    # ------------------------------------------------------------
    # A) Re-color existing 2D projections by geometry
    # ------------------------------------------------------------
    print("[A1] Coloring existing PCA 2D by full-vs-padded geometry...")
    recolor_existing(
        existing / "pca_2d_coordinates.csv",
        "pc1", "pc2",
        "Healthy latent PCA 2D — patch geometry",
        out / "all_pca_2d_full_vs_padded.png",
    )

    print("[A2] Coloring existing LLE 2D by full-vs-padded geometry...")
    recolor_existing(
        existing / "lle_2d_coordinates.csv",
        "lle1", "lle2",
        "Healthy latent LLE 2D — patch geometry",
        out / "all_lle_2d_full_vs_padded.png",
    )

    print("[A3] Coloring existing t-SNE 2D by full-vs-padded geometry...")
    recolor_existing(
        existing / "tsne_2d_coordinates.csv",
        "tsne1", "tsne2",
        "Healthy latent t-SNE 2D — patch geometry",
        out / "all_tsne_2d_full_vs_padded.png",
    )

    # ------------------------------------------------------------
    # B) Re-run reductions using ONLY true 256x256 patches
    # ------------------------------------------------------------
    lle_neighbors = min(args.lle_neighbors, n_full - 1)
    perplexity = min(args.tsne_perplexity, max(5.0, (n_full - 1) / 3.0))

    summary = {
        "input_embeddings": str(emb_path),
        "all_healthy_patches": int(n_all),
        "full_256x256_patches": int(n_full),
        "padded_edge_patches_excluded": int(n_pad),
        "latent_dim": int(x.shape[1]),
        "embedding_averaging": False,
        "pre_scaling": False,
        "pre_normalization": False,
        "seed": args.seed,
        "lle_neighbors": int(lle_neighbors),
        "tsne_perplexity": float(perplexity),
        "tsne_iterations": int(args.tsne_iterations),
    }

    print("[B1/B2] Full-only PCA 2D / 3D...")
    pca3 = PCA(n_components=3)
    pca3_coords = pca3.fit_transform(x_full)
    pca2_coords = pca3_coords[:, :2]

    save_2d(
        pca2_coords,
        "Healthy FULL 256x256 patch latent vectors — PCA 2D",
        out / "full256_pca_2d.png",
    )
    save_3d(
        pca3_coords,
        meta_full,
        "Healthy FULL 256x256 patch latent vectors — PCA 3D",
        out / "full256_pca_3d_interactive.html",
    )
    save_coords(
        meta_full, pca2_coords, ["pc1", "pc2"],
        out / "full256_pca_2d_coordinates.csv",
    )
    save_coords(
        meta_full, pca3_coords, ["pc1", "pc2", "pc3"],
        out / "full256_pca_3d_coordinates.csv",
    )

    summary["pca_explained_variance_ratio"] = [
        float(v) for v in pca3.explained_variance_ratio_
    ]
    summary["pca_first_2_total_variance"] = float(
        pca3.explained_variance_ratio_[:2].sum()
    )
    summary["pca_first_3_total_variance"] = float(
        pca3.explained_variance_ratio_[:3].sum()
    )

    print("[B3] Full-only LLE 2D...")
    lle2 = LocallyLinearEmbedding(
        n_neighbors=lle_neighbors,
        n_components=2,
        method="standard",
        eigen_solver="arpack",
        random_state=args.seed,
        n_jobs=-1,
    )
    lle2_coords = lle2.fit_transform(x_full)
    save_2d(
        lle2_coords,
        "Healthy FULL 256x256 patch latent vectors — LLE 2D",
        out / "full256_lle_2d.png",
    )
    save_coords(
        meta_full, lle2_coords, ["lle1", "lle2"],
        out / "full256_lle_2d_coordinates.csv",
    )

    print("[B4] Full-only LLE 3D...")
    lle3 = LocallyLinearEmbedding(
        n_neighbors=lle_neighbors,
        n_components=3,
        method="standard",
        eigen_solver="arpack",
        random_state=args.seed,
        n_jobs=-1,
    )
    lle3_coords = lle3.fit_transform(x_full)
    save_3d(
        lle3_coords,
        meta_full,
        "Healthy FULL 256x256 patch latent vectors — LLE 3D",
        out / "full256_lle_3d_interactive.html",
    )
    save_coords(
        meta_full, lle3_coords, ["lle1", "lle2", "lle3"],
        out / "full256_lle_3d_coordinates.csv",
    )

    summary["lle_2d_reconstruction_error"] = float(lle2.reconstruction_error_)
    summary["lle_3d_reconstruction_error"] = float(lle3.reconstruction_error_)

    print("[B5] Full-only t-SNE 2D...")
    tsne2 = make_tsne(
        x_full, 2, args.seed, perplexity, args.tsne_iterations
    )
    save_2d(
        tsne2,
        "Healthy FULL 256x256 patch latent vectors — t-SNE 2D",
        out / "full256_tsne_2d.png",
    )
    save_coords(
        meta_full, tsne2, ["tsne1", "tsne2"],
        out / "full256_tsne_2d_coordinates.csv",
    )

    print("[B6] Full-only t-SNE 3D...")
    tsne3 = make_tsne(
        x_full, 3, args.seed, perplexity, args.tsne_iterations
    )
    save_3d(
        tsne3,
        meta_full,
        "Healthy FULL 256x256 patch latent vectors — t-SNE 3D",
        out / "full256_tsne_3d_interactive.html",
    )
    save_coords(
        meta_full, tsne3, ["tsne1", "tsne2", "tsne3"],
        out / "full256_tsne_3d_coordinates.csv",
    )

    (out / "padding_control_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 94)
    print("DONE")
    print("=" * 94)
    print("Geometry-colored controls:")
    print(" ", out / "all_pca_2d_full_vs_padded.png")
    print(" ", out / "all_lle_2d_full_vs_padded.png")
    print(" ", out / "all_tsne_2d_full_vs_padded.png")
    print()
    print("Full-256-only six visualizations:")
    print(" ", out / "full256_pca_2d.png")
    print(" ", out / "full256_pca_3d_interactive.html")
    print(" ", out / "full256_lle_2d.png")
    print(" ", out / "full256_lle_3d_interactive.html")
    print(" ", out / "full256_tsne_2d.png")
    print(" ", out / "full256_tsne_3d_interactive.html")
    print()
    print("Summary:")
    print(" ", out / "padding_control_summary.json")


if __name__ == "__main__":
    main()
