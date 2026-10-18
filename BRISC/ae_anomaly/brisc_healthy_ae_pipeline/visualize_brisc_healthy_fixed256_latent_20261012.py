#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC healthy fixed-256 latent-space visualization.

Input
-----
train_healthy_fixed256_patch_embeddings_2048d.npz

Expected current shape:
    3517 healthy patches x 2048 latent dimensions

Produces the 6 visualizations requested in the meeting:
    1) PCA  2D
    2) PCA  3D
    3) LLE  2D
    4) LLE  3D
    5) t-SNE 2D
    6) t-SNE 3D

Important
---------
- HEALTHY patches only.
- Uses each patch embedding separately.
- NO image-level averaging.
- Raw 2048-D embeddings are used by default; no hidden normalization/scaling.
- 3D plots are saved as standalone interactive HTML files and can be rotated.
- Coordinate CSV files are also saved so the 3D data can be re-opened elsewhere.
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
        "Plotly is required for rotatable 3D HTML plots.\n"
        "Install once with: pip install plotly\n"
        f"Original import error: {e}"
    )


def load_embeddings(npz_path: Path):
    data = np.load(npz_path, allow_pickle=True)

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
    missing = [k for k in required if k not in data.files]
    if missing:
        raise RuntimeError(f"Missing keys in NPZ: {missing}")

    x = data["embeddings"].astype(np.float32)

    meta = pd.DataFrame({
        "image": data["image"].astype(str),
        "source_image": data["source_image"].astype(str),
        "patch_id": data["patch_id"].astype(str),
        "row_idx": data["row_idx"].astype(int),
        "col_idx": data["col_idx"].astype(int),
        "valid_width": data["valid_width"].astype(int),
        "valid_height": data["valid_height"].astype(int),
        "original_width": data["original_width"].astype(int),
        "original_height": data["original_height"].astype(int),
    })

    finite = np.isfinite(x).all(axis=1)
    if not finite.all():
        bad = int((~finite).sum())
        print(f"WARNING: dropping {bad} rows containing NaN/Inf.")
        x = x[finite]
        meta = meta.loc[finite].reset_index(drop=True)

    return x, meta


def maybe_sample(x, meta, max_points, seed):
    if max_points <= 0 or len(x) <= max_points:
        return x, meta

    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(x), size=max_points, replace=False))
    print(f"NOTE: --max_points={max_points}; using a reproducible subset.")
    return x[idx], meta.iloc[idx].reset_index(drop=True)


def save_coords(meta, coords, names, out_csv):
    df = meta.copy()
    for i, name in enumerate(names):
        df[name] = coords[:, i]
    df.to_csv(out_csv, index=False)
    return df


def save_2d_png(coords, title, out_png):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.65)
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


def save_3d_html(coords, meta, title, out_html):
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
                    "C3=%{z:.5f}"
                    "<extra></extra>"
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

    # Standalone HTML: plotly.js is embedded, so the file remains rotatable offline.
    fig.write_html(
        str(out_html),
        include_plotlyjs=True,
        full_html=True,
    )


def make_tsne(n_components, x, seed, perplexity, iterations):
    kwargs = dict(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        method="barnes_hut",
        angle=0.5,
    )

    # sklearn renamed n_iter -> max_iter in newer versions.
    try:
        model = TSNE(max_iter=iterations, **kwargs)
    except TypeError:
        model = TSNE(n_iter=iterations, **kwargs)

    return model.fit_transform(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lle_neighbors", type=int, default=20)
    ap.add_argument("--tsne_perplexity", type=float, default=30.0)
    ap.add_argument("--tsne_iterations", type=int, default=1500)
    ap.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="0 = use ALL healthy patches. Nonzero is only for quick debugging.",
    )
    args = ap.parse_args()

    npz_path = Path(args.embeddings)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    x, meta = load_embeddings(npz_path)
    x, meta = maybe_sample(x, meta, args.max_points, args.seed)

    n, d = x.shape

    if n < 10:
        raise RuntimeError(f"Too few samples: n={n}")

    perplexity = min(float(args.tsne_perplexity), max(5.0, (n - 1) / 3.0))
    lle_neighbors = min(int(args.lle_neighbors), n - 1)

    print("=" * 92)
    print("BRISC HEALTHY LATENT-SPACE DIMENSION REDUCTION")
    print("=" * 92)
    print(f"Input                  : {npz_path}")
    print(f"Healthy patch vectors  : {n}")
    print(f"Original dimensions    : {d}")
    print("Embedding averaging    : NONE")
    print("Pre-scaling            : NONE")
    print("Pre-normalization      : NONE")
    print(f"LLE neighbors          : {lle_neighbors}")
    print(f"t-SNE perplexity       : {perplexity}")
    print(f"t-SNE iterations       : {args.tsne_iterations}")
    print(f"Random seed            : {args.seed}")
    print()

    summary = {
        "input": str(npz_path),
        "n_healthy_patches": int(n),
        "original_dim": int(d),
        "embedding_averaging": False,
        "pre_scaling": False,
        "pre_normalization": False,
        "seed": int(args.seed),
        "lle_neighbors": int(lle_neighbors),
        "tsne_perplexity": float(perplexity),
        "tsne_iterations": int(args.tsne_iterations),
    }

    # ------------------------------------------------------------
    # PCA
    # ------------------------------------------------------------
    print("[1/6 + 2/6] PCA 2D / 3D...")
    pca3 = PCA(n_components=3)
    pca3_coords = pca3.fit_transform(x)

    pca2_coords = pca3_coords[:, :2]

    save_2d_png(
        pca2_coords,
        "Healthy patch latent vectors — PCA 2D",
        out / "pca_2d.png",
    )
    save_coords(
        meta, pca2_coords, ["pc1", "pc2"],
        out / "pca_2d_coordinates.csv",
    )

    save_3d_html(
        pca3_coords,
        meta,
        "Healthy patch latent vectors — PCA 3D",
        out / "pca_3d_interactive.html",
    )
    save_coords(
        meta, pca3_coords, ["pc1", "pc2", "pc3"],
        out / "pca_3d_coordinates.csv",
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

    # ------------------------------------------------------------
    # LLE
    # ------------------------------------------------------------
    print("[3/6] LLE 2D...")
    lle2 = LocallyLinearEmbedding(
        n_neighbors=lle_neighbors,
        n_components=2,
        method="standard",
        eigen_solver="arpack",
        random_state=args.seed,
        n_jobs=-1,
    )
    lle2_coords = lle2.fit_transform(x)

    save_2d_png(
        lle2_coords,
        "Healthy patch latent vectors — LLE 2D",
        out / "lle_2d.png",
    )
    save_coords(
        meta, lle2_coords, ["lle1", "lle2"],
        out / "lle_2d_coordinates.csv",
    )

    print("[4/6] LLE 3D...")
    lle3 = LocallyLinearEmbedding(
        n_neighbors=lle_neighbors,
        n_components=3,
        method="standard",
        eigen_solver="arpack",
        random_state=args.seed,
        n_jobs=-1,
    )
    lle3_coords = lle3.fit_transform(x)

    save_3d_html(
        lle3_coords,
        meta,
        "Healthy patch latent vectors — LLE 3D",
        out / "lle_3d_interactive.html",
    )
    save_coords(
        meta, lle3_coords, ["lle1", "lle2", "lle3"],
        out / "lle_3d_coordinates.csv",
    )

    summary["lle_2d_reconstruction_error"] = float(lle2.reconstruction_error_)
    summary["lle_3d_reconstruction_error"] = float(lle3.reconstruction_error_)

    # ------------------------------------------------------------
    # t-SNE
    # ------------------------------------------------------------
    print("[5/6] t-SNE 2D... (this can take a while)")
    tsne2_coords = make_tsne(
        2, x, args.seed, perplexity, args.tsne_iterations
    )

    save_2d_png(
        tsne2_coords,
        "Healthy patch latent vectors — t-SNE 2D",
        out / "tsne_2d.png",
    )
    save_coords(
        meta, tsne2_coords, ["tsne1", "tsne2"],
        out / "tsne_2d_coordinates.csv",
    )

    print("[6/6] t-SNE 3D... (this can take a while)")
    tsne3_coords = make_tsne(
        3, x, args.seed, perplexity, args.tsne_iterations
    )

    save_3d_html(
        tsne3_coords,
        meta,
        "Healthy patch latent vectors — t-SNE 3D",
        out / "tsne_3d_interactive.html",
    )
    save_coords(
        meta, tsne3_coords, ["tsne1", "tsne2", "tsne3"],
        out / "tsne_3d_coordinates.csv",
    )

    (out / "dimension_reduction_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 92)
    print("DONE — SIX REQUESTED VISUALIZATIONS")
    print("=" * 92)
    print("PCA  2D :", out / "pca_2d.png")
    print("PCA  3D :", out / "pca_3d_interactive.html")
    print("LLE  2D :", out / "lle_2d.png")
    print("LLE  3D :", out / "lle_3d_interactive.html")
    print("t-SNE 2D:", out / "tsne_2d.png")
    print("t-SNE 3D:", out / "tsne_3d_interactive.html")
    print()
    print("All 3D HTML files are standalone and rotatable.")
    print("Coordinate CSV files and dimension_reduction_summary.json were also saved.")


if __name__ == "__main__":
    main()
