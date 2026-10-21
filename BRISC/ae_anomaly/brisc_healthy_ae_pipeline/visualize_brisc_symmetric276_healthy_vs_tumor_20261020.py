#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Healthy vs tumor-positive latent overlays for the *frozen* Symmetric-276 AE.

No new AE training, no image resizing, and no image-wise embedding averaging.
Healthy features must come from the matching Symmetric-276 checkpoint run.
Tumor-positive samples come exclusively from segmentation TRAIN masks (not test).
PCA, LLE, and t-SNE each fit on healthy and tumor samples *jointly*.

Run from brisc_healthy_ae_pipeline alongside
  train_eval_brisc_healthy_symmetric276_resnet_20261019.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE

import train_eval_brisc_healthy_symmetric276_resnet_20261019 as ae

TUMOR_SPLIT = "tumor_train_seg_only_for_analysis"


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--healthy_npz", type=Path, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--max_per_group", type=int, default=1200)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--neighbors", type=int, default=20)
    p.add_argument("--perplexity", type=float, default=30.0)
    p.add_argument("--full_only", action="store_true", help="Use only 256x256 fully valid patches in both groups")
    return p.parse_args()


def choose(indices, n, seed):
    indices = np.asarray(indices)
    if len(indices) <= n:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=n, replace=False))


def make_healthy(args):
    with np.load(args.healthy_npz, allow_pickle=False) as f:
        required = ("embeddings", "image", "patch_id", "valid_width", "valid_height",
                    "original_width", "original_height", "row_idx", "col_idx")
        missing = [x for x in required if x not in f.files]
        if missing:
            raise RuntimeError(f"Healthy NPZ missing keys: {missing}")
        x = f["embeddings"].astype(np.float32)
        all_idx = np.arange(len(x))
        if args.full_only:
            all_idx = all_idx[(f["valid_width"] == 256) & (f["valid_height"] == 256)]
        idx = choose(all_idx, args.max_per_group, args.seed)
        data = x[idx].copy()
        meta = pd.DataFrame({
            "group": "Healthy train",
            "image": f["image"][idx].astype(str),
            "patch_id": f["patch_id"][idx].astype(str),
            "row_idx": f["row_idx"][idx], "col_idx": f["col_idx"][idx],
            "valid_width": f["valid_width"][idx], "valid_height": f["valid_height"][idx],
            "original_width": f["original_width"][idx], "original_height": f["original_height"][idx],
            "tumor_fraction": np.zeros(len(idx), dtype=np.float32),
        })
    return data, meta


def make_tumor(args, device):
    rows = ae.read_manifest(args.manifest)
    available = sorted(set(r["split"] for r in rows))
    positive = [r for r in rows if r["split"] == TUMOR_SPLIT and int(r["patch_label"]) == 1]
    if not positive:
        raise RuntimeError(f"No tumor-positive segmentation TRAIN patches in manifest. "
                           f"Expected split={TUMOR_SPLIT!r}. Available splits: {available}. "
                           "Recreate audit manifest with --include_seg_train.")
    if args.full_only:
        positive = [r for r in positive if int(r["valid_width"]) == 256 and int(r["valid_height"]) == 256]
    idx = choose(np.arange(len(positive)), args.max_per_group, args.seed + 1)
    selected = [positive[i] for i in idx]
    print(f"Positive segtrain patches available: {len(positive)}; selected: {len(selected)}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("context_padding") not in ("symmetric_10_each_side", None):
        raise RuntimeError(f"Unexpected checkpoint context: {checkpoint.get('context_padding')}")
    model = ae.NativeResNetAutoencoder(base_channels=int(checkpoint.get("base_channels", 64)))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()

    feats = []
    # Image cache avoids reading repeated patches of the same MRI from disk.
    cached_path, cached_img = None, None
    for start in range(0, len(selected), args.batch_size):
        batch_rows = selected[start:start + args.batch_size]
        prepared = []
        for r in batch_rows:
            source = r["source_image"]
            if source != cached_path:
                cached_img = ae.load_native_gray(Path(source))
                cached_path = source
            x0, y0 = int(r["x0"]), int(r["y0"])
            x1, y1 = int(r["x1_valid"]), int(r["y1_valid"])
            patch = cached_img[y0:y1, x0:x1]
            if patch.shape != (int(r["valid_height"]), int(r["valid_width"])):
                raise ValueError(f"Manifest geometry mismatch: {source}, {r['patch_id']}")
            p256 = ae.pad_patch_symmetric(patch, patch_size=256)
            prepared.append(ae.pad_context_276(p256))
        tensor = torch.from_numpy(np.stack(prepared)[:, None]).float().to(device)
        with torch.inference_mode():
            internal = ae.torch_symmetric_pad2(tensor)
            z = model.encode_features(internal)
            emb = model.latent_embedding(z)
        feats.append(emb.cpu().numpy().astype(np.float32))
        done = min(start + args.batch_size, len(selected))
        if done % 200 < args.batch_size or done == len(selected):
            print(f"  Extracted tumor-positive features: {done}/{len(selected)}")

    data = np.concatenate(feats, axis=0)
    meta = pd.DataFrame([{
        "group": "Tumor-positive segtrain", "image": Path(r["source_image"]).name,
        "patch_id": r["patch_id"], "row_idx": int(r["row_idx"]), "col_idx": int(r["col_idx"]),
        "valid_width": int(r["valid_width"]), "valid_height": int(r["valid_height"]),
        "original_width": int(r["original_width"]), "original_height": int(r["original_height"]),
        "tumor_fraction": float(r["tumor_fraction"]),
    } for r in selected])
    return data, meta


def draw_2d(y, groups, name, out):
    fig, ax = plt.subplots(figsize=(9, 7))
    for label, color, alpha in (("Healthy train", "#2563eb", 0.45),
                                ("Tumor-positive segtrain", "#dc2626", 0.48)):
        mask = (groups == label)
        ax.scatter(y[mask, 0], y[mask, 1], c=color, alpha=alpha, s=9, label=f"{label} (n={mask.sum()})", rasterized=True)
    ax.set(xlabel=f"{name} 1", ylabel=f"{name} 2", title=f"{name} — Healthy / tumor-positive (joint fit)")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(out / f"{name.lower()}_2d.png", dpi=180)
    plt.close(fig)


def draw_3d(y, groups, name, out, meta):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    for label, color in (("Healthy train", "#2563eb"), ("Tumor-positive segtrain", "#dc2626")):
        mask = groups == label
        ax.scatter(y[mask, 0], y[mask, 1], y[mask, 2], c=color, alpha=0.30, s=7, label=f"{label} (n={mask.sum()})", rasterized=True)
    ax.set(xlabel=f"{name} 1", ylabel=f"{name} 2", zlabel=f"{name} 3")
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(out / f"{name.lower()}_3d.png", dpi=180)
    plt.close(fig)

    fig = go.Figure()
    for label, color in (("Healthy train", "#2563eb"), ("Tumor-positive segtrain", "#dc2626")):
        mask = groups == label
        subset = meta.loc[mask]
        hover = [f"{r.image}<br>patch={r.patch_id}<br>valid={r.valid_width}x{r.valid_height}<br>tumor_fraction={r.tumor_fraction:.4f}"
                 for r in subset.itertuples(index=False)]
        fig.add_trace(go.Scatter3d(x=y[mask, 0], y=y[mask, 1], z=y[mask, 2],
                                   mode="markers", name=label, text=hover, hoverinfo="text",
                                   marker=dict(size=2.8, color=color, opacity=0.62)))
    fig.update_layout(title=f"{name} 3D — healthy vs tumor-positive (joint fit)",
                      scene=dict(xaxis_title="Dimension 1", yaxis_title="Dimension 2", zaxis_title="Dimension 3"),
                      legend=dict(orientation="h", y=1.06), margin=dict(l=0, r=0, t=50, b=0))
    fig.write_html(out / f"{name.lower()}_3d_interactive.html", include_plotlyjs=True)


def main():
    args = arguments()
    if args.max_per_group < 5 or args.batch_size < 1:
        raise ValueError("max_per_group must be >=5 and batch_size >=1")
    for p in (args.manifest, args.checkpoint, args.healthy_npz):
        if not p.is_file():
            raise FileNotFoundError(p)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Model is FROZEN; tumor labels are used only for selection and plotting.")
    print("Data source: healthy TRAIN vs tumor-positive SEGMENTATION TRAIN; no test-set tuning.")
    print(f"Device: {device}, full_only={args.full_only}")

    healthy_x, healthy_meta = make_healthy(args)
    tumor_x, tumor_meta = make_tumor(args, device)
    if healthy_x.shape[1] != tumor_x.shape[1] or healthy_x.shape[1] != 2048:
        raise RuntimeError(f"Embedding dimensions mismatch: {healthy_x.shape}, {tumor_x.shape}")
    x = np.concatenate([healthy_x, tumor_x], axis=0)
    meta = pd.concat([healthy_meta, tumor_meta], ignore_index=True)
    groups = meta["group"].to_numpy()
    if not np.isfinite(x).all():
        raise RuntimeError("Nonfinite embedding values")
    n = x.shape[0]
    print(f"Joint projection: {healthy_x.shape[0]} healthy + {tumor_x.shape[0]} tumor-positive; {n} x {x.shape[1]}")
    meta.to_csv(args.output_dir / "sample_metadata.csv", index=False)

    # PCA directly on RAW 2048-D features; no feature-standardization changes.
    pca = PCA(n_components=3, svd_solver="randomized", random_state=args.seed)
    ypca = pca.fit_transform(x)
    draw_2d(ypca, groups, "PCA", args.output_dir)
    draw_3d(ypca, groups, "PCA", args.output_dir, meta)
    for dims in (2, 3):
        y = ypca[:, :dims]
        temp = meta.copy()
        for i in range(dims):
            temp[f"component_{i+1}"] = y[:, i]
        temp.to_csv(args.output_dir / f"pca_{dims}d_coordinates.csv", index=False)

    # Same joint PCA50 representation for LLE/t-SNE, for tractable runtime.
    pre_dim = min(50, n - 2, x.shape[1])
    xp = PCA(n_components=pre_dim, svd_solver="randomized", random_state=args.seed).fit_transform(x)
    neighbors = min(args.neighbors, n - 2)
    perplexity = min(args.perplexity, max(2.0, (n - 1) / 3))
    config = dict(healthy_count=len(healthy_x), tumor_count=len(tumor_x), latent_dim=2048,
                  tumor_split=TUMOR_SPLIT, seed=args.seed, full_only=args.full_only,
                  pca_first3_variance=pca.explained_variance_ratio_.tolist(),
                  nonlinear_pca_pre_dims=pre_dim, lle_neighbors=neighbors,
                  tsne_perplexity=perplexity, pretrained_encoder_frozen=True,
                  fit_jointly=True, supervised_classifier_used=False)

    for method in ("LLE", "TSNE"):
        for dim in (2, 3):
            print(f"Computing {method} {dim}D for {n} samples...")
            if method == "LLE":
                algorithm = LocallyLinearEmbedding(n_neighbors=neighbors, n_components=dim,
                                                   method="standard", eigen_solver="arpack", random_state=args.seed)
                y = algorithm.fit_transform(xp)
            else:
                algorithm = TSNE(n_components=dim, perplexity=perplexity, learning_rate="auto",
                                 init="pca", random_state=args.seed, max_iter=1000)
                y = algorithm.fit_transform(xp)
            label = "tSNE" if method == "TSNE" else "LLE"
            result = meta.copy()
            for i in range(dim):
                result[f"component_{i+1}"] = y[:, i]
            result.to_csv(args.output_dir / f"{label.lower()}_{dim}d_coordinates.csv", index=False)
            if dim == 2:
                draw_2d(y, groups, label, args.output_dir)
            else:
                draw_3d(y, groups, label, args.output_dir, meta)

    (args.output_dir / "projection_summary.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print("FINISHED. Saved all six plots, 3 interactive HTML files, and coordinate CSVs:")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()
