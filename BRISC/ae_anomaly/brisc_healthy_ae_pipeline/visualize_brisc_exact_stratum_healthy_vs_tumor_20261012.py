#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC exact-stratum latent comparison:
Healthy vs Tumor-source mask-negative vs Tumor-positive

Purpose
-------
The previous controls showed that latent space strongly depends on:
- source resolution
- patch row/column position
- valid patch geometry / padding

This script therefore compares classes only INSIDE the same exact stratum:
(original_width, original_height, row_idx, col_idx, valid_width, valid_height)

For each sufficiently populated stratum it produces:
- PCA 2D plot
- optional t-SNE 2D plot
- count / centroid-distance summary

No retraining.
No resizing.
No embedding averaging.
Only fixed-256 patch embeddings are used.

Recommended first run:
- full 256x256 patches only
- top exact strata with enough samples
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def exact_key(rec):
    return (
        int(rec["original_width"]),
        int(rec["original_height"]),
        int(rec["row_idx"]),
        int(rec["col_idx"]),
        int(rec["valid_width"]),
        int(rec["valid_height"]),
    )


def key_name(key):
    ow, oh, r, c, vw, vh = key
    return f"{ow}x{oh}_r{r}c{c}_valid{vw}x{vh}"


def load_healthy(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)

    rows = []
    for i in range(len(emb)):
        rows.append({
            "source_image": str(d["source_image"][i]),
            "patch_id": str(d["patch_id"][i]),
            "row_idx": int(d["row_idx"][i]),
            "col_idx": int(d["col_idx"][i]),
            "valid_width": int(d["valid_width"][i]),
            "valid_height": int(d["valid_height"][i]),
            "original_width": int(d["original_width"][i]),
            "original_height": int(d["original_height"][i]),
            "class_name": "healthy_train",
        })
    return emb, rows


def load_tumor(npz_path: Path, manifest_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)

    manifest = read_csv_rows(manifest_path)
    index = {
        (r["source_image"], r["patch_id"]): r
        for r in manifest
        if r["split"] == "tumor_test"
    }

    rows = []
    for i in range(len(emb)):
        source_image = str(d["source_image"][i])
        patch_id = str(d["patch_id"][i])
        m = index.get((source_image, patch_id))
        if m is None:
            raise RuntimeError(
                f"Could not join tumor embedding to manifest: "
                f"{source_image} / {patch_id}"
            )

        label = int(m["patch_label"])
        rows.append({
            "source_image": source_image,
            "patch_id": patch_id,
            "row_idx": int(d["row_idx"][i]),
            "col_idx": int(d["col_idx"][i]),
            "valid_width": int(d["valid_width"][i]),
            "valid_height": int(d["valid_height"][i]),
            "original_width": int(m["original_width"]),
            "original_height": int(m["original_height"]),
            "tumor_pixels": int(m["tumor_pixels"]),
            "tumor_fraction": float(m["tumor_fraction"]),
            "class_name": (
                "tumor_positive"
                if label == 1
                else "tumor_mask_negative"
            ),
        })

    return emb, rows


def group_indices(records, full_only=True):
    groups = defaultdict(list)
    for i, r in enumerate(records):
        if full_only and (
            int(r["valid_width"]) != 256
            or int(r["valid_height"]) != 256
        ):
            continue
        groups[exact_key(r)].append(i)
    return groups


def centroid_cosine(a, b):
    a = np.mean(l2_normalize(a), axis=0)
    b = np.mean(l2_normalize(b), axis=0)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def pca_plot(x, labels, title, out_png):
    pca = PCA(n_components=2)
    z = pca.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))

    order = [
        "healthy_train",
        "tumor_mask_negative",
        "tumor_positive",
    ]

    for cls in order:
        mask = labels == cls
        if mask.any():
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=14,
                alpha=0.62,
                label=f"{cls} (n={int(mask.sum())})",
            )

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return z, pca


def tsne_plot(x, labels, seed, title, out_png):
    n = len(x)
    perplexity = min(30.0, max(5.0, (n - 1) / 3.0))

    kwargs = dict(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        random_state=seed,
        method="barnes_hut",
        angle=0.5,
    )

    try:
        model = TSNE(max_iter=1500, **kwargs)
    except TypeError:
        model = TSNE(n_iter=1500, **kwargs)

    z = model.fit_transform(x)

    fig, ax = plt.subplots(figsize=(10, 8))

    for cls in [
        "healthy_train",
        "tumor_mask_negative",
        "tumor_positive",
    ]:
        mask = labels == cls
        if mask.any():
            ax.scatter(
                z[mask, 0],
                z[mask, 1],
                s=14,
                alpha=0.62,
                label=f"{cls} (n={int(mask.sum())})",
            )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)

    return z, perplexity


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)
    ap.add_argument("--healthy_embeddings", required=True)
    ap.add_argument("--tumor_embeddings", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--min_healthy", type=int, default=40)
    ap.add_argument("--min_tumor_negative", type=int, default=10)
    ap.add_argument("--min_tumor_positive", type=int, default=10)
    ap.add_argument("--max_groups", type=int, default=8)

    ap.add_argument(
        "--include_padded",
        action="store_true",
        help="By default only valid 256x256 patches are used.",
    )
    ap.add_argument(
        "--run_tsne",
        action="store_true",
        help="Also run t-SNE 2D inside each selected exact stratum.",
    )
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    h_emb, h_rows = load_healthy(Path(args.healthy_embeddings))
    t_emb, t_rows = load_tumor(
        Path(args.tumor_embeddings),
        Path(args.manifest),
    )

    full_only = not args.include_padded

    h_groups = group_indices(h_rows, full_only=full_only)
    t_groups = group_indices(t_rows, full_only=full_only)

    candidate_keys = sorted(
        set(h_groups.keys()) | set(t_groups.keys())
    )

    eligible = []

    for key in candidate_keys:
        h_idx = h_groups.get(key, [])
        t_idx = t_groups.get(key, [])

        tneg = [
            i for i in t_idx
            if t_rows[i]["class_name"] == "tumor_mask_negative"
        ]
        tpos = [
            i for i in t_idx
            if t_rows[i]["class_name"] == "tumor_positive"
        ]

        if (
            len(h_idx) >= args.min_healthy
            and len(tneg) >= args.min_tumor_negative
            and len(tpos) >= args.min_tumor_positive
        ):
            eligible.append(
                (key, h_idx, tneg, tpos)
            )

    eligible.sort(
        key=lambda x: (
            len(x[1]) + len(x[2]) + len(x[3])
        ),
        reverse=True,
    )
    eligible = eligible[:args.max_groups]

    print("=" * 98)
    print("BRISC EXACT-STRATUM LATENT COMPARISON")
    print("=" * 98)
    print(f"Healthy embeddings       : {len(h_emb)}")
    print(f"Tumor-source embeddings  : {len(t_emb)}")
    print(f"Embedding dim            : {h_emb.shape[1]}")
    print(f"Full 256x256 only        : {full_only}")
    print(f"Eligible exact strata    : {len(eligible)}")
    print(f"t-SNE requested          : {args.run_tsne}")
    print()

    if not eligible:
        raise RuntimeError(
            "No exact strata met the minimum count criteria. "
            "Try lowering --min_healthy / --min_tumor_negative / --min_tumor_positive."
        )

    summary_rows = []

    for rank, (key, h_idx, tneg_idx, tpos_idx) in enumerate(eligible, 1):
        name = key_name(key)

        h_x = h_emb[np.asarray(h_idx)]
        n_x = t_emb[np.asarray(tneg_idx)]
        p_x = t_emb[np.asarray(tpos_idx)]

        x = np.concatenate([h_x, n_x, p_x], axis=0)
        labels = np.asarray(
            ["healthy_train"] * len(h_x)
            + ["tumor_mask_negative"] * len(n_x)
            + ["tumor_positive"] * len(p_x)
        )

        print(
            f"[{rank}/{len(eligible)}] {name} | "
            f"healthy={len(h_x)} neg={len(n_x)} pos={len(p_x)}"
        )

        pca_coords, pca = pca_plot(
            x,
            labels,
            f"{name} — Healthy vs Tumor-source patches (PCA 2D)",
            out / f"{name}_pca2d.png",
        )

        coord_df = pd.DataFrame({
            "class_name": labels,
            "pc1": pca_coords[:, 0],
            "pc2": pca_coords[:, 1],
        })
        coord_df.to_csv(
            out / f"{name}_pca2d_coordinates.csv",
            index=False,
        )

        tsne_perplexity = None
        if args.run_tsne:
            tsne_coords, tsne_perplexity = tsne_plot(
                x,
                labels,
                args.seed,
                f"{name} — Healthy vs Tumor-source patches (t-SNE 2D)",
                out / f"{name}_tsne2d.png",
            )

            pd.DataFrame({
                "class_name": labels,
                "tsne1": tsne_coords[:, 0],
                "tsne2": tsne_coords[:, 1],
            }).to_csv(
                out / f"{name}_tsne2d_coordinates.csv",
                index=False,
            )

        # Original 2048-D centroid cosine relationships.
        cos_h_neg = centroid_cosine(h_x, n_x)
        cos_h_pos = centroid_cosine(h_x, p_x)
        cos_neg_pos = centroid_cosine(n_x, p_x)

        # PCA centroid distances are descriptive only.
        pca_centroids = {}
        for cls in [
            "healthy_train",
            "tumor_mask_negative",
            "tumor_positive",
        ]:
            z = pca_coords[labels == cls]
            pca_centroids[cls] = np.mean(z, axis=0)

        dist_h_neg = float(
            np.linalg.norm(
                pca_centroids["healthy_train"]
                - pca_centroids["tumor_mask_negative"]
            )
        )
        dist_h_pos = float(
            np.linalg.norm(
                pca_centroids["healthy_train"]
                - pca_centroids["tumor_positive"]
            )
        )
        dist_neg_pos = float(
            np.linalg.norm(
                pca_centroids["tumor_mask_negative"]
                - pca_centroids["tumor_positive"]
            )
        )

        summary_rows.append({
            "stratum": name,
            "original_width": key[0],
            "original_height": key[1],
            "row_idx": key[2],
            "col_idx": key[3],
            "valid_width": key[4],
            "valid_height": key[5],
            "n_healthy": len(h_x),
            "n_tumor_mask_negative": len(n_x),
            "n_tumor_positive": len(p_x),
            "pca_var_pc1": float(pca.explained_variance_ratio_[0]),
            "pca_var_pc2": float(pca.explained_variance_ratio_[1]),
            "pca_var_pc1_pc2_total": float(
                pca.explained_variance_ratio_[:2].sum()
            ),
            "centroid_cosine_healthy_vs_mask_negative": cos_h_neg,
            "centroid_cosine_healthy_vs_tumor_positive": cos_h_pos,
            "centroid_cosine_mask_negative_vs_tumor_positive": cos_neg_pos,
            "pca_centroid_distance_healthy_vs_mask_negative": dist_h_neg,
            "pca_centroid_distance_healthy_vs_tumor_positive": dist_h_pos,
            "pca_centroid_distance_mask_negative_vs_tumor_positive": dist_neg_pos,
            "tsne_perplexity": tsne_perplexity,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(
        out / "exact_stratum_latent_summary.csv",
        index=False,
    )

    compact = {
        "full_256_only": full_only,
        "n_selected_strata": len(summary_rows),
        "selected_strata": summary_rows,
        "interpretation_note": (
            "Plots are descriptive controls. A visual separation must not be "
            "interpreted as classification performance without an independent "
            "quantitative test."
        ),
    }

    (out / "exact_stratum_latent_summary.json").write_text(
        json.dumps(compact, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 98)
    print("SUMMARY")
    print("=" * 98)

    cols = [
        "stratum",
        "n_healthy",
        "n_tumor_mask_negative",
        "n_tumor_positive",
        "centroid_cosine_healthy_vs_mask_negative",
        "centroid_cosine_healthy_vs_tumor_positive",
        "centroid_cosine_mask_negative_vs_tumor_positive",
    ]
    print(summary_df[cols].to_string(index=False))

    print()
    print("Saved to:", out)
    print("Primary summary:")
    print(" ", out / "exact_stratum_latent_summary.csv")
    print(" ", out / "exact_stratum_latent_summary.json")


if __name__ == "__main__":
    main()
