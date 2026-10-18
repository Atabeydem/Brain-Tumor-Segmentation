#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC fixed-256 patch-level cosine similarity evaluation.

Uses:
- healthy TRAIN patch embeddings as the reference bank
- threshold derived ONLY from healthy train patches
- tumor patch labels from patch_manifest_256.csv
- each patch embedding independently (NO averaging)

Threshold:
- For each healthy train patch, nearest healthy reference is searched while
  excluding ALL patches from the same source image (image-disjoint LOO).
- 5th percentile of those nearest-neighbor cosine similarities is used as
  the healthy acceptance threshold.

Evaluation:
1) Patch level:
   - 406 independent healthy-test patches (negative)
   - 3341 tumor-source test patches, whose true patch label comes from masks
     (1684 positive / 1657 mask-negative in the current audit)
2) Image level:
   - 140 healthy test images (negative)
   - 860 tumor test images (positive)
   - image is predicted tumor if ANY patch falls below threshold
     => image score = minimum patch cosine similarity

No retraining.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_eval_brisc_healthy_fixed256_resnet_20261011 as base


def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def l2_normalize(x: np.ndarray, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    acc = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "n": int(len(y_true)),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": float(acc),
        "precision_tumor": float(precision),
        "recall_sensitivity_tumor": float(recall),
        "specificity_healthy": float(specificity),
        "f1_tumor": float(f1),
    }


def max_cosine_to_reference(query_np, ref_np, device, chunk_size=256):
    q = torch.from_numpy(l2_normalize(query_np).astype(np.float32)).to(device)
    r = torch.from_numpy(l2_normalize(ref_np).astype(np.float32)).to(device)

    outs = []
    with torch.no_grad():
        for i in range(0, q.shape[0], chunk_size):
            sims = q[i:i+chunk_size] @ r.T
            outs.append(sims.max(dim=1).values.cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_image_disjoint_nn(train_emb, train_images, device, chunk_size=128):
    """
    For every healthy train patch, find nearest healthy patch from ANOTHER image.
    This avoids same-image patches making threshold artificially easy.
    """
    x_np = l2_normalize(train_emb).astype(np.float32)
    x = torch.from_numpy(x_np).to(device)

    # Integer group ids are cheaper to compare than strings on GPU.
    unique_names, group_ids_np = np.unique(train_images.astype(str), return_inverse=True)
    group_ids = torch.from_numpy(group_ids_np.astype(np.int64)).to(device)

    scores = []
    with torch.no_grad():
        for i in range(0, x.shape[0], chunk_size):
            q = x[i:i+chunk_size]
            sims = q @ x.T

            q_groups = group_ids[i:i+chunk_size].unsqueeze(1)
            same_image = q_groups == group_ids.unsqueeze(0)
            sims = sims.masked_fill(same_image, -1.0)

            scores.append(sims.max(dim=1).values.cpu().numpy())

    return np.concatenate(scores, axis=0)


class TumorPatchDataset(Dataset):
    def __init__(self, rows, patch_size=256):
        self.rows = list(rows)
        self.patch_size = patch_size

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = Path(r["source_image"])
        arr = base.load_native_gray(p)

        x0, y0 = int(r["x0"]), int(r["y0"])
        x1, y1 = int(r["x1_valid"]), int(r["y1_valid"])
        patch = arr[y0:y1, x0:x1]

        patch_pad = base.pad_patch_replicate(patch, self.patch_size)
        x = torch.from_numpy(patch_pad).unsqueeze(0).float()

        meta = {
            "source_image": str(p),
            "image": p.name,
            "patch_id": r["patch_id"],
            "row_idx": int(r["row_idx"]),
            "col_idx": int(r["col_idx"]),
            "patch_label": int(r["patch_label"]),
            "tumor_pixels": int(r["tumor_pixels"]),
            "tumor_fraction": float(r["tumor_fraction"]),
            "valid_width": int(r["valid_width"]),
            "valid_height": int(r["valid_height"]),
        }
        return x, meta


def collate_tumor(batch):
    x = torch.stack([a for a, _ in batch], 0)
    meta = [b for _, b in batch]
    return x, meta


@torch.no_grad()
def extract_tumor_embeddings(model, rows, device, batch_size=2, workers=2):
    ds = TumorPatchDataset(rows, patch_size=256)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(workers > 0),
        collate_fn=collate_tumor,
    )

    embs, meta_all = [], []
    done = 0

    model.eval()
    for x, meta in loader:
        x = x.to(device, non_blocking=True)
        _, emb = model(x, return_embedding=True)
        embs.append(emb.cpu().numpy().astype(np.float32))
        meta_all.extend(meta)
        done += len(meta)
        if done % 200 < len(meta) or done == len(ds):
            print(f"  tumor patch embeddings: {done}/{len(ds)}")

    return np.concatenate(embs, axis=0), meta_all


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--train_embeddings", required=True)
    ap.add_argument("--test_healthy_embeddings", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--healthy_acceptance", type=float, default=0.95)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    manifest = read_csv_rows(Path(args.manifest))
    tumor_rows = [r for r in manifest if r["split"] == "tumor_test"]
    healthy_test_manifest = [r for r in manifest if r["split"] == "healthy_test"]

    train_npz = np.load(args.train_embeddings, allow_pickle=True)
    healthy_test_npz = np.load(args.test_healthy_embeddings, allow_pickle=True)

    train_emb = train_npz["embeddings"].astype(np.float32)
    train_images = train_npz["source_image"].astype(str)

    healthy_test_emb = healthy_test_npz["embeddings"].astype(np.float32)
    healthy_test_images = healthy_test_npz["source_image"].astype(str)
    healthy_test_patch_ids = healthy_test_npz["patch_id"].astype(str)

    print("=" * 94)
    print("FIXED-256 PATCH-LEVEL COSINE SIMILARITY EVALUATION")
    print("=" * 94)
    print(f"Healthy train reference patches : {len(train_emb)}")
    print(f"Healthy test patches            : {len(healthy_test_emb)}")
    print(f"Tumor-source test patches       : {len(tumor_rows)}")
    print(f"Embedding dim                   : {train_emb.shape[1]}")
    print(f"Healthy acceptance target       : {args.healthy_acceptance:.3f}")
    print("Patch embedding averaging       : NONE")
    print("Threshold data                  : HEALTHY TRAIN ONLY")
    print("Train threshold LOO             : excludes all same-image patches")
    print(f"Device                          : {device}")
    print()

    print("Computing healthy-train image-disjoint NN cosine distribution...")
    train_nn = train_image_disjoint_nn(train_emb, train_images, device)

    lower_q = (1.0 - args.healthy_acceptance) * 100.0
    threshold = float(np.percentile(train_nn, lower_q))

    print(f"Threshold ({lower_q:.1f}th percentile): {threshold:.8f}")
    print(f"Train NN mean/std               : {train_nn.mean():.8f} / {train_nn.std():.8f}")
    print()

    print("Scoring independent healthy-test patches...")
    healthy_test_scores = max_cosine_to_reference(
        healthy_test_emb, train_emb, device
    )

    ckpt = torch.load(args.checkpoint, map_location=device)
    base_channels = int(ckpt.get("base_channels", 64))
    latent_dim = int(ckpt.get("latent_dim", base_channels * 8 * 2 * 2))
    model = base.NativeResNetAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("Extracting tumor-source patch embeddings...")
    tumor_emb, tumor_meta = extract_tumor_embeddings(
        model, tumor_rows, device,
        batch_size=args.batch_size,
        workers=args.workers
    )

    if tumor_emb.shape[1] != train_emb.shape[1] or train_emb.shape[1] != latent_dim:
        raise RuntimeError(
            f"Embedding dim mismatch: train={train_emb.shape[1]}, "
            f"tumor={tumor_emb.shape[1]}, checkpoint={latent_dim}"
        )

    print("Scoring tumor-source patches...")
    tumor_scores = max_cosine_to_reference(
        tumor_emb, train_emb, device
    )

    # Save tumor embeddings for PCA/LLE/t-SNE follow-up.
    np.savez_compressed(
        out / f"tumor_test_fixed256_patch_embeddings_{latent_dim}d.npz",
        embeddings=tumor_emb,
        source_image=np.array([m["source_image"] for m in tumor_meta]),
        image=np.array([m["image"] for m in tumor_meta]),
        patch_id=np.array([m["patch_id"] for m in tumor_meta]),
        row_idx=np.array([m["row_idx"] for m in tumor_meta], dtype=np.int32),
        col_idx=np.array([m["col_idx"] for m in tumor_meta], dtype=np.int32),
        patch_label=np.array([m["patch_label"] for m in tumor_meta], dtype=np.int8),
        tumor_pixels=np.array([m["tumor_pixels"] for m in tumor_meta], dtype=np.int32),
        tumor_fraction=np.array([m["tumor_fraction"] for m in tumor_meta], dtype=np.float32),
        valid_width=np.array([m["valid_width"] for m in tumor_meta], dtype=np.int32),
        valid_height=np.array([m["valid_height"] for m in tumor_meta], dtype=np.int32),
    )

    patch_rows = []

    # Healthy independent test patches: all true negative.
    for i, score in enumerate(healthy_test_scores):
        pred = int(score < threshold)
        patch_rows.append({
            "source_group": "healthy_test",
            "source_image": healthy_test_images[i],
            "patch_id": healthy_test_patch_ids[i],
            "true_label": 0,
            "pred_label": pred,
            "cosine_similarity": float(score),
            "tumor_pixels": 0,
            "tumor_fraction": 0.0,
        })

    # Tumor-source patches: true patch labels are mask-based.
    for m, score in zip(tumor_meta, tumor_scores):
        pred = int(score < threshold)
        patch_rows.append({
            "source_group": (
                "tumor_positive"
                if m["patch_label"] == 1
                else "tumor_source_mask_negative"
            ),
            "source_image": m["source_image"],
            "patch_id": m["patch_id"],
            "true_label": int(m["patch_label"]),
            "pred_label": pred,
            "cosine_similarity": float(score),
            "tumor_pixels": int(m["tumor_pixels"]),
            "tumor_fraction": float(m["tumor_fraction"]),
        })

    write_csv(out / "patch_cosine_scores.csv", patch_rows)

    patch_metrics = binary_metrics(
        [r["true_label"] for r in patch_rows],
        [r["pred_label"] for r in patch_rows],
    )

    # Subgroup summaries.
    subgroup = {}
    for group in ["healthy_test", "tumor_source_mask_negative", "tumor_positive"]:
        rr = [r for r in patch_rows if r["source_group"] == group]
        subgroup[group] = {
            "n": len(rr),
            "mean_cosine_similarity": float(np.mean([r["cosine_similarity"] for r in rr])),
            "std_cosine_similarity": float(np.std([r["cosine_similarity"] for r in rr])),
            "predicted_tumor_count": int(sum(r["pred_label"] for r in rr)),
            "predicted_tumor_rate": float(np.mean([r["pred_label"] for r in rr])),
        }

    # Image-level: ANY anomalous patch => tumor.
    by_image = defaultdict(list)
    image_true = {}

    for r in patch_rows:
        by_image[r["source_image"]].append(r)
        if r["source_group"] == "healthy_test":
            image_true[r["source_image"]] = 0
        else:
            image_true[r["source_image"]] = 1

    image_rows = []
    for source_image, rr in sorted(by_image.items()):
        min_score = float(min(r["cosine_similarity"] for r in rr))
        pred = int(any(r["pred_label"] == 1 for r in rr))
        true = int(image_true[source_image])

        image_rows.append({
            "source_image": source_image,
            "true_label": true,
            "pred_label": pred,
            "min_patch_cosine_similarity": min_score,
            "n_patches": len(rr),
            "n_predicted_tumor_patches": int(sum(r["pred_label"] for r in rr)),
            "n_true_tumor_patches": int(sum(r["true_label"] for r in rr)),
        })

    write_csv(out / "image_cosine_scores.csv", image_rows)

    image_metrics = binary_metrics(
        [r["true_label"] for r in image_rows],
        [r["pred_label"] for r in image_rows],
    )

    summary = {
        "checkpoint": str(args.checkpoint),
        "latent_dim": latent_dim,
        "reference_bank": "healthy train fixed256 patch embeddings",
        "n_reference_patches": int(len(train_emb)),
        "threshold_method": (
            "healthy-train image-disjoint nearest-neighbor cosine; "
            f"{lower_q:.1f}th percentile"
        ),
        "threshold": threshold,
        "healthy_acceptance_target": args.healthy_acceptance,
        "train_image_disjoint_nn": {
            "mean": float(train_nn.mean()),
            "std": float(train_nn.std()),
            "min": float(train_nn.min()),
            "max": float(train_nn.max()),
        },
        "patch_level": patch_metrics,
        "patch_subgroups": subgroup,
        "image_level": image_metrics,
        "image_level_rule": "tumor if ANY patch cosine falls below threshold",
        "important_note": (
            "Threshold uses healthy train only. Tumor masks are used only for "
            "evaluation labels. Patch embeddings are never averaged."
        ),
    }

    (out / "cosine_patch_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== PATCH LEVEL ===")
    print(json.dumps(patch_metrics, indent=2))
    print()
    print("=== PATCH SUBGROUPS ===")
    print(json.dumps(subgroup, indent=2))
    print()
    print("=== IMAGE LEVEL (ANY ANOMALOUS PATCH => TUMOR) ===")
    print(json.dumps(image_metrics, indent=2))
    print()
    print("Saved outputs:", out)


if __name__ == "__main__":
    main()
