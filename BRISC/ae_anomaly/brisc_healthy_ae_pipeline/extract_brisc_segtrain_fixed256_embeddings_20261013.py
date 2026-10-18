#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extract BRISC segmentation-train fixed-256 patch embeddings from the already trained
healthy-only ResNet checkpoint.

Prerequisite
------------
Re-run the fixed256 audit with --include_seg_train so the manifest contains split:
    tumor_train_seg_only_for_analysis

No retraining. No resize. Patch embeddings are kept separate.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

import train_eval_brisc_healthy_fixed256_resnet_20261011 as base


class SegPatchDataset(Dataset):
    def __init__(self, rows, patch_size=256):
        self.rows = list(rows)
        self.patch_size = int(patch_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        arr = base.load_native_gray(Path(r["source_image"]))

        x0, y0 = int(r["x0"]), int(r["y0"])
        x1, y1 = int(r["x1_valid"]), int(r["y1_valid"])
        patch = arr[y0:y1, x0:x1]
        patch = base.pad_patch_replicate(patch, self.patch_size)

        x = torch.from_numpy(patch).unsqueeze(0).float()

        meta = {
            "source_image": str(r["source_image"]),
            "patch_id": str(r["patch_id"]),
            "row_idx": int(r["row_idx"]),
            "col_idx": int(r["col_idx"]),
            "valid_width": int(r["valid_width"]),
            "valid_height": int(r["valid_height"]),
            "original_width": int(r["original_width"]),
            "original_height": int(r["original_height"]),
            "patch_label": int(r["patch_label"]),
            "tumor_pixels": int(r["tumor_pixels"]),
            "tumor_fraction": float(r["tumor_fraction"]),
        }
        return x, meta


def collate(batch):
    x = torch.stack([a for a, _ in batch], dim=0)
    meta = [b for _, b in batch]
    return x, meta


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    rows = base.read_manifest(Path(args.manifest))
    rows = [r for r in rows if r["split"] == "tumor_train_seg_only_for_analysis"]
    if not rows:
        raise RuntimeError(
            "No segmentation-train rows found. Re-run prepare_brisc_fixed256_patch_audit_20261011.py "
            "with --include_seg_train first."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    base_channels = int(ckpt.get("base_channels", 64))
    latent_dim = int(ckpt.get("latent_dim", base_channels * 8 * 2 * 2))

    model = base.NativeResNetAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ds = SegPatchDataset(rows)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
        collate_fn=collate,
    )

    all_emb, all_meta = [], []
    done = 0

    print("=" * 92)
    print("BRISC SEGMENTATION-TRAIN FIXED256 EMBEDDING EXTRACTION")
    print("=" * 92)
    print(f"Patches      : {len(ds)}")
    print(f"Latent dim   : {latent_dim}")
    print(f"Device       : {device}")
    print("Retraining   : NO")
    print()

    for x, meta in loader:
        x = x.to(device, non_blocking=True)
        _, emb = model(x, return_embedding=True)
        all_emb.append(emb.cpu().numpy().astype(np.float32))
        all_meta.extend(meta)
        done += len(meta)
        if done % 500 < len(meta) or done == len(ds):
            print(f"  {done}/{len(ds)}")

    emb = np.concatenate(all_emb, axis=0)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        out,
        embeddings=emb,
        source_image=np.array([m["source_image"] for m in all_meta]),
        patch_id=np.array([m["patch_id"] for m in all_meta]),
        row_idx=np.array([m["row_idx"] for m in all_meta], dtype=np.int32),
        col_idx=np.array([m["col_idx"] for m in all_meta], dtype=np.int32),
        valid_width=np.array([m["valid_width"] for m in all_meta], dtype=np.int32),
        valid_height=np.array([m["valid_height"] for m in all_meta], dtype=np.int32),
        original_width=np.array([m["original_width"] for m in all_meta], dtype=np.int32),
        original_height=np.array([m["original_height"] for m in all_meta], dtype=np.int32),
        patch_label=np.array([m["patch_label"] for m in all_meta], dtype=np.int8),
        tumor_pixels=np.array([m["tumor_pixels"] for m in all_meta], dtype=np.int32),
        tumor_fraction=np.array([m["tumor_fraction"] for m in all_meta], dtype=np.float32),
    )

    print()
    print("Saved:", out)
    print("Shape:", emb.shape)
    print("Tumor-positive patches:", int(sum(m["patch_label"] for m in all_meta)))
    print("Mask-negative patches :", int(len(all_meta) - sum(m["patch_label"] for m in all_meta)))


if __name__ == "__main__":
    main()
