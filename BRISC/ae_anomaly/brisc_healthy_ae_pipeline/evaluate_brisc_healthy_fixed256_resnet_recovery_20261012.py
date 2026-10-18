#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch

import train_eval_brisc_healthy_fixed256_resnet_20261011 as base

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def safe_image_metrics(orig, recon):
    mse = float(np.mean((orig - recon) ** 2))
    mae = float(np.mean(np.abs(orig - recon)))
    psnr = base.psnr_from_mse(mse)

    ssim = None
    if HAS_SKIMAGE:
        min_side = int(min(orig.shape[-2], orig.shape[-1]))
        if min_side >= 3:
            win_size = min(7, min_side)
            if win_size % 2 == 0:
                win_size -= 1
            if win_size >= 3:
                ssim = float(
                    ssim_fn(
                        orig,
                        recon,
                        data_range=1.0,
                        win_size=win_size,
                    )
                )

    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--save_examples", type=int, default=10)
    args = ap.parse_args()

    base.image_metrics = safe_image_metrics

    manifest_path = Path(args.manifest)
    ckpt_path = Path(args.checkpoint)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = base.read_manifest(manifest_path)
    train_rows = base.manifest_rows_for_split(rows, "healthy_train")
    test_rows = base.manifest_rows_for_split(rows, "healthy_test")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    base_channels = int(ckpt.get("base_channels", 64))
    latent_dim = int(ckpt.get("latent_dim", base_channels * 8 * 2 * 2))

    model = base.NativeResNetAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    print("=" * 90)
    print("FIXED-256 RESNET EVALUATION RECOVERY")
    print("=" * 90)
    print(f"Checkpoint epoch      : {ckpt.get('epoch')}")
    print(f"Checkpoint train MSE  : {ckpt.get('train_mse')}")
    print(f"Latent embedding      : {latent_dim}-D")
    print(f"Train patches         : {len(train_rows)}")
    print(f"Test patches          : {len(test_rows)}")
    print(f"Device                : {device}")
    print("Retraining             : NO")
    print("SSIM fix               : adaptive valid win_size")
    print()

    print("Evaluating healthy TRAIN...")
    train_patch_rows, train_image_rows, d1 = base.evaluate_fixed256(
        model, train_rows, device, out, "train_healthy_fixed256",
        patch_size=args.patch_size, save_examples=0
    )

    print()
    print("Evaluating healthy TEST...")
    test_patch_rows, test_image_rows, d2 = base.evaluate_fixed256(
        model, test_rows, device, out, "test_healthy_fixed256",
        patch_size=args.patch_size, save_examples=args.save_examples
    )

    if d1 != latent_dim or d2 != latent_dim:
        raise RuntimeError(f"Latent dim mismatch: train={d1}, test={d2}, expected={latent_dim}")

    final = {
        "retrained": False,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_train_mse": ckpt.get("train_mse"),
        "latent_embedding_dim": latent_dim,
        "spatial_resize": False,
        "patch_strategy": "fixed 256x256, stride 256",
        "patch_embeddings_kept_separate": True,
        "image_embedding_average": False,
        "train_patch_metrics": base.summarize(train_patch_rows),
        "train_whole_image_metrics": base.summarize(train_image_rows),
        "test_patch_metrics": base.summarize(test_patch_rows),
        "test_whole_image_metrics": base.summarize(test_image_rows),
        "previous_native_2x2_best_psnr_db": 64.509215,
    }

    new_psnr = final["test_whole_image_metrics"]["mean_psnr_db"]
    final["delta_vs_previous_native_2x2_db"] = float(new_psnr - 64.509215)

    (out / "final_summary_recovered.json").write_text(
        json.dumps(final, indent=2), encoding="utf-8"
    )

    print()
    print("=== FINAL HEALTHY TEST - WHOLE NATIVE IMAGE ===")
    print(json.dumps(final["test_whole_image_metrics"], indent=2))
    print()
    print("=== FINAL HEALTHY TEST - PATCH LEVEL ===")
    print(json.dumps(final["test_patch_metrics"], indent=2))
    print()
    print("=== COMPARISON WITH PREVIOUS BEST ===")
    print("Previous native-2x2 best : 64.5092 dB")
    print(f"Fixed-256 current        : {new_psnr:.4f} dB")
    print(f"Delta                    : {new_psnr - 64.509215:+.4f} dB")
    print()
    print("Saved outputs:", out)


if __name__ == "__main__":
    main()
