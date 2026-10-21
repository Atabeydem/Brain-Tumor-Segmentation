#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC healthy-only symmetric-context fixed-256 ResNet autoencoder.

A controlled continuation of train_eval_brisc_healthy_fixed256_resnet_20261011.py:
- Original tiles: 256x256, stride 256. NO spatial resizing.
- Incomplete source-image edge tiles are symmetrically extended to 256x256.
- Every 256x256 tile receives 10px symmetric context on EACH side (276x276).
- The unchanged 3-down/3-up ResNet architecture needs a spatial multiple of 8.
  It maps 276 -> 272, so a model-I/O adapter adds 2px symmetric border
  (276 -> 280) and crops 2px on output (280 -> 276).
- Reconstruction and loss are computed on the ORIGINAL VALID source pixels
  only; the 10px context and edge-padding are excluded from the loss.
- Cropped central outputs are reassembled into original native image geometry.
- No U-Net long skip, BatchNorm, or alteration to learnable model layers.
- Healthy-only training and separate 2048-D patch embeddings (base64).

IMPORTANT: model's internal tensor is 280x280, not 276x276, because 276 is not
  divisible by 8. The pipeline's input/output are 276x276; this adaptation is
  purely deterministic, parameter-free and explicitly documented.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False


def load_native_gray(path: Path):
    # Grayscale conversion only; NO spatial resizing.
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def read_manifest(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def manifest_rows_for_split(rows, split_name):
    out = [r for r in rows if r["split"] == split_name]
    if not out:
        raise RuntimeError(f"No rows found for split={split_name!r} in manifest.")
    return out


def pad_patch_symmetric(patch: np.ndarray, patch_size=256):
    """Symmetric-pad only RIGHT/BOTTOM where source image has no valid pixels."""
    h, w = patch.shape
    if h <= 0 or w <= 0:
        raise RuntimeError(f"Empty patch encountered: shape={patch.shape}")
    if h > patch_size or w > patch_size:
        raise RuntimeError(f"Patch exceeds fixed size: shape={patch.shape}")
    return np.pad(
        patch, ((0, patch_size - h), (0, patch_size - w)),
        mode="symmetric",
    ).astype(np.float32, copy=False)


CONTEXT_PIXELS = 10
COMPAT_PIXELS = 2


def pad_context_276(patch256: np.ndarray):
    if patch256.shape != (256, 256):
        raise ValueError(f"Expected (256, 256), got {patch256.shape}")
    return np.pad(
        patch256,
        ((CONTEXT_PIXELS, CONTEXT_PIXELS),
         (CONTEXT_PIXELS, CONTEXT_PIXELS)),
        mode="symmetric",
    ).astype(np.float32, copy=False)


def valid_context_mask(valid_h: int, valid_w: int):
    """Mask only genuine pixels; zero-valued entries are NOT image padding."""
    m = np.zeros((276, 276), dtype=np.float32)
    m[CONTEXT_PIXELS:CONTEXT_PIXELS + valid_h,
      CONTEXT_PIXELS:CONTEXT_PIXELS + valid_w] = 1.0
    return m


def torch_symmetric_pad2(x: torch.Tensor):
    """NumPy-like symmetric reflection (includes border pixel); differentiable."""
    if x.ndim != 4 or x.shape[-2:] != (276, 276):
        raise ValueError(f"Expected NCHW 276x276 tensor, got {tuple(x.shape)}")
    p = COMPAT_PIXELS
    x = torch.cat((x[..., :p].flip(-1), x, x[..., -p:].flip(-1)), dim=-1)
    x = torch.cat((x[..., :p, :].flip(-2), x, x[..., -p:, :].flip(-2)), dim=-2)
    assert x.shape[-2:] == (280, 280)
    return x


def forward_276(model, x276, return_embedding=False):
    """Unchanged model on 280x280 internal tensor; adapter returns 276x276."""
    x280 = torch_symmetric_pad2(x276)
    result = model(x280, return_embedding=return_embedding)
    if return_embedding:
        y280, embedding = result
    else:
        y280 = result
    if y280.shape[-2:] != (280, 280):
        raise RuntimeError(f"Unchanged ResNet did not return 280x280: {y280.shape}")
    p = COMPAT_PIXELS
    y276 = y280[..., p:-p, p:-p]
    if y276.shape != x276.shape:
        raise RuntimeError("Internal pad/unpad mismatch")
    if return_embedding:
        return y276, embedding
    return y276


def valid_mask(valid_h, valid_w, patch_size=256):
    m = np.zeros((patch_size, patch_size), dtype=np.float32)
    m[:valid_h, :valid_w] = 1.0
    return m


class Fixed256PatchDataset(Dataset):
    def __init__(self, rows, patch_size=256):
        self.rows = list(rows)
        self.patch_size = int(patch_size)

        for r in self.rows:
            if int(r["patch_size"]) != self.patch_size:
                raise RuntimeError(
                    f"Manifest patch_size={r['patch_size']} does not match requested "
                    f"{self.patch_size}."
                )
            # Healthy training/test splits must not contain positive labels.
            if r["split"] in {"healthy_train", "healthy_test"} and int(r["patch_label"]) != 0:
                raise RuntimeError("Healthy split contains patch_label != 0.")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        p = Path(r["source_image"])
        arr = load_native_gray(p)

        x0 = int(r["x0"])
        y0 = int(r["y0"])
        x1 = int(r["x1_valid"])
        y1 = int(r["y1_valid"])

        patch = arr[y0:y1, x0:x1]
        vh, vw = patch.shape

        expected_h = int(r["valid_height"])
        expected_w = int(r["valid_width"])
        if (vh, vw) != (expected_h, expected_w):
            raise RuntimeError(
                f"Manifest/source mismatch for {p.name} {r['patch_id']}: "
                f"actual={(vh, vw)} expected={(expected_h, expected_w)}"
            )

        patch256 = pad_patch_symmetric(patch, self.patch_size)
        patch_pad = pad_context_276(patch256)
        valid = valid_context_mask(vh, vw)

        x = torch.from_numpy(patch_pad).unsqueeze(0).float()
        m = torch.from_numpy(valid).unsqueeze(0).float()

        meta = {
            "source_image": str(p),
            "image_id": r["image_id"],
            "patch_id": r["patch_id"],
            "row_idx": int(r["row_idx"]),
            "col_idx": int(r["col_idx"]),
            "x0": x0,
            "y0": y0,
            "x1_valid": x1,
            "y1_valid": y1,
            "valid_width": vw,
            "valid_height": vh,
            "original_width": int(r["original_width"]),
            "original_height": int(r["original_height"]),
        }
        return x, m, meta


class ResidualBlock(nn.Module):
    """
    ResNet-style local residual block.
    This is NOT a U-Net encoder-to-decoder skip connection.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=True)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        out = out + identity
        return F.relu(out, inplace=True)


class NativeResNetAutoencoder(nn.Module):
    """
    Same architecture as the previous native-patch ResNet experiment.

    - NO U-Net-style long encoder->decoder skips.
    - Local residual connections exist inside residual blocks.
    - NO BatchNorm.
    - 3 downsampling stages.
    - base64 => 2048-D latent embedding:
        bottleneck channels = 512
        AdaptiveAvgPool2d(2x2) => 512*2*2 = 2048
    """
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels

        self.stem = nn.Conv2d(1, b, 3, padding=1, bias=True)
        self.enc1 = ResidualBlock(b)

        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1, bias=True)
        self.enc2 = ResidualBlock(b * 2)

        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1, bias=True)
        self.enc3 = ResidualBlock(b * 4)

        self.down3 = nn.Conv2d(b * 4, b * 8, 4, stride=2, padding=1, bias=True)

        self.bot1 = ResidualBlock(b * 8)
        self.bot2 = ResidualBlock(b * 8)

        self.up3 = nn.ConvTranspose2d(
            b * 8, b * 4, 4, stride=2, padding=1, bias=True
        )
        self.dec3 = ResidualBlock(b * 4)

        self.up2 = nn.ConvTranspose2d(
            b * 4, b * 2, 4, stride=2, padding=1, bias=True
        )
        self.dec2 = ResidualBlock(b * 2)

        self.up1 = nn.ConvTranspose2d(
            b * 2, b, 4, stride=2, padding=1, bias=True
        )
        self.dec1 = ResidualBlock(b)

        self.out = nn.Conv2d(b, 1, 1, bias=True)
        self.latent_pool = nn.AdaptiveAvgPool2d((2, 2))

    def encode_features(self, x):
        x = F.relu(self.stem(x), inplace=True)
        x = self.enc1(x)

        x = F.relu(self.down1(x), inplace=True)
        x = self.enc2(x)

        x = F.relu(self.down2(x), inplace=True)
        x = self.enc3(x)

        x = F.relu(self.down3(x), inplace=True)

        z = self.bot1(x)
        z = self.bot2(z)
        return z

    def latent_embedding(self, z):
        return self.latent_pool(z).flatten(1)

    def decode_features(self, z):
        x = F.relu(self.up3(z), inplace=True)
        x = self.dec3(x)

        x = F.relu(self.up2(x), inplace=True)
        x = self.dec2(x)

        x = F.relu(self.up1(x), inplace=True)
        x = self.dec1(x)

        return self.out(x)

    def forward(self, x, return_embedding=False):
        z = self.encode_features(x)
        emb = self.latent_embedding(z)
        recon = self.decode_features(z)
        if return_embedding:
            return recon, emb
        return recon


def masked_mse(pred, target, valid):
    sq = (pred - target) ** 2 * valid
    return sq.sum() / valid.sum().clamp_min(1.0)


def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def image_metrics(orig, recon):
    mse = float(np.mean((orig - recon) ** 2))
    mae = float(np.mean(np.abs(orig - recon)))
    psnr = psnr_from_mse(mse)

    ssim = None
    if HAS_SKIMAGE:
        min_side = min(orig.shape)
        if min_side >= 3:
            win = min(7, min_side)
            if win % 2 == 0:
                win -= 1
            ssim = float(ssim_fn(orig, recon, data_range=1.0, win_size=win))

    return {
        "mse": mse,
        "mae": mae,
        "psnr_db": psnr,
        "ssim": ssim,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows):
    out = {"n": len(rows)}
    for key in ["mse", "mae", "psnr_db", "ssim"]:
        vals = [
            r[key] for r in rows
            if r.get(key) is not None and np.isfinite(r[key])
        ]
        out[f"mean_{key}"] = float(np.mean(vals)) if vals else None
        out[f"std_{key}"] = float(np.std(vals)) if vals else None
    return out


def train_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_sq = 0.0
    total_valid = 0.0

    for x, valid, _ in loader:
        x = x.to(device, non_blocking=True)
        valid = valid.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp,
        ):
            pred = forward_276(model, x)
            loss = masked_mse(pred, x, valid)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            sq_sum = (((pred - x) ** 2) * valid).sum().detach().cpu().item()
            n_valid = valid.sum().detach().cpu().item()

        total_sq += sq_sum
        total_valid += n_valid

    return total_sq / max(total_valid, 1.0)


@torch.no_grad()
def evaluate_fixed256(
    model,
    rows,
    device,
    output_dir,
    split_name,
    patch_size=256,
    save_examples=10,
):
    model.eval()

    grouped = defaultdict(list)
    for r in rows:
        grouped[r["source_image"]].append(r)

    # Stable spatial order.
    for source_image in grouped:
        grouped[source_image].sort(
            key=lambda r: (int(r["row_idx"]), int(r["col_idx"]))
        )

    patch_rows = []
    image_rows = []
    embeddings = []
    emb_meta = []
    examples = []

    image_items = sorted(grouped.items(), key=lambda kv: kv[0])

    for image_idx, (source_image, image_patch_rows) in enumerate(image_items):
        p = Path(source_image)
        orig = load_native_gray(p)
        h, w = orig.shape

        recon_full = np.zeros_like(orig, dtype=np.float32)

        for r in image_patch_rows:
            x0 = int(r["x0"])
            y0 = int(r["y0"])
            x1 = int(r["x1_valid"])
            y1 = int(r["y1_valid"])

            patch = orig[y0:y1, x0:x1]
            ph, pw = patch.shape
            patch_pad = pad_context_276(pad_patch_symmetric(patch, patch_size))

            x = (
                torch.from_numpy(patch_pad)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            pred, emb = forward_276(model, x, return_embedding=True)
            pred = torch.clamp(pred, 0.0, 1.0)

            recon_patch = (
                pred[0, 0, CONTEXT_PIXELS:CONTEXT_PIXELS + ph,
                         CONTEXT_PIXELS:CONTEXT_PIXELS + pw]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            emb_np = (
                emb[0]
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            recon_full[y0:y1, x0:x1] = recon_patch

            m = image_metrics(patch, recon_patch)
            patch_rows.append({
                "image": p.name,
                "source_image": str(p),
                "patch_id": r["patch_id"],
                "row_idx": int(r["row_idx"]),
                "col_idx": int(r["col_idx"]),
                "x0": x0,
                "y0": y0,
                "x1_valid": x1,
                "y1_valid": y1,
                "patch_width": pw,
                "patch_height": ph,
                "padded": int(ph != patch_size or pw != patch_size),
                **m,
            })

            embeddings.append(emb_np)
            emb_meta.append({
                "image": p.name,
                "source_image": str(p),
                "patch_id": r["patch_id"],
                "row_idx": int(r["row_idx"]),
                "col_idx": int(r["col_idx"]),
                "x0": x0,
                "y0": y0,
                "valid_width": pw,
                "valid_height": ph,
                "original_width": w,
                "original_height": h,
            })

        image_rows.append({
            "image": p.name,
            "source_image": str(p),
            "width": w,
            "height": h,
            "n_patches": len(image_patch_rows),
            **image_metrics(orig, recon_full),
        })

        if len(examples) < save_examples:
            examples.append((p.name, orig, recon_full))

        if (image_idx + 1) % 25 == 0 or (image_idx + 1) == len(image_items):
            print(
                f"  {split_name}: {image_idx+1}/{len(image_items)} images"
            )

    emb_arr = np.stack(embeddings, axis=0)
    latent_dim = int(emb_arr.shape[1])

    np.savez_compressed(
        output_dir / f"{split_name}_patch_embeddings_{latent_dim}d.npz",
        embeddings=emb_arr,
        image=np.array([x["image"] for x in emb_meta]),
        source_image=np.array([x["source_image"] for x in emb_meta]),
        patch_id=np.array([x["patch_id"] for x in emb_meta]),
        row_idx=np.array([x["row_idx"] for x in emb_meta], dtype=np.int32),
        col_idx=np.array([x["col_idx"] for x in emb_meta], dtype=np.int32),
        x0=np.array([x["x0"] for x in emb_meta], dtype=np.int32),
        y0=np.array([x["y0"] for x in emb_meta], dtype=np.int32),
        valid_width=np.array([x["valid_width"] for x in emb_meta], dtype=np.int32),
        valid_height=np.array([x["valid_height"] for x in emb_meta], dtype=np.int32),
        original_width=np.array([x["original_width"] for x in emb_meta], dtype=np.int32),
        original_height=np.array([x["original_height"] for x in emb_meta], dtype=np.int32),
    )

    write_csv(
        output_dir / f"{split_name}_patch_metrics.csv",
        patch_rows,
    )
    write_csv(
        output_dir / f"{split_name}_image_metrics.csv",
        image_rows,
    )

    if examples:
        thumb = 260
        row_h = 300
        canvas = Image.new(
            "RGB",
            (thumb * 3 + 80, row_h * len(examples) + 40),
            "white",
        )
        draw = ImageDraw.Draw(canvas)

        draw.text((10, 10), "Original", fill="black")
        draw.text((thumb + 30, 10), "Reconstructed", fill="black")
        draw.text((2 * thumb + 50, 10), "Abs Error (norm)", fill="black")

        y = 40
        for name, orig, rec in examples:
            err = np.abs(orig - rec)
            err = err / (err.max() + 1e-8)

            rendered = []
            for arr in [orig, rec, err]:
                pil = Image.fromarray(
                    (np.clip(arr, 0, 1) * 255).astype(np.uint8),
                    mode="L",
                )
                # Display-only sizing. This does NOT alter training/evaluation data.
                scale = min(
                    thumb / pil.width,
                    (row_h - 35) / pil.height,
                )
                size = (
                    max(1, int(pil.width * scale)),
                    max(1, int(pil.height * scale)),
                )
                rendered.append(
                    pil.resize(size, Image.Resampling.NEAREST).convert("RGB")
                )

            draw.text((10, y), name, fill="black")
            yy = y + 25
            for j, im in enumerate(rendered):
                x_pos = 10 + j * (thumb + 20)
                canvas.paste(im, (x_pos, yy))
            y += row_h

        canvas.save(
            output_dir / f"{split_name}_reconstruction_examples.png"
        )

    return patch_rows, image_rows, latent_dim


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_root", required=True)
    ap.add_argument(
        "--manifest",
        default=r".\outputs\BRISC_FIXED256_PATCH_AUDIT\patch_manifest_256.csv",
    )
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_examples", type=int, default=10)
    ap.add_argument("--smoke_test", action="store_true", help="One real-data forward pass, no training")

    args = ap.parse_args()

    if args.patch_size != 256:
        raise ValueError(
            f"This experiment is intentionally fixed at 256x256; got {args.patch_size}."
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    root = Path(args.dataset_root)
    manifest_path = Path(args.manifest)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = read_manifest(manifest_path)
    train_rows = manifest_rows_for_split(rows, "healthy_train")
    test_rows = manifest_rows_for_split(rows, "healthy_test")

    # Verify manifest points to the same dataset root (at least existence).
    missing = [
        r["source_image"]
        for r in train_rows[:20] + test_rows[:20]
        if not Path(r["source_image"]).exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Manifest source_image paths do not exist on this machine. "
            f"First missing example: {missing[0]}"
        )

    train_images = len({r["source_image"] for r in train_rows})
    test_images = len({r["source_image"] for r in test_rows})
    train_padded = sum(
        int(int(r["valid_width"]) < 256 or int(r["valid_height"]) < 256)
        for r in train_rows
    )
    test_padded = sum(
        int(int(r["valid_width"]) < 256 or int(r["valid_height"]) < 256)
        for r in test_rows
    )

    latent_dim_expected = args.base_channels * 8 * 2 * 2

    print("=" * 92)
    print("HEALTHY FIXED-256 / SYMMETRIC CONTEXT-276 RESNET AUTOENCODER")
    print("=" * 92)
    print(f"Dataset root          : {root}")
    print(f"Manifest              : {manifest_path}")
    print(f"Healthy train images  : {train_images}")
    print(f"Healthy test images   : {test_images}")
    print(f"Train patches         : {len(train_rows)}")
    print(f"Test patches          : {len(test_rows)}")
    print(f"Train padded patches  : {train_padded}")
    print(f"Test padded patches   : {test_padded}")
    print("Spatial resize        : NONE")
    print("Patch strategy        : fixed 256x256, stride 256")
    print("Edge padding          : symmetric to 256; excluded from loss/metrics")
    print("Context padding       : symmetric 10px per side => 276x276")
    print("Model compatibility   : symmetric 2px => 280x280 internal; crop to 276")
    print("Output processing     : crop 10px => 256x256 before reassembly")
    print("U-Net long skips      : NONE")
    print("Residual connections  : LOCAL inside ResNet blocks")
    print("BatchNorm             : NONE")
    print(f"Base channels         : {args.base_channels}")
    print(f"Latent embedding      : {latent_dim_expected}-D")
    print("Image embedding avg   : NONE")
    print(f"Epochs                : {args.epochs}")
    print(f"Batch size            : {args.batch_size}")
    print()

    if train_images != 1067:
        print(f"WARNING: expected 1067 healthy train images, found {train_images}")
    if test_images != 140:
        print(f"WARNING: expected 140 healthy test images, found {test_images}")

    train_ds = Fixed256PatchDataset(train_rows, patch_size=args.patch_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    model = NativeResNetAutoencoder(
        base_channels=args.base_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    if args.smoke_test:
        model.eval()
        x0, m0, meta0 = train_ds[0]
        with torch.no_grad():
            pred0, emb0 = forward_276(
                model, x0.unsqueeze(0).to(device), return_embedding=True
            )
        assert pred0.shape[-2:] == (276, 276)
        assert emb0.shape == (1, latent_dim_expected)
        core = pred0[..., CONTEXT_PIXELS:-CONTEXT_PIXELS,
                           CONTEXT_PIXELS:-CONTEXT_PIXELS]
        assert core.shape[-2:] == (256, 256)
        assert m0.shape[-2:] == (276, 276)
        assert int(m0.sum()) == meta0["valid_height"] * meta0["valid_width"]
        print("SMOKE TEST PASSED")
        print(f"Data tensor           : {tuple(x0.shape)}")
        print(f"Model output          : {tuple(pred0.shape)}")
        print(f"Final central tile    : {tuple(core.shape)}")
        print(f"Latent embedding      : {tuple(emb0.shape)}")
        print(f"Mask valid pixels     : {int(m0.sum())}")
        print("NO training was performed.")
        return

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    print(f"Device                : {device}")
    print(f"Parameters            : {n_params:,}")
    print()

    history = []
    best_train_mse = float("inf")
    best_epoch = None

    for epoch in range(1, args.epochs + 1):
        mse = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp,
        )
        psnr = psnr_from_mse(mse)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_mse": mse,
            "train_psnr_db": psnr,
        }
        history.append(row)
        write_csv(out / "history.csv", history)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "train_mse": mse,
            "base_channels": args.base_channels,
            "architecture": "NativeResNetAutoencoder",
            "long_encoder_decoder_skip_connections": False,
            "residual_block_connections": True,
            "residual_type": "local_identity_addition",
            "batchnorm": False,
            "latent_dim": latent_dim_expected,
            "fixed_patch_size": 256,
            "patch_stride": 256,
            "context_patch_size": 276,
            "model_internal_size": 280,
            "context_padding": "symmetric_10_each_side",
            "compat_padding": "symmetric_2_each_side",
            "edge_padding": "symmetric",
            "padding_excluded_from_loss": True,
            "spatial_resize": False,
            "healthy_only": True,
            "train_images": train_images,
            "train_patches": len(train_rows),
            "manifest": str(manifest_path),
        }

        torch.save(
            ckpt,
            out / "last_autoencoder.pt",
        )

        is_best = mse < best_train_mse
        if is_best:
            best_train_mse = mse
            best_epoch = epoch
            torch.save(
                ckpt,
                out / "best_train_autoencoder.pt",
            )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.3e} | "
            f"train_mse={mse:.8e} | "
            f"train_psnr={psnr:.2f} dB"
            + ("  <-- BEST TRAIN" if is_best else "")
        )

        scheduler.step()

    print()
    print("Training finished.")
    print(f"Best train epoch      : {best_epoch}")
    print(f"Best train MSE        : {best_train_mse:.8e}")
    print("Primary evaluation    : best_train_autoencoder.pt")
    print()

    # Load best checkpoint before embeddings/evaluation.
    best_ckpt = torch.load(
        out / "best_train_autoencoder.pt",
        map_location=device,
    )
    model.load_state_dict(
        best_ckpt["model_state_dict"],
        strict=True,
    )
    model.eval()

    print("Evaluating healthy TRAIN for patch embeddings/diagnostics...")
    train_patch_rows, train_image_rows, latent_dim_train = evaluate_fixed256(
        model,
        train_rows,
        device,
        out,
        "train_healthy_fixed256",
        patch_size=args.patch_size,
        save_examples=0,
    )

    print()
    print("Evaluating independent healthy TEST...")
    test_patch_rows, test_image_rows, latent_dim_test = evaluate_fixed256(
        model,
        test_rows,
        device,
        out,
        "test_healthy_fixed256",
        patch_size=args.patch_size,
        save_examples=args.save_examples,
    )

    assert latent_dim_train == latent_dim_test == latent_dim_expected

    final = {
        "experiment": "healthy-only symmetric-context 276 ResNet autoencoder",
        "architecture": (
            "Same ResNet-style encoder-decoder as prior native-2x2 experiment; "
            "local residual blocks; no U-Net long skips"
        ),
        "healthy_only_training": True,
        "spatial_resize": False,
        "patch_strategy": "fixed 256x256 non-overlapping tiles, stride 256",
        "edge_padding": (
            "symmetric padding outside valid source region; "
            "padding excluded from loss and reconstruction metrics"
        ),
        "context_padding": "10px symmetric on 256 tile => 276",
        "model_internal_padding": "2px symmetric on 276 => 280; crop output to 276",
        "final_unpadding": "crop context 10px each side => 256 before reassembly",
        "long_encoder_decoder_skip_connections": False,
        "residual_block_connections": True,
        "batchnorm": False,
        "latent_embedding_dim": latent_dim_expected,
        "patch_embeddings_kept_separate": True,
        "image_embedding_average": False,
        "train_images": train_images,
        "train_patches": len(train_rows),
        "train_padded_patches": train_padded,
        "test_images": test_images,
        "test_patches": len(test_rows),
        "test_padded_patches": test_padded,
        "epochs": args.epochs,
        "best_train_epoch": best_epoch,
        "best_train_mse": best_train_mse,
        "base_channels": args.base_channels,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "min_lr": args.min_lr,
        "model_params": n_params,
        "train_patch_metrics": summarize(train_patch_rows),
        "train_whole_image_metrics": summarize(train_image_rows),
        "test_patch_metrics": summarize(test_patch_rows),
        "test_whole_image_metrics": summarize(test_image_rows),
        "primary_result": "test_whole_image_metrics",
        "previous_reference": {
            "fixed256_resnet_base64_whole_image_psnr_db": 63.4857973,
            "native_2x2_resnet_base64_low_lr_best_psnr_db": 64.509215,
            "note": (
                "Previous reference used native 2x2 patches with variable patch sizes. "
                "Current run changes patch preparation to fixed 256x256 without resize."
            ),
        },
        "note": (
            "No patch embeddings are averaged. Healthy patch embeddings are saved "
            "individually for later PCA/LLE/t-SNE and cosine-similarity analysis."
        ),
    }

    (out / "final_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== FINAL HEALTHY TEST - WHOLE NATIVE IMAGE ===")
    print(json.dumps(final["test_whole_image_metrics"], indent=2))

    print()
    print("=== FINAL HEALTHY TEST - PATCH LEVEL ===")
    print(json.dumps(final["test_patch_metrics"], indent=2))

    print()
    new_psnr = final["test_whole_image_metrics"]["mean_psnr_db"]
    delta = new_psnr - 64.509215
    print("=== COMPARISON WITH PREVIOUS BEST ===")
    print(f"Prior fixed-256 whole image : 63.4858 dB")
    print(f"Symmetric-276 whole image   : {new_psnr:.4f} dB")
    print(f"Delta vs fixed-256          : {new_psnr-63.4857973:+.4f} dB")
    print(f"Prior native-2x2 best       : 64.5092 dB")
    print(f"Delta vs native-2x2        : {delta:+.4f} dB")

    print()
    print("Saved outputs:", out)
    print(
        "Patch embeddings:",
        out / f"train_healthy_fixed256_patch_embeddings_{latent_dim_expected}d.npz",
    )


if __name__ == "__main__":
    main()
