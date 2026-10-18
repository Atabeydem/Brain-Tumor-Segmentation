#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import random
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

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PATCH_NAMES = ["TL", "TR", "BL", "BR"]


def list_images(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def split_boxes_2x2(width, height):
    xm, ym = width // 2, height // 2
    return {
        "TL": (0, 0, xm, ym),
        "TR": (xm, 0, width, ym),
        "BL": (0, ym, xm, height),
        "BR": (xm, ym, width, height),
    }


def load_native_gray(path: Path):
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


class NativePatchDataset(Dataset):
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)
        self.records = []
        for image_idx, p in enumerate(self.image_paths):
            with Image.open(p) as im:
                w, h = im.size
            boxes = split_boxes_2x2(w, h)
            for patch_name in PATCH_NAMES:
                self.records.append((image_idx, patch_name, boxes[patch_name]))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        image_idx, patch_name, box = self.records[idx]
        p = self.image_paths[image_idx]
        arr = load_native_gray(p)

        x0, y0, x1, y1 = box
        patch = arr[y0:y1, x0:x1]
        x = torch.from_numpy(patch).unsqueeze(0)

        return {
            "x": x,
            "image_idx": image_idx,
            "image_name": p.name,
            "image_path": str(p),
            "patch_name": patch_name,
            "box": box,
            "native_h": patch.shape[0],
            "native_w": patch.shape[1],
        }


def round_up(v, multiple=8):
    return ((v + multiple - 1) // multiple) * multiple


def collate_native(batch):
    # Padding only. No spatial interpolation / resize.
    max_h = round_up(max(item["native_h"] for item in batch), 8)
    max_w = round_up(max(item["native_w"] for item in batch), 8)

    xs, masks, meta = [], [], []

    for item in batch:
        x = item["x"]
        _, h, w = x.shape

        pad_r = max_w - w
        pad_b = max_h - h

        x_pad = F.pad(
            x.unsqueeze(0),
            (0, pad_r, 0, pad_b),
            mode="replicate"
        ).squeeze(0)

        valid = torch.zeros((1, max_h, max_w), dtype=torch.float32)
        valid[:, :h, :w] = 1.0

        xs.append(x_pad)
        masks.append(valid)
        meta.append({
            k: item[k]
            for k in [
                "image_idx", "image_name", "image_path",
                "patch_name", "box", "native_h", "native_w"
            ]
        })

    return torch.stack(xs, 0), torch.stack(masks, 0), meta


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class NativePlainCNNAutoencoderSkipAdd(nn.Module):
    """
    Matched skip-ON version of the previous Plain CNN AE.

    SAME as skip-OFF baseline:
      - healthy-only
      - native 2x2 patches
      - no spatial resize
      - no BatchNorm
      - same base channels
      - same encoder depth
      - same decoder depth
      - same 1024-D latent embedding

    ONLY intended architectural change:
      encoder features are added to decoder features
      at matching spatial/channel resolutions.

    Additive skips introduce zero trainable parameters.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(1, b)
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)

        self.enc2 = ConvBlock(b * 2, b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)

        self.enc3 = ConvBlock(b * 4, b * 4)
        self.down3 = nn.Conv2d(b * 4, b * 8, 4, stride=2, padding=1)

        self.bottleneck = ConvBlock(b * 8, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(b * 4, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(b * 2, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(b, b)

        self.out = nn.Conv2d(b, 1, 1)

        self.latent_pool = nn.AdaptiveAvgPool2d((2, 2))

    def encode(self, x):
        e1 = self.enc1(x)
        x = F.relu(self.down1(e1), inplace=True)

        e2 = self.enc2(x)
        x = F.relu(self.down2(e2), inplace=True)

        e3 = self.enc3(x)
        x = F.relu(self.down3(e3), inplace=True)

        z = self.bottleneck(x)
        return e1, e2, e3, z

    def latent_embedding(self, z):
        return self.latent_pool(z).flatten(1)

    def decode(self, e1, e2, e3, z):
        x = F.relu(self.up3(z), inplace=True)
        x = x + e3
        x = self.dec3(x)

        x = F.relu(self.up2(x), inplace=True)
        x = x + e2
        x = self.dec2(x)

        x = F.relu(self.up1(x), inplace=True)
        x = x + e1
        x = self.dec1(x)

        return self.out(x)

    def forward(self, x, return_embedding=False):
        e1, e2, e3, z = self.encode(x)
        emb = self.latent_embedding(z)
        recon = self.decode(e1, e2, e3, z)

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
    ssim = float(ssim_fn(orig, recon, data_range=1.0)) if HAS_SKIMAGE else None
    return {"mse": mse, "mae": mae, "psnr_db": psnr, "ssim": ssim}


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
            enabled=use_amp
        ):
            pred = model(x)
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
def evaluate_native(model, image_paths, device, output_dir, split_name, save_examples=10):
    model.eval()

    patch_rows = []
    image_rows = []
    patch_embeddings = []
    patch_meta = []
    image_embeddings = []
    image_names = []
    examples = []

    for image_idx, p in enumerate(image_paths):
        orig = load_native_gray(p)
        h, w = orig.shape
        boxes = split_boxes_2x2(w, h)

        recon_full = np.zeros_like(orig, dtype=np.float32)
        image_patch_embs = []

        for patch_name in PATCH_NAMES:
            x0, y0, x1, y1 = boxes[patch_name]
            patch = orig[y0:y1, x0:x1]
            ph, pw = patch.shape

            pad_h = round_up(ph, 8)
            pad_w = round_up(pw, 8)

            x = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
            x = F.pad(
                x,
                (0, pad_w - pw, 0, pad_h - ph),
                mode="replicate"
            ).to(device)

            pred, emb = model(x, return_embedding=True)
            pred = torch.clamp(pred, 0.0, 1.0)

            recon_patch = pred[0, 0, :ph, :pw].cpu().numpy()
            emb_np = emb[0].cpu().numpy().astype(np.float32)

            recon_full[y0:y1, x0:x1] = recon_patch

            patch_rows.append({
                "image": p.name,
                "patch": patch_name,
                "patch_width": pw,
                "patch_height": ph,
                **image_metrics(patch, recon_patch)
            })

            patch_embeddings.append(emb_np)
            patch_meta.append((p.name, patch_name, pw, ph))
            image_patch_embs.append(emb_np)

        image_rows.append({
            "image": p.name,
            "width": w,
            "height": h,
            **image_metrics(orig, recon_full)
        })

        image_embeddings.append(
            np.mean(np.stack(image_patch_embs, axis=0), axis=0)
        )
        image_names.append(p.name)

        if len(examples) < save_examples:
            examples.append((p.name, orig, recon_full))

        if (image_idx + 1) % 25 == 0:
            print(f"  {split_name}: {image_idx+1}/{len(image_paths)} images")

    patch_embeddings = np.stack(patch_embeddings, axis=0)
    image_embeddings = np.stack(image_embeddings, axis=0)

    np.savez_compressed(
        output_dir / f"{split_name}_patch_embeddings_1024d.npz",
        embeddings=patch_embeddings,
        image=np.array([x[0] for x in patch_meta]),
        patch=np.array([x[1] for x in patch_meta]),
        patch_width=np.array([x[2] for x in patch_meta]),
        patch_height=np.array([x[3] for x in patch_meta]),
    )

    np.savez_compressed(
        output_dir / f"{split_name}_image_embeddings_1024d.npz",
        embeddings=image_embeddings,
        image=np.array(image_names),
    )

    write_csv(output_dir / f"{split_name}_patch_metrics.csv", patch_rows)
    write_csv(output_dir / f"{split_name}_image_metrics.csv", image_rows)

    if examples:
        thumb = 260
        row_h = 300
        canvas = Image.new(
            "RGB",
            (thumb * 3 + 80, row_h * len(examples) + 40),
            "white"
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
                    mode="L"
                )
                scale = min(
                    thumb / pil.width,
                    (row_h - 35) / pil.height
                )
                sz = (
                    max(1, int(pil.width * scale)),
                    max(1, int(pil.height * scale))
                )
                rendered.append(
                    pil.resize(sz, Image.Resampling.NEAREST).convert("RGB")
                )

            draw.text((10, y), name, fill="black")
            yy = y + 25

            for j, im in enumerate(rendered):
                x = 10 + j * (thumb + 20)
                canvas.paste(im, (x, yy))

            y += row_h

        canvas.save(
            output_dir / f"{split_name}_reconstruction_examples.png"
        )

    return patch_rows, image_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_examples", type=int, default=10)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir = root / "classification_task" / "train" / "no_tumor"
    test_dir = root / "classification_task" / "test" / "no_tumor"

    train_images = list_images(train_dir)
    test_images = list_images(test_dir)

    print("=" * 88)
    print("HEALTHY NATIVE-PATCH MATCHED CNN AUTOENCODER - ADDITIVE SKIP ON")
    print("=" * 88)
    print(f"Healthy train images : {len(train_images)}")
    print(f"Healthy test images  : {len(test_images)}")
    print(f"Train patches        : {len(train_images) * 4}")
    print(f"Test patches         : {len(test_images) * 4}")
    print("Spatial resize       : NONE")
    print("Patch strategy       : native 2x2")
    print("Temporary padding    : replicate padding only")
    print("Skip connections     : ON (element-wise additive)")
    print("Skip parameters      : 0 additional trainable params")
    print("BatchNorm            : NONE")
    print(f"Base channels        : {args.base_channels}")
    print("Latent embedding     : 1024-D")
    print(f"Epochs               : {args.epochs}")
    print(f"Batch size           : {args.batch_size}")
    print()

    if len(train_images) != 1067:
        print(f"WARNING: expected 1067 healthy train images, found {len(train_images)}")
    if len(test_images) != 140:
        print(f"WARNING: expected 140 healthy test images, found {len(test_images)}")

    train_ds = NativePatchDataset(train_images)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.workers > 0),
        collate_fn=collate_native,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = NativePlainCNNAutoencoderSkipAdd(
        base_channels=args.base_channels
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(f"Device               : {device}")
    print(f"Parameters           : {n_params:,}")
    print()

    history = []
    best_train_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        mse = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            use_amp
        )
        psnr = psnr_from_mse(mse)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_mse": mse,
            "train_psnr_db": psnr
        }
        history.append(row)
        write_csv(out / "history.csv", history)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "train_mse": mse,
            "base_channels": args.base_channels,
            "architecture": "NativePlainCNNAutoencoderSkipAdd",
            "skip_connections": True,
            "skip_type": "elementwise_addition",
            "skip_additional_trainable_parameters": 0,
            "latent_dim": args.base_channels * 8 * 2 * 2,
            "native_2x2_patches": True,
            "spatial_resize": False,
            "healthy_only": True,
            "train_images": len(train_images),
            "train_patches": len(train_ds),
        }

        torch.save(ckpt, out / "last_autoencoder.pt")

        is_best = mse < best_train_mse
        if is_best:
            best_train_mse = mse
            torch.save(ckpt, out / "best_train_autoencoder.pt")

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
    print("Primary evaluation checkpoint: last_autoencoder.pt")
    print()

    print("Evaluating healthy TRAIN for embeddings/diagnostics...")
    train_patch_rows, train_image_rows = evaluate_native(
        model,
        train_images,
        device,
        out,
        "train_healthy",
        save_examples=0
    )

    print()
    print("Evaluating independent healthy TEST...")
    test_patch_rows, test_image_rows = evaluate_native(
        model,
        test_images,
        device,
        out,
        "test_healthy",
        save_examples=args.save_examples
    )

    final = {
        "experiment": "matched healthy-only native 2x2 patch CNN autoencoder - additive skip ON",
        "healthy_only_training": True,
        "spatial_resize": False,
        "native_patch_strategy": "fixed 2x2",
        "temporary_padding": "replicate padding, cropped before metrics",
        "skip_connections": True,
        "skip_type": "elementwise_addition",
        "skip_additional_trainable_parameters": 0,
        "batchnorm": False,
        "latent_embedding_dim": args.base_channels * 8 * 2 * 2,
        "train_images": len(train_images),
        "train_patches": len(train_ds),
        "test_images": len(test_images),
        "test_patches": len(test_images) * 4,
        "epochs": args.epochs,
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
        "comparison_target": {
            "previous_skip_off_whole_test_psnr_db": 42.92278409164336,
            "previous_skip_off_whole_test_ssim": 0.991892501941177,
            "note": "Compare only because preprocessing/training protocol/base channels/depth are matched; intended variable is additive skip connections."
        }
    }

    (out / "final_summary.json").write_text(
        json.dumps(final, indent=2),
        encoding="utf-8"
    )

    print()
    print("=== FINAL HEALTHY TEST - WHOLE NATIVE IMAGE ===")
    print(json.dumps(final["test_whole_image_metrics"], indent=2))

    print()
    print("=== FINAL HEALTHY TEST - PATCH LEVEL ===")
    print(json.dumps(final["test_patch_metrics"], indent=2))

    print()
    print("=== MATCHED SKIP COMPARISON ===")
    print(f"Skip OFF reference PSNR : {42.92278409164336:.4f} dB")
    print(
        f"Skip ON current PSNR    : "
        f"{final['test_whole_image_metrics']['mean_psnr_db']:.4f} dB"
    )
    print(
        f"Delta (ON - OFF)        : "
        f"{final['test_whole_image_metrics']['mean_psnr_db'] - 42.92278409164336:+.4f} dB"
    )

    print()
    print("Saved outputs:", out)


if __name__ == "__main__":
    main()
