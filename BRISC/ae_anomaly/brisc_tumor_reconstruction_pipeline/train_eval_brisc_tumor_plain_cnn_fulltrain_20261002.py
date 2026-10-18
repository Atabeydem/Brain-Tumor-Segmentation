#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ============================================================
# Data
# ============================================================
def list_images(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_gray(path: Path, size: int, nearest=False):
    with Image.open(path) as im:
        im = im.convert("L")
        rs = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        im = im.resize((size, size), resample=rs)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def processed_hash(path: Path, size: int):
    with Image.open(path) as im:
        im = im.convert("L")
        im = im.resize((size, size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


class ImageDataset(Dataset):
    def __init__(self, paths, image_size):
        self.paths = list(paths)
        self.image_size = image_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        x = load_gray(p, self.image_size, nearest=False)
        return x, str(p)


class PairDataset(Dataset):
    def __init__(self, pairs, image_size):
        self.pairs = list(pairs)
        self.image_size = image_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        x = load_gray(img_path, self.image_size, nearest=False)
        m = load_gray(mask_path, self.image_size, nearest=True)
        return x, m, str(img_path)


# ============================================================
# Plain CNN Autoencoder
# No skip connections, no BatchNorm, linear output.
# ============================================================
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


class PlainCNNAutoencoder(nn.Module):
    def __init__(self, base_channels=64):
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

    def forward(self, x):
        x = self.enc1(x)
        x = torch.relu(self.down1(x))
        x = self.enc2(x)

        x = torch.relu(self.down2(x))
        x = self.enc3(x)

        x = torch.relu(self.down3(x))
        x = self.bottleneck(x)

        x = torch.relu(self.up3(x))
        x = self.dec3(x)

        x = torch.relu(self.up2(x))
        x = self.dec2(x)

        x = torch.relu(self.up1(x))
        x = self.dec1(x)

        return self.out(x)


# ============================================================
# Metrics
# ============================================================
def psnr_from_mse(mse):
    if mse <= 0:
        return float("inf")
    return float(10.0 * math.log10(1.0 / mse))


def region_metrics(target, recon, mask):
    diff = target - recon
    sq = diff ** 2
    ab = np.abs(diff)

    vals_sq = sq[mask]
    vals_ab = ab[mask]
    mse = float(vals_sq.mean())
    mae = float(vals_ab.mean())
    return mse, mae, psnr_from_mse(mse)


def eval_one(target, recon, mask):
    full_mse = float(np.mean((target - recon) ** 2))
    full_mae = float(np.mean(np.abs(target - recon)))
    full_psnr = psnr_from_mse(full_mse)

    mask = mask.astype(bool)
    nonmask = ~mask

    mask_mse, mask_mae, mask_psnr = region_metrics(target, recon, mask)
    non_mse, non_mae, non_psnr = region_metrics(target, recon, nonmask)

    ssim = float(ssim_fn(target, recon, data_range=1.0)) if HAS_SKIMAGE else None

    return {
        "full_mse": full_mse,
        "full_mae": full_mae,
        "full_psnr_db": full_psnr,
        "mask_mse": mask_mse,
        "mask_mae": mask_mae,
        "mask_psnr_db": mask_psnr,
        "nonmask_mse": non_mse,
        "nonmask_mae": non_mae,
        "nonmask_psnr_db": non_psnr,
        "mask_area_ratio": float(mask.mean()),
        "ssim": ssim,
    }


def summarize(rows):
    result = {"n_images": len(rows)}
    keys = [
        "full_mse", "full_mae", "full_psnr_db",
        "mask_mse", "mask_mae", "mask_psnr_db",
        "nonmask_mse", "nonmask_mae", "nonmask_psnr_db",
        "mask_area_ratio", "ssim"
    ]
    for k in keys:
        vals = [r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        result[f"mean_{k}"] = float(np.mean(vals)) if vals else None
        result[f"std_{k}"] = float(np.std(vals)) if vals else None
    return result


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def to_pil(arr):
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8), mode="L")


def save_examples(examples, out_path, cell=180):
    if not examples:
        return
    labels = ["Original", "Reconstructed", "Abs Error", "GT Mask"]
    margin, label_h, name_h = 12, 26, 22
    width = margin + len(labels) * (cell + margin)
    height = margin + len(examples) * (cell + label_h + name_h + margin)

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)

    for c, label in enumerate(labels):
        draw.text((margin + c*(cell+margin), 4), label, fill="black")

    y = label_h + margin
    for ex in examples:
        draw.text((margin, y), ex["name"], fill="black")
        yy = y + name_h

        err = np.abs(ex["orig"] - ex["recon"])
        err_norm = err / (err.max() + 1e-8)
        imgs = [
            to_pil(ex["orig"]),
            to_pil(ex["recon"]),
            to_pil(err_norm),
            Image.fromarray((ex["mask"].astype(np.uint8) * 255), mode="L"),
        ]
        for c, im in enumerate(imgs):
            canvas.paste(
                im.convert("RGB").resize((cell, cell)),
                (margin + c*(cell+margin), yy)
            )
        y = yy + cell + margin

    canvas.save(out_path)


# ============================================================
# Train / Eval
# ============================================================
def train_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    total_loss = 0.0
    total_n = 0
    loss_fn = nn.MSELoss()

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp
        ):
            y = model(x)
            loss = loss_fn(y, x)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += float(loss.detach().cpu()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, pairs, image_size, device, save_n=12):
    model.eval()
    rows = []
    examples = []

    for idx, (img_path, mask_path) in enumerate(pairs):
        x = load_gray(img_path, image_size, nearest=False).unsqueeze(0).to(device)
        mask = load_gray(mask_path, image_size, nearest=True).squeeze().numpy() > 0.5

        recon = torch.clamp(model(x), 0.0, 1.0)

        orig_np = x.detach().cpu().squeeze().numpy()
        recon_np = recon.detach().cpu().squeeze().numpy()

        m = eval_one(orig_np, recon_np, mask)
        rows.append({"image": img_path.name, **m})

        if len(examples) < save_n:
            examples.append({
                "name": img_path.name,
                "orig": orig_np,
                "recon": recon_np,
                "mask": mask,
            })

        if (idx + 1) % 100 == 0:
            print(f"  evaluated {idx+1}/{len(pairs)}")

    return rows, examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=4)  # kept for CLI compatibility
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_examples", type=int, default=12)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    root = Path(args.dataset_root)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_dir = root / "segmentation_task" / "train" / "images"
    test_img_dir = root / "segmentation_task" / "test" / "images"
    test_mask_dir = root / "segmentation_task" / "test" / "masks"

    train_images = list_images(train_dir)
    test_images = list_images(test_img_dir)
    test_masks = list_images(test_mask_dir)
    mask_map = {p.stem: p for p in test_masks}
    test_pairs = [(p, mask_map[p.stem]) for p in test_images if p.stem in mask_map]

    if len(train_images) != 3933:
        print(f"WARNING: expected 3933 train images, found {len(train_images)}")
    if len(test_pairs) != 860:
        print(f"WARNING: expected 860 test pairs, found {len(test_pairs)}")

    # clean non-overlap test subset after actual preprocessing
    print("Building post-preprocessing train hashes...")
    train_hashes = set()
    for i, p in enumerate(train_images, 1):
        train_hashes.add(processed_hash(p, args.image_size))
        if i % 500 == 0:
            print(f"  train hashed {i}/{len(train_images)}")

    clean_pairs = []
    overlap_pairs = []
    for img, mask in test_pairs:
        if processed_hash(img, args.image_size) in train_hashes:
            overlap_pairs.append((img, mask))
        else:
            clean_pairs.append((img, mask))

    print(f"Provided tumor train images: {len(train_images)}")
    print(f"Provided tumor test pairs: {len(test_pairs)}")
    print(f"Excluded post-preprocessing overlaps: {len(overlap_pairs)}")
    print(f"Clean test pairs: {len(clean_pairs)}")

    excluded = [{"image": i.name, "mask": m.name} for i, m in overlap_pairs]
    write_csv(out_dir / "excluded_overlap_test_images.csv", excluded)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"

    model = PlainCNNAutoencoder(base_channels=args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    train_ds = ImageDataset(train_images, args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.workers > 0),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_lr,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print()
    print("=" * 80)
    print("PLAIN CNN TUMOR SELF-RECONSTRUCTION - FULL TRAIN")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Model params: {n_params:,}")
    print(f"Training images: ALL {len(train_images)} provided tumor-train images")
    print("Validation split: NONE")
    print(f"External clean test: {len(clean_pairs)} images")
    print("Target: tumor image -> same tumor image")
    print("Architecture: plain convolutional autoencoder, NO skip connections, NO BatchNorm")
    print(f"Image size: {args.image_size} | Base channels: {args.base_channels}")
    print(f"Batch: {args.batch_size} | Epochs: {args.epochs} | LR: {args.lr}")
    print()

    history = []
    best_train_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_mse = train_epoch(
            model, train_loader, optimizer, scaler, device, use_amp
        )
        train_psnr = psnr_from_mse(train_mse)
        lr_now = optimizer.param_groups[0]["lr"]

        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_mse": train_mse,
            "train_psnr_db": train_psnr,
        }
        history.append(row)
        write_csv(out_dir / "history.csv", history)

        ckpt = {
            "model_state_dict": model.state_dict(),
            "epoch": epoch,
            "train_mse": train_mse,
            "image_size": args.image_size,
            "base_channels": args.base_channels,
            "architecture": "PlainCNNAutoencoder_3down_no_skip_no_bn",
            "training_images": len(train_images),
            "target": "tumor image -> same tumor image",
        }

        torch.save(ckpt, out_dir / "last_autoencoder.pt")

        is_best = train_mse < best_train_mse
        if is_best:
            best_train_mse = train_mse
            torch.save(ckpt, out_dir / "best_train_autoencoder.pt")

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"lr={lr_now:.3e} | "
            f"train_mse={train_mse:.8e} | "
            f"train_psnr={train_psnr:.2f} dB"
            + ("  <-- BEST TRAIN" if is_best else "")
        )

        scheduler.step()

    print()
    print("Training finished.")
    print("Primary evaluation checkpoint: last_autoencoder.pt (fixed 200-epoch protocol)")
    print()

    # Fixed protocol: evaluate LAST model, not train-selected best.
    model.eval()

    print("Evaluating PROVIDED 860 test split...")
    all_rows, _ = evaluate(
        model, test_pairs, args.image_size, device, save_n=0
    )
    all_summary = summarize(all_rows)
    write_csv(out_dir / "test860_metrics_per_case.csv", all_rows)
    (out_dir / "test860_summary.json").write_text(
        json.dumps(all_summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("Evaluating CLEAN non-overlap test subset...")
    clean_rows, examples = evaluate(
        model, clean_pairs, args.image_size, device, save_n=args.save_examples
    )
    clean_summary = summarize(clean_rows)
    write_csv(out_dir / "clean_test_metrics_per_case.csv", clean_rows)
    (out_dir / "clean_test_summary.json").write_text(
        json.dumps(clean_summary, indent=2),
        encoding="utf-8",
    )
    save_examples(
        examples,
        out_dir / "clean_test_reconstruction_examples.png"
    )

    final_summary = {
        "experiment": "Plain CNN tumor self-reconstruction - all 3933 provided tumor-train images",
        "architecture": "PlainCNNAutoencoder_3down_no_skip_no_bn",
        "model_params": n_params,
        "training_images": len(train_images),
        "validation_images": 0,
        "provided_test_n": len(test_pairs),
        "excluded_overlap_n": len(overlap_pairs),
        "clean_test_n": len(clean_pairs),
        "epochs": args.epochs,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "batch_size": args.batch_size,
        "initial_lr": args.lr,
        "min_lr": args.min_lr,
        "checkpoint_for_evaluation": "last_autoencoder.pt",
        "provided_test": all_summary,
        "clean_test": clean_summary,
        "note": (
            "All 3933 provided segmentation_task/train tumor images participate in gradient updates. "
            "No validation set is held out. Hyperparameters are fixed before training. "
            "Primary generalization result is the post-preprocessing non-overlap clean test subset."
        ),
    }
    (out_dir / "final_summary.json").write_text(
        json.dumps(final_summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=== FINAL CLEAN TEST ===")
    print(json.dumps(clean_summary, indent=2))
    print()
    print("Saved outputs:", out_dir)


if __name__ == "__main__":
    main()
