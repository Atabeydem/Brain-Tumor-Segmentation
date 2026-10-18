r"""
Train a 2D U-Net baseline for BRISC tumor segmentation.

Run from:
    C:\Users\Atabey\Brain-Tumor-Segmentation\BRISC\segmentation

Quick CPU test:
    python .\train_brisc_unet.py --image_size 128 --batch_size 8 --epochs 2 --output_dir "..\outputs\unet_baseline_128_e2"

More meaningful baseline:
    python .\train_brisc_unet.py --image_size 128 --batch_size 8 --epochs 10 --output_dir "..\outputs\unet_baseline_128_e10"
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "brisc_Dataset2025" / "brisc2025"


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False,
        })
        .fillna(False)
        .astype(bool)
    )


def norm_path(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/")


def norm_key(value) -> str:
    return norm_path(value).lower()


def simplified_stem(value) -> str:
    text = norm_key(value)
    if not text:
        return ""
    stem = Path(text).stem.lower()
    stem = re.sub(r"(_mask|_masks|_seg|_segmentation|-mask|-seg|-segmentation)$", "", stem)
    return stem


def row_pair_key(row) -> str:
    parts = []
    for col in ["split", "index", "tumor_code", "plane_code", "sequence"]:
        if col in row and not pd.isna(row[col]):
            parts.append(str(row[col]).strip().lower())
    return "|".join(parts)


def build_mask_lookup(masks: pd.DataFrame) -> dict:
    lookup = {}

    def add(key, row):
        if key and key not in lookup:
            lookup[key] = row

    for _, row in masks.iterrows():
        rel = norm_key(row.get("relative_path", ""))
        filename = norm_key(row.get("filename", ""))
        linked = norm_key(row.get("linked_image", ""))

        for value in [linked, Path(linked).name if linked else ""]:
            add(value, row)
            add(simplified_stem(value), row)

        for value in [rel, filename, Path(rel).name if rel else ""]:
            add(value, row)
            add(simplified_stem(value), row)

        add(row_pair_key(row), row)

    return lookup


def find_mask_for_image(image_row, mask_lookup):
    rel = norm_key(image_row.get("relative_path", ""))
    filename = norm_key(image_row.get("filename", ""))
    candidates = [
        rel,
        filename,
        Path(rel).name if rel else "",
        simplified_stem(rel),
        simplified_stem(filename),
        row_pair_key(image_row),
    ]
    for key in candidates:
        if key and key in mask_lookup:
            return mask_lookup[key]
    return None


class BRISCSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        image_size: int = 128,
        augment: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.augment = augment

        manifest_path = self.dataset_path / "manifest.csv"
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        required_columns = {"relative_path", "split", "task", "is_mask"}
        missing = required_columns.difference(manifest.columns)
        if missing:
            raise ValueError("manifest.csv missing columns: " + ", ".join(sorted(missing)))

        manifest["is_mask"] = to_bool_series(manifest["is_mask"])
        df = manifest[(manifest["task"] == "segmentation") & (manifest["split"] == split)].copy()
        images = df[~df["is_mask"]].reset_index(drop=True)
        masks = df[df["is_mask"]].reset_index(drop=True)

        mask_lookup = build_mask_lookup(masks)
        pairs = []
        missing_pairs = []
        for _, image_row in images.iterrows():
            mask_row = find_mask_for_image(image_row, mask_lookup)
            if mask_row is None:
                missing_pairs.append(image_row.get("relative_path", image_row.get("filename", "?")))
            else:
                pairs.append((image_row, mask_row))

        if not pairs:
            raise ValueError(f"No segmentation image-mask pairs found for split={split}.")
        if missing_pairs:
            examples = "\n".join([str(x) for x in missing_pairs[:5]])
            raise ValueError(
                f"Found {len(missing_pairs)} images without masks for split={split}.\n"
                f"Examples:\n{examples}"
            )

        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def _load_image(self, relative_path: str) -> torch.Tensor:
        image_path = self.dataset_path / norm_path(relative_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("L")
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        arr = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(arr[None, :, :])

    def _load_mask(self, relative_path: str) -> torch.Tensor:
        mask_path = self.dataset_path / norm_path(relative_path)
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")

        mask = Image.open(mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = np.asarray(mask).astype(np.float32)
        arr = (arr > 0).astype(np.float32)
        return torch.from_numpy(arr[None, :, :])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_row, mask_row = self.pairs[idx]
        image = self._load_image(image_row["relative_path"])
        mask = self._load_mask(mask_row["relative_path"])

        if self.augment:
            if random.random() < 0.5:
                image = torch.flip(image, dims=[2])
                mask = torch.flip(mask, dims=[2])
            if random.random() < 0.5:
                image = torch.flip(image, dims=[1])
                mask = torch.flip(mask, dims=[1])

        return image, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c, c * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec4 = DoubleConv(c * 16 + c * 8, c * 8)
        self.up3 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec3 = DoubleConv(c * 8 + c * 4, c * 4)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec2 = DoubleConv(c * 4 + c * 2, c * 2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = DoubleConv(c * 2 + c, c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    dims = (1, 2, 3)
    intersection = torch.sum(preds * targets, dims)
    union = torch.sum(preds, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = torch.sum(probs * targets, dims)
    union = torch.sum(probs, dims) + torch.sum(targets, dims)
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = soft_dice_loss(logits, targets)
    return bce + dice


def run_one_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_items = 0

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for images, masks in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = segmentation_loss(logits, masks)
            dice = dice_from_logits(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_dice += dice.item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.4f}", dice=f"{dice.item():.4f}")

    return total_loss / max(total_items, 1), total_dice / max(total_items, 1)


def save_prediction_grid(model, loader, device, save_path: Path, max_items: int = 6):
    import matplotlib.pyplot as plt

    model.eval()
    images, masks = next(iter(loader))
    images = images[:max_items].to(device)
    masks = masks[:max_items].to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(images))
        preds = (probs > 0.5).float()

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()

    rows = min(max_items, images.shape[0])
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        axes[i, 0].imshow(images[i, 0], cmap="gray")
        axes[i, 0].set_title("MRI image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Predicted mask")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title("Ground truth mask")
        axes[i, 2].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)
    brisc_dir = Path(__file__).resolve().parent.parent
    return brisc_dir / "outputs" / "unet_baseline_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="BRISC 2D U-Net segmentation baseline")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== BRISC U-Net Segmentation Baseline ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Using device: {device}")

    train_ds = BRISCSegmentationDataset(
        dataset_path=str(dataset_path),
        split=args.train_split,
        image_size=args.image_size,
        augment=True,
    )
    val_ds = BRISCSegmentationDataset(
        dataset_path=str(dataset_path),
        split=args.val_split,
        image_size=args.image_size,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet2D(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    print(f"Train pairs: {len(train_ds)}")
    print(f"Val pairs: {len(val_ds)}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")

    config = vars(args).copy()
    config["dataset_path"] = str(dataset_path)
    config["output_dir"] = str(output_dir.resolve())
    config["device"] = str(device)
    config["model"] = "UNet2D"
    (output_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    best_val_loss = float("inf")
    best_val_dice = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_dice = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_loss, val_dice = run_one_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": current_lr,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.5f} | val_dice={val_dice:.4f} | lr={current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "best_unet.pt")
            save_prediction_grid(model, val_loader, device, output_dir / "best_prediction_examples.png")
            print(f"  Saved best checkpoint. best_val_loss={best_val_loss:.5f}, best_val_dice={best_val_dice:.4f}")

    torch.save(model.state_dict(), output_dir / "last_unet_state_dict.pt")
    print("=== Training completed ===")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
