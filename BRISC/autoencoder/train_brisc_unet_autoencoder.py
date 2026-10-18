r"""
Train a U-Net-style 2D denoising autoencoder on the BRISC 2025 dataset.

This script is intended as the next experiment after the plain convolutional
autoencoder. It uses skip connections to preserve spatial details and reduce
blurry reconstructions.

Expected dataset structure:
    brisc2025/
        manifest.csv
        classification_task/...
        segmentation_task/...

Run from:
    C:\Users\Atabey\Brain-Tumor-Segmentation\BRISC\autoencoder

Quick experiment:
    python .\train_brisc_unet_autoencoder.py --task segmentation --image_size 128 --batch_size 16 --epochs 5 --random_erasing_prob 0.0 --noise_std 0.03 --output_dir "..\outputs\autoencoder_seg_128_e5_unet_ae"
"""

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "brisc_Dataset2025" / "brisc2025"


def to_bool_series(series: pd.Series) -> pd.Series:
    """
    Convert manifest boolean-like values to real booleans safely.

    This avoids the pandas issue where astype(bool) turns the string 'False'
    into True because every non-empty string is truthy.
    """
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "1": True,
                "yes": True,
                "y": True,
                "false": False,
                "0": False,
                "no": False,
                "n": False,
            }
        )
        .fillna(False)
        .astype(bool)
    )


class BRISCAutoencoderDataset(Dataset):
    """
    Loads only BRISC image files, not masks.

    The model receives a corrupted/noisy image and reconstructs the clean image.
    """

    def __init__(
        self,
        dataset_path: str,
        split: str = "train",
        image_size: int = 256,
        task: str = "all",
        use_grayscale: bool = True,
        noise_std: float = 0.05,
        random_erasing_prob: float = 0.30,
    ):
        self.dataset_path = Path(dataset_path)
        self.split = split
        self.image_size = image_size
        self.task = task
        self.use_grayscale = use_grayscale
        self.noise_std = noise_std
        self.random_erasing_prob = random_erasing_prob

        manifest_path = self.dataset_path / "manifest.csv"
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        required_columns = {"relative_path", "split", "task", "is_mask"}
        missing_columns = required_columns.difference(manifest.columns)
        if missing_columns:
            raise ValueError(
                "manifest.csv is missing required column(s): "
                + ", ".join(sorted(missing_columns))
            )

        manifest["is_mask"] = to_bool_series(manifest["is_mask"])

        df = manifest[(manifest["split"] == split) & (~manifest["is_mask"])]
        if task != "all":
            df = df[df["task"] == task]

        if df.empty:
            available_splits = sorted(manifest["split"].dropna().unique().tolist())
            available_tasks = sorted(manifest["task"].dropna().unique().tolist())
            raise ValueError(
                f"No image rows found for split='{split}', task='{task}'.\n"
                f"Available splits: {available_splits}\n"
                f"Available tasks: {available_tasks}"
            )

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, relative_path: str) -> torch.Tensor:
        relative_path = str(relative_path).replace("\\", "/")
        image_path = self.dataset_path / relative_path

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        mode = "L" if self.use_grayscale else "RGB"
        image = Image.open(image_path).convert(mode)
        image = image.resize((self.image_size, self.image_size), Image.BILINEAR)

        arr = np.asarray(image).astype(np.float32) / 255.0

        if self.use_grayscale:
            arr = arr[None, :, :]  # [1, H, W]
        else:
            arr = np.transpose(arr, (2, 0, 1))  # [3, H, W]

        return torch.from_numpy(arr)

    def _corrupt(self, clean: torch.Tensor) -> torch.Tensor:
        noisy = clean.clone()

        if self.noise_std > 0:
            noisy = noisy + torch.randn_like(noisy) * self.noise_std

        if random.random() < self.random_erasing_prob:
            _, h, w = noisy.shape
            erase_h = random.randint(max(8, h // 12), max(9, h // 4))
            erase_w = random.randint(max(8, w // 12), max(9, w // 4))
            top = random.randint(0, h - erase_h)
            left = random.randint(0, w - erase_w)
            noisy[:, top : top + erase_h, left : left + erase_w] = 0.0

        return torch.clamp(noisy, 0.0, 1.0)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]
        clean = self._load_image(row["relative_path"])
        corrupted = self._corrupt(clean)
        return corrupted, clean


class ConvBlock(nn.Module):
    """Two Conv-BN-ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetAutoencoder2D(nn.Module):
    """
    U-Net-style denoising autoencoder for BRISC MRI images.

    Compared with a plain bottleneck autoencoder, skip connections preserve
    fine spatial details and usually produce sharper reconstructions.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = ConvBlock(c * 8, c * 16)

        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 16, c * 8, kernel_size=3, padding=1),
        )
        self.dec4 = ConvBlock(c * 16, c * 8)

        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 8, c * 4, kernel_size=3, padding=1),
        )
        self.dec3 = ConvBlock(c * 8, c * 4)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 4, c * 2, kernel_size=3, padding=1),
        )
        self.dec2 = ConvBlock(c * 4, c * 2)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1),
        )
        self.dec1 = ConvBlock(c * 2, c)

        self.out_conv = nn.Sequential(
            nn.Conv2d(c, in_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _match_size(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != ref.shape[-2:]:
            x = F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        d4 = self._match_size(d4, e4)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        d3 = self._match_size(d3, e3)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self._match_size(d2, e2)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = self._match_size(d1, e1)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.out_conv(d1)
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    l1 = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    return l1 + 0.25 * mse


def save_reconstruction_grid(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: Path,
    max_items: int = 6,
) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    corrupted, clean = next(iter(loader))
    corrupted = corrupted[:max_items].to(device)
    clean = clean[:max_items].to(device)

    with torch.no_grad():
        reconstructed = model(corrupted)

    corrupted = corrupted.cpu().numpy()
    clean = clean.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()

    rows = min(max_items, corrupted.shape[0])
    fig, axes = plt.subplots(rows, 3, figsize=(9, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        axes[i, 0].imshow(corrupted[i, 0], cmap="gray")
        axes[i, 0].set_title("Corrupted input")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(reconstructed[i, 0], cmap="gray")
        axes[i, 1].set_title("Reconstruction")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(clean[i, 0], cmap="gray")
        axes[i, 2].set_title("Clean target")
        axes[i, 2].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_one_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    total_items = 0

    progress = tqdm(loader, desc="train" if train else "val", leave=False)
    for corrupted, clean in progress:
        corrupted = corrupted.to(device, non_blocking=True)
        clean = clean.to(device, non_blocking=True)

        with torch.set_grad_enabled(train):
            pred = model(corrupted)
            loss = reconstruction_loss(pred, clean)

            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = corrupted.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        progress.set_postfix(loss=f"{loss.item():.5f}")

    return total_loss / max(total_items, 1)


def resolve_output_dir(output_dir_arg: str | None) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg)

    brisc_dir = Path(__file__).resolve().parent.parent
    return brisc_dir / "outputs" / "unet_autoencoder_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="BRISC U-Net-style 2D denoising autoencoder training")

    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=(
            "Path to the BRISC dataset root folder that contains manifest.csv. "
            f"Default: {DEFAULT_DATASET_PATH}"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output folder. Default: BRISC/outputs/unet_autoencoder_v1",
    )
    parser.add_argument("--task", type=str, default="all", choices=["all", "classification", "segmentation"])
    parser.add_argument("--train_split", type=str, default="train")
    parser.add_argument("--val_split", type=str, default="test")
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--noise_std", type=float, default=0.05)
    parser.add_argument("--random_erasing_prob", type=float, default=0.30)
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
    print("=== BRISC U-Net Denoising Autoencoder Training ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Using device: {device}")

    train_ds = BRISCAutoencoderDataset(
        dataset_path=str(dataset_path),
        split=args.train_split,
        image_size=args.image_size,
        task=args.task,
        use_grayscale=True,
        noise_std=args.noise_std,
        random_erasing_prob=args.random_erasing_prob,
    )
    val_ds = BRISCAutoencoderDataset(
        dataset_path=str(dataset_path),
        split=args.val_split,
        image_size=args.image_size,
        task=args.task,
        use_grayscale=True,
        noise_std=args.noise_std,
        random_erasing_prob=args.random_erasing_prob,
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

    model = UNetAutoencoder2D(in_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=3,
        factor=0.5,
    )

    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds)}")
    print(f"Task: {args.task}")
    print(f"Image size: {args.image_size}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Base channels: {args.base_channels}")

    config = vars(args).copy()
    config["dataset_path"] = str(dataset_path)
    config["output_dir"] = str(output_dir.resolve())
    config["device"] = str(device)
    config["model"] = "UNetAutoencoder2D"
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_one_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train={train_loss:.5f} | val={val_loss:.5f} | lr={current_lr:.6f}"
        )

        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "best_autoencoder.pt")
            save_reconstruction_grid(
                model,
                val_loader,
                device,
                output_dir / "best_reconstruction_examples.png",
            )
            print(f"  Saved best checkpoint. best_val={best_val:.5f}")

    torch.save(model.state_dict(), output_dir / "last_autoencoder_state_dict.pt")
    print("=== Training completed ===")
    print(f"Best validation loss: {best_val:.5f}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
