"""
Train a 2D denoising autoencoder on the BRISC 2025 dataset.

Expected dataset root:
    brisc2025/
        manifest.csv
        classification_task/...
        segmentation_task/...

Example:
    python BRISC/autoencoder/train_brisc_autoencoder.py \
        --dataset_path "D:/Downloads/brisc_Dataset2025/brisc2025" \
        --output_dir "BRISC/outputs/autoencoder_v1" \
        --image_size 256 \
        --batch_size 16 \
        --epochs 30
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


class BRISCAutoencoderDataset(Dataset):
    """Loads BRISC images only. Masks are excluded.

    The model receives a corrupted/noisy image and tries to reconstruct
    the clean image. This is a denoising autoencoder setup.
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
        if not manifest_path.exists():
            raise FileNotFoundError(f"manifest.csv not found: {manifest_path}")

        manifest = pd.read_csv(manifest_path)
        manifest["is_mask"] = manifest["is_mask"].astype(bool)

        df = manifest[(manifest["split"] == split) & (~manifest["is_mask"])]
        if task != "all":
            df = df[df["task"] == task]

        if df.empty:
            raise ValueError(f"No image rows found for split={split}, task={task}")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, relative_path: str) -> torch.Tensor:
        image_path = self.dataset_path / str(relative_path).replace("\\", "/")
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

        # Random rectangle masking; useful for learning robust reconstruction.
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


class ConvAutoencoder2D(nn.Module):
    """Small 2D convolutional autoencoder for 256x256 BRISC images."""

    def __init__(self, in_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),

            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),

            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c, in_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        out = self.decoder(z)
        # Safety for image sizes that are not perfectly divisible by 16.
        if out.shape[-2:] != x.shape[-2:]:
            out = F.interpolate(out, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # L1 gives sharper reconstruction than pure MSE in many image tasks.
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

    rows = max_items
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
        progress.set_postfix(loss=loss.item())

    return total_loss / max(total_items, 1)


def parse_args():
    parser = argparse.ArgumentParser(description="BRISC 2D denoising autoencoder training")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to brisc2025 root folder")
    parser.add_argument("--output_dir", type=str, default="BRISC/outputs/autoencoder_v1")
    parser.add_argument("--task", type=str, default="all", choices=["all", "classification", "segmentation"])
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = BRISCAutoencoderDataset(
        dataset_path=args.dataset_path,
        split="train",
        image_size=args.image_size,
        task=args.task,
        use_grayscale=True,
        noise_std=args.noise_std,
        random_erasing_prob=args.random_erasing_prob,
    )
    val_ds = BRISCAutoencoderDataset(
        dataset_path=args.dataset_path,
        split="test",
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

    model = ConvAutoencoder2D(in_channels=1, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)

    print(f"Train images: {len(train_ds)}")
    print(f"Val images: {len(val_ds)}")

    config = vars(args)
    config["device"] = str(device)
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_one_epoch(model, val_loader, optimizer, device, train=False)
        scheduler.step(val_loss)

        record = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(record)
        print(f"Epoch {epoch:03d}/{args.epochs} | train={train_loss:.5f} | val={val_loss:.5f}")

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
    print(f"Training completed. Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
