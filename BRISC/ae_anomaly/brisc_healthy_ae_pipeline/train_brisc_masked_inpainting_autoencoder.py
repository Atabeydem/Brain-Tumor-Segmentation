import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from brisc_ae_common import (
    list_images,
    set_seed,
    load_grayscale_tensor,
    psnr_from_mse,
    save_contact_sheet,
    tensor_to_pil,
)


# =========================
# Dataset
# =========================
class BRISCMaskedInpaintingDataset(Dataset):
    def __init__(self, pairs, image_size):
        self.pairs = list(pairs)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]

        # MRI görüntüsü
        img = load_grayscale_tensor(img_path, self.image_size, nearest=False)   # [1,H,W]

        # Maske
        mask = load_grayscale_tensor(mask_path, self.image_size, nearest=True)   # [1,H,W]
        mask = (mask > 0.5).float()

        # Tümör bölgesini karart
        masked_img = img * (1.0 - mask)

        # 2 kanal input: [masked_image, mask]
        inp = torch.cat([masked_img, mask], dim=0)  # [2,H,W]

        # Target: orijinal clean MRI
        target = img

        return inp, target, mask, str(img_path), str(mask_path)


def build_pairs(images_dir, masks_dir):
    img_files = list_images(images_dir)
    mask_files = list_images(masks_dir)

    mask_map = {p.stem: p for p in mask_files}
    pairs = []

    for img_path in img_files:
        mask_path = mask_map.get(img_path.stem)
        if mask_path is not None:
            pairs.append((img_path, mask_path))

    if len(pairs) == 0:
        raise RuntimeError(f"No matched image-mask pairs found in:\n{images_dir}\n{masks_dir}")

    return pairs


# =========================
# Model
# =========================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetMaskedInpaintingAE(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=32):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(c1, c2)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ConvBlock(c2, c3)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(c3, c4)

        self.up3 = nn.ConvTranspose2d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(c3 + c3, c3)

        self.up2 = nn.ConvTranspose2d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(c2 + c2, c2)

        self.up1 = nn.ConvTranspose2d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(c1 + c1, c1)

        self.out_conv = nn.Conv2d(c1, out_channels, kernel_size=1)
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b = self.bottleneck(self.pool3(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out_conv(d1)
        out = self.out_act(out)
        return out


# =========================
# Loss / Evaluation
# =========================
def weighted_masked_mse_loss(recon, target, mask, mask_weight=20.0):
    # mask=1 olan bölgede loss'u büyüt
    weights = 1.0 + (mask_weight - 1.0) * mask
    loss = ((recon - target) ** 2) * weights
    return loss.mean()


def evaluate(model, loader, device, mask_weight):
    model.eval()

    weighted_losses = []
    full_mses = []
    psnrs = []
    mask_mses = []

    with torch.no_grad():
        for inp, target, mask, _, _ in loader:
            inp = inp.to(device)
            target = target.to(device)
            mask = mask.to(device)

            recon = model(inp)

            weighted_loss = weighted_masked_mse_loss(recon, target, mask, mask_weight=mask_weight)
            weighted_losses.append(float(weighted_loss.item()))

            mse_per = ((recon - target) ** 2).flatten(1).mean(dim=1).detach().cpu().numpy()
            full_mses.extend([float(x) for x in mse_per])
            psnrs.extend([psnr_from_mse(float(x)) for x in mse_per])

            # Sadece maskeli bölgedeki MSE
            diff2 = (recon - target) ** 2
            mask_sum = mask.flatten(1).sum(dim=1)  # [B]

            for b in range(inp.size(0)):
                if mask_sum[b] > 0:
                    mmse = diff2[b][mask[b] > 0.5].mean().item()
                    mask_mses.append(float(mmse))

    return {
        "weighted_mse_loss": float(np.mean(weighted_losses)) if weighted_losses else float("nan"),
        "full_mse_loss": float(np.mean(full_mses)) if full_mses else float("nan"),
        "mean_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
        "mask_region_mse": float(np.mean(mask_mses)) if mask_mses else float("nan"),
    }


# =========================
# Main
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mask_weight", type=float, default=20.0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.dataset_root)
    images_dir = root / "segmentation_task" / "train" / "images"
    masks_dir = root / "segmentation_task" / "train" / "masks"

    pairs = build_pairs(images_dir, masks_dir)
    random.Random(args.seed).shuffle(pairs)

    n_val = max(1, int(len(pairs) * args.val_split))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]

    train_ds = BRISCMaskedInpaintingDataset(train_pairs, args.image_size)
    val_ds = BRISCMaskedInpaintingDataset(val_pairs, args.image_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNetMaskedInpaintingAE(
        in_channels=2,
        out_channels=1,
        base_channels=args.base_channels
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = vars(args).copy()
    config.update({
        "n_total_pairs": len(pairs),
        "n_train": len(train_pairs),
        "n_val": len(val_pairs),
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "model_name": "UNetMaskedInpaintingAE",
        "input_channels": 2,
        "output_channels": 1,
    })
    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("Device:", device)
    print("Images dir:", images_dir)
    print("Masks dir:", masks_dir)
    print("Total pairs:", len(pairs))
    print("Train pairs:", len(train_pairs), "Val pairs:", len(val_pairs))
    print("Output:", out_dir)
    print("Image size:", args.image_size)
    print("Base channels:", args.base_channels)
    print("Mask weight:", args.mask_weight)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for inp, target, mask, _, _ in train_loader:
            inp = inp.to(device)
            target = target.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
            recon = model(inp)
            loss = weighted_masked_mse_loss(recon, target, mask, mask_weight=args.mask_weight)
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        val_stats = evaluate(model, val_loader, device, mask_weight=args.mask_weight)

        row = {
            "epoch": epoch,
            "train_weighted_mse_loss": float(np.mean(train_losses)) if train_losses else float("nan"),
            "val_weighted_mse_loss": val_stats["weighted_mse_loss"],
            "val_full_mse_loss": val_stats["full_mse_loss"],
            "val_mean_psnr": val_stats["mean_psnr"],
            "val_mask_region_mse": val_stats["mask_region_mse"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_wmse={row['train_weighted_mse_loss']:.8f} | "
            f"val_wmse={row['val_weighted_mse_loss']:.8f} | "
            f"val_full_mse={row['val_full_mse_loss']:.8f} | "
            f"val_psnr={row['val_mean_psnr']:.2f} dB | "
            f"val_mask_mse={row['val_mask_region_mse']:.8f}"
        )

        if row["val_weighted_mse_loss"] < best_val:
            best_val = row["val_weighted_mse_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_epoch": epoch,
                "best_val_weighted_mse_loss": row["val_weighted_mse_loss"],
                "best_val_full_mse_loss": row["val_full_mse_loss"],
                "best_val_mean_psnr": row["val_mean_psnr"],
                "best_val_mask_region_mse": row["val_mask_region_mse"],
            }, out_dir / "best_autoencoder.pt")

        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            model.eval()
            sample_rows = []

            with torch.no_grad():
                for i in range(min(6, len(val_ds))):
                    inp, target, mask, img_path, _ = val_ds[i]

                    recon = model(inp.unsqueeze(0).to(device)).cpu().squeeze(0)

                    masked_img = inp[0:1]
                    mask_img = inp[1:2]

                    sample_rows.append((
                        Path(img_path).name,
                        [
                            tensor_to_pil(masked_img),
                            tensor_to_pil(mask_img),
                            tensor_to_pil(recon),
                            tensor_to_pil(target),
                        ]
                    ))

            save_contact_sheet(
                sample_rows,
                ["Masked Input", "Mask", "Reconstructed", "Target"],
                out_dir / f"recon_samples_epoch_{epoch:03d}.png"
            )

    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nDone.")
    print("Best checkpoint:", out_dir / "best_autoencoder.pt")


if __name__ == "__main__":
    main()