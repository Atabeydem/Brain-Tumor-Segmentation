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
    load_checkpoint_model,
    psnr_from_mse,
    save_contact_sheet,
    tensor_to_pil,
)


class HealthyReconDataset(Dataset):
    def __init__(self, files, image_size, noise_std=0.0, mask_ratio=0.0):
        self.files = list(files)
        self.image_size = int(image_size)
        self.noise_std = float(noise_std)
        self.mask_ratio = float(mask_ratio)

    def __len__(self):
        return len(self.files)

    def _corrupt(self, x):
        y = x.clone()
        if self.noise_std > 0:
            y = y + torch.randn_like(y) * self.noise_std
        if self.mask_ratio > 0:
            _, h, w = y.shape
            side = max(8, int((h * w * self.mask_ratio) ** 0.5))
            side = min(side, h, w)
            top = random.randint(0, h - side)
            left = random.randint(0, w - side)
            y[:, top:top + side, left:left + side] = 0.0
        return torch.clamp(y, 0.0, 1.0)

    def __getitem__(self, idx):
        clean = load_grayscale_tensor(self.files[idx], self.image_size, nearest=False)
        inp = self._corrupt(clean)
        return inp, clean, str(self.files[idx])


def evaluate(model, loader, device):
    model.eval()
    criterion = nn.MSELoss(reduction="mean")
    losses = []
    psnrs = []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            recon = model(x)
            loss = criterion(recon, y)
            losses.append(float(loss.item()))
            mse_per = ((recon - y) ** 2).flatten(1).mean(dim=1).detach().cpu().numpy()
            psnrs.extend([psnr_from_mse(float(m)) for m in mse_per])
    return {
        "mse_loss": float(np.mean(losses)) if losses else float("nan"),
        "mean_psnr": float(np.mean(psnrs)) if psnrs else float("nan"),
    }

def freeze_batchnorm_layers(model):
    """
    Keep BatchNorm layers fixed during fine-tuning.
    This prevents running mean/variance from drifting with small batch sizes.
    """
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--lr", type=float, default=0.00002)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument(
    "--freeze_bn",
    action="store_true",
    help="Freeze BatchNorm layers during fine-tuning."
)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, old_config, image_size = load_checkpoint_model(Path(args.checkpoint), device)

    if args.freeze_bn:
        freeze_batchnorm_layers(model)
        print("BatchNorm layers are frozen during fine-tuning.")

    root = Path(args.dataset_root)
    healthy_train_dir = root / "classification_task" / "train" / "no_tumor"
    files = list_images(healthy_train_dir)
    if not files:
        raise RuntimeError(f"No images found: {healthy_train_dir}")

    random.Random(args.seed).shuffle(files)
    n_val = max(1, int(len(files) * args.val_split))
    val_files = files[:n_val]
    train_files = files[n_val:]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = dict(old_config)
    config.update({
        "finetuned_from": str(args.checkpoint),
        "output_dir": str(args.output_dir),
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "val_split": args.val_split,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "noise_std": args.noise_std,
        "mask_ratio": args.mask_ratio,
        "freeze_bn": args.freeze_bn,
        "healthy_train_dir": str(healthy_train_dir),
        "n_train": len(train_files),
        "n_val": len(val_files),
        "image_size": image_size,
    })
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    train_ds = HealthyReconDataset(train_files, image_size, noise_std=args.noise_std, mask_ratio=args.mask_ratio)
    val_ds = HealthyReconDataset(val_files, image_size, noise_std=0.0, mask_ratio=0.0)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=args.lr
    )
    criterion = nn.MSELoss()

    print("Device:", device)
    print("Finetuning from:", args.checkpoint)
    print("Image size:", image_size)
    print("Healthy train:", len(train_files), "Healthy val:", len(val_files))
    print("Output:", out_dir)
    print("noise_std:", args.noise_std, "mask_ratio:", args.mask_ratio)

    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_bn:
            freeze_batchnorm_layers(model)
        train_losses = []
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            recon = model(x)
            loss = criterion(recon, y)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.item()))

        val_stats = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_mse_loss": float(np.mean(train_losses)),
            "val_mse_loss": val_stats["mse_loss"],
            "val_mean_psnr": val_stats["mean_psnr"],
        }
        history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_mse={row['train_mse_loss']:.8f} | val_mse={row['val_mse_loss']:.8f} | val_psnr={row['val_mean_psnr']:.2f} dB")

        if row["val_mse_loss"] < best_val:
            best_val = row["val_mse_loss"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "best_epoch": epoch,
                "best_val_mse_loss": best_val,
                "best_val_mean_psnr": row["val_mean_psnr"],
            }, out_dir / "best_autoencoder.pt")

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            model.eval()
            sample_rows = []
            with torch.no_grad():
                for i in range(min(6, len(val_ds))):
                    x, y, path = val_ds[i]
                    recon = model(x.unsqueeze(0).to(device)).cpu().squeeze(0)
                    sample_rows.append((Path(path).name, [tensor_to_pil(x), tensor_to_pil(recon), tensor_to_pil(y)]))
            save_contact_sheet(sample_rows, ["Input", "Reconstructed", "Target"], out_dir / f"recon_samples_epoch_{epoch:03d}.png")

    with open(out_dir / "training_history.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    print("\nDone.")
    print("Best checkpoint:", out_dir / "best_autoencoder.pt")


if __name__ == "__main__":
    main()
