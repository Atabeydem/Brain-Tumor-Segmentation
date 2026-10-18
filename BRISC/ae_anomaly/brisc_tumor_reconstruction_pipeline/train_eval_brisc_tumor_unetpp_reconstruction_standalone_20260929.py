import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

try:
    from skimage.metrics import structural_similarity as structural_similarity
except Exception:
    structural_similarity = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def list_images(folder):
    return sorted(
        p for p in Path(folder).rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


class ImageOnlyDataset(Dataset):
    def __init__(self, image_dir, image_size=256):
        self.files = list_images(image_dir)
        if not self.files:
            raise FileNotFoundError(f"No images found in {image_dir}")

        self.tf = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        with Image.open(p) as im:
            x = self.tf(im.convert("L"))
        return x, str(p)


class TumorPairDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        images = list_images(image_dir)
        masks = {p.stem: p for p in list_images(mask_dir)}
        self.pairs = [(p, masks[p.stem]) for p in images if p.stem in masks]

        if not self.pairs:
            raise FileNotFoundError("No matched tumor image-mask pairs found.")

        self.image_tf = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        with Image.open(ip) as im:
            x = self.image_tf(im.convert("L"))
        with Image.open(mp) as mm:
            mask = (self.mask_tf(mm.convert("L")) > 0.5).float()
        return x, mask, str(ip)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetPlusPlusAutoencoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = ConvBlock(1, b)
        self.conv1_0 = ConvBlock(b, b * 2)
        self.conv2_0 = ConvBlock(b * 2, b * 4)
        self.conv3_0 = ConvBlock(b * 4, b * 8)

        self.conv0_1 = ConvBlock(b + b * 2, b)
        self.conv1_1 = ConvBlock(b * 2 + b * 4, b * 2)
        self.conv2_1 = ConvBlock(b * 4 + b * 8, b * 4)

        self.conv0_2 = ConvBlock(b * 2 + b * 2, b)
        self.conv1_2 = ConvBlock(b * 4 + b * 4, b * 2)

        self.conv0_3 = ConvBlock(b * 3 + b * 2, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    @staticmethod
    def up_to(x, ref):
        return F.interpolate(
            x,
            size=ref.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_to(x3_0, x2_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_to(x2_1, x1_0)], dim=1))

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up_to(x1_2, x0_0)], dim=1)
        )
        return self.out(x0_3)


def per_image_mse(pred, target):
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def mse_to_psnr(mse):
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-12))


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    psnr_sum = 0.0
    n = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        pred = model(x)
        mse_each = per_image_mse(pred, x)
        loss = mse_each.mean()

        if training:
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_metric = torch.clamp(pred.detach(), 0.0, 1.0)
            metric_mse = per_image_mse(pred_metric, x.detach())
            psnr_sum += mse_to_psnr(metric_mse).sum().item()

        bs = x.size(0)
        loss_sum += loss.detach().item() * bs
        n += bs

    return loss_sum / max(n, 1), psnr_sum / max(n, 1)


def save_checkpoint(path, model, optimizer, scheduler, epoch, args, best_val_mse):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "model_name": "UNetPlusPlusAutoencoder",
        "experiment": "tumor_reconstruction_baseline",
        "args": vars(args),
    }, path)


@torch.no_grad()
def evaluate(model, loader, device):
    rows = []
    examples = []
    model.eval()

    for x, mask, paths in loader:
        x = x.to(device, non_blocking=True)
        pred = torch.clamp(model(x), 0.0, 1.0).cpu().numpy()
        x_np = x.cpu().numpy()
        mask_np = mask.numpy()

        for i in range(x_np.shape[0]):
            target = x_np[i, 0]
            recon = pred[i, 0]
            m = mask_np[i, 0] > 0.5
            nm = ~m

            sq = (recon - target) ** 2
            ae = np.abs(recon - target)

            full_mse = float(sq.mean())
            mask_mse = float(sq[m].mean())
            nonmask_mse = float(sq[nm].mean())

            row = {
                "image": paths[i],
                "full_mse": full_mse,
                "full_mae": float(ae.mean()),
                "full_psnr_db": float(-10.0 * math.log10(max(full_mse, 1e-12))),
                "mask_mse": mask_mse,
                "mask_mae": float(ae[m].mean()),
                "mask_psnr_db": float(-10.0 * math.log10(max(mask_mse, 1e-12))),
                "nonmask_mse": nonmask_mse,
                "nonmask_mae": float(ae[nm].mean()),
                "nonmask_psnr_db": float(-10.0 * math.log10(max(nonmask_mse, 1e-12))),
                "mask_area_ratio": float(m.mean()),
                "mask_to_nonmask_mse_ratio": float(mask_mse / max(nonmask_mse, 1e-12)),
                "ssim": None,
            }

            if structural_similarity is not None:
                row["ssim"] = float(
                    structural_similarity(target, recon, data_range=1.0)
                )

            rows.append(row)

            if len(examples) < 8:
                examples.append(
                    (target, recon, np.abs(target - recon), m, row["full_psnr_db"])
                )

    return rows, examples


def summarize(rows):
    keys = [
        "full_mse", "full_mae", "full_psnr_db",
        "mask_mse", "mask_mae", "mask_psnr_db",
        "nonmask_mse", "nonmask_mae", "nonmask_psnr_db",
        "mask_area_ratio", "mask_to_nonmask_mse_ratio",
    ]
    out = {"n_images": len(rows)}

    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[f"mean_{key}"] = float(vals.mean())
        out[f"std_{key}"] = float(vals.std())

    ssims = [r["ssim"] for r in rows if r["ssim"] is not None]
    out["mean_ssim"] = float(np.mean(ssims)) if ssims else None
    out["std_ssim"] = float(np.std(ssims)) if ssims else None

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--base_channels", type=int, default=32)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler_patience", type=int, default=15)
    ap.add_argument("--scheduler_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_img = root / "segmentation_task" / "train" / "images"
    test_img = root / "segmentation_task" / "test" / "images"
    test_mask = root / "segmentation_task" / "test" / "masks"

    full_train = ImageOnlyDataset(train_img, args.image_size)

    val_len = max(1, int(round(len(full_train) * args.val_fraction)))
    train_len = len(full_train) - val_len
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        full_train, [train_len, val_len], generator=g
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    model = UNetPlusPlusAutoencoder(args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    best_path = out / "best_autoencoder.pt"
    last_path = out / "last_autoencoder.pt"
    history_path = out / "history.csv"
    best_val_mse = float("inf")

    print(f"Device: {device}")
    print(f"Tumor train images total: {len(full_train)}")
    print(f"Train: {train_len} | Val: {val_len}")
    print(f"Target: tumor image -> same tumor image")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch", "lr",
            "train_mse", "train_psnr_db",
            "val_mse", "val_psnr_db",
            "best_val_mse",
        ])

        for epoch in range(1, args.epochs + 1):
            tr_mse, tr_psnr = run_epoch(model, train_loader, device, optimizer)
            va_mse, va_psnr = run_epoch(model, val_loader, device, None)

            scheduler.step(va_mse)
            lr = optimizer.param_groups[0]["lr"]

            marker = ""
            if va_mse < best_val_mse:
                best_val_mse = va_mse
                save_checkpoint(
                    best_path, model, optimizer, scheduler,
                    epoch, args, best_val_mse
                )
                marker = "  <-- BEST"

            w.writerow([
                epoch, lr,
                tr_mse, tr_psnr,
                va_mse, va_psnr,
                best_val_mse,
            ])
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"lr={lr:.2e} | "
                f"train_mse={tr_mse:.8e} | train_psnr={tr_psnr:.2f} dB | "
                f"val_mse={va_mse:.8e} | val_psnr={va_psnr:.2f} dB"
                f"{marker}"
            )

    save_checkpoint(
        last_path, model, optimizer, scheduler,
        args.epochs, args, best_val_mse
    )

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    test_ds = TumorPairDataset(test_img, test_mask, args.image_size)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda")
    )

    rows, examples = evaluate(model, test_loader, device)
    summary = summarize(rows)
    summary.update({
        "split": "segmentation_task/test",
        "checkpoint": str(best_path),
        "training_target": "tumor image -> same tumor image",
        "note": (
            "This is a tumor-image reconstruction baseline, not tumor removal. "
            "Full-image and GT-mask-region metrics are reported separately."
        ),
    })

    eval_dir = out / "eval_test860"
    eval_dir.mkdir(parents=True, exist_ok=True)

    metrics_csv = eval_dir / "tumor_reconstruction_test860_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_json = eval_dir / "tumor_reconstruction_test860_summary.json"
    summary_json.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    if plt is not None:
        fig, axes = plt.subplots(
            len(examples), 4,
            figsize=(12, 3 * len(examples))
        )
        if len(examples) == 1:
            axes = np.array([axes])

        for i, (target, recon, err, mask, psnr) in enumerate(examples):
            axes[i, 0].imshow(target, cmap="gray", vmin=0, vmax=1)
            axes[i, 0].set_title("Original tumor image")
            axes[i, 1].imshow(recon, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title(f"Reconstructed | {psnr:.2f} dB")
            axes[i, 2].imshow(err, cmap="gray")
            axes[i, 2].set_title("Absolute error")
            axes[i, 3].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[i, 3].set_title("GT tumor mask")

            for j in range(4):
                axes[i, j].axis("off")

        fig.tight_layout()
        fig.savefig(
            eval_dir / "tumor_reconstruction_examples.png",
            dpi=180,
            bbox_inches="tight",
        )
        plt.close(fig)

    print("\n=== INDEPENDENT TUMOR TEST (860) ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved outputs:", eval_dir)


if __name__ == "__main__":
    main()
