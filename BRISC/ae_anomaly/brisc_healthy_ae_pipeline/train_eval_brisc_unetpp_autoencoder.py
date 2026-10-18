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
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision import transforms

try:
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:
    skimage_ssim = None

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


class HealthyImageDataset(Dataset):
    def __init__(self, image_dir: Path, image_size=256):
        self.image_dir = Path(image_dir)
        self.files = sorted(
            p for p in self.image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        )
        if not self.files:
            raise FileNotFoundError(f"No images found in: {self.image_dir}")

        self.transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        with Image.open(path) as im:
            im = im.convert("L")
            x = self.transform(im)
        return x, str(path)


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
    """
    Compact U-Net++ / Nested U-Net autoencoder.

    - 4 resolution levels (0..3)
    - nested dense skip pathways
    - no BatchNorm
    - linear output head (no Sigmoid)
    - input/output: 1-channel grayscale
    """
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels

        self.pool = nn.MaxPool2d(2, 2)

        # Encoder backbone
        self.conv0_0 = ConvBlock(1, b)
        self.conv1_0 = ConvBlock(b, b * 2)
        self.conv2_0 = ConvBlock(b * 2, b * 4)
        self.conv3_0 = ConvBlock(b * 4, b * 8)

        # Nested skip pathway: j=1
        self.conv0_1 = ConvBlock(b + b * 2, b)
        self.conv1_1 = ConvBlock(b * 2 + b * 4, b * 2)
        self.conv2_1 = ConvBlock(b * 4 + b * 8, b * 4)

        # j=2
        self.conv0_2 = ConvBlock(b * 2 + b * 2, b)
        self.conv1_2 = ConvBlock(b * 4 + b * 4, b * 2)

        # j=3
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
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        # Nested decoder
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_to(x3_0, x2_0)], dim=1))

        x0_2 = self.conv0_2(
            torch.cat([x0_0, x0_1, self.up_to(x1_1, x0_0)], dim=1)
        )
        x1_2 = self.conv1_2(
            torch.cat([x1_0, x1_1, self.up_to(x2_1, x1_0)], dim=1)
        )

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up_to(x1_2, x0_0)], dim=1)
        )

        return self.out(x0_3)


def batch_per_image_mse(pred, target):
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def mse_to_psnr(mse_tensor):
    return -10.0 * torch.log10(torch.clamp(mse_tensor, min=1e-12))


def run_epoch(model, loader, device, optimizer=None, scaler=None, use_amp=False):
    training = optimizer is not None
    model.train(training)

    loss_sum = 0.0
    psnr_sum = 0.0
    n = 0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        amp_enabled = use_amp and device.type == "cuda"
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            pred = model(x)
            per_img_mse = batch_per_image_mse(pred, x)
            loss = per_img_mse.mean()

        if training:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        loss_sum += loss.detach().item() * bs

        # Metrics are measured on the valid image range.
        with torch.no_grad():
            pred_metric = torch.clamp(pred.detach(), 0.0, 1.0)
            metric_mse = batch_per_image_mse(pred_metric, x.detach())
            psnr_sum += mse_to_psnr(metric_mse).sum().item()

        n += bs

    return loss_sum / max(n, 1), psnr_sum / max(n, 1)


@torch.no_grad()
def evaluate_dataset(
    model, dataset, device, batch_size, workers, use_amp, max_images=None
):
    if max_images is not None and len(dataset) > max_images:
        dataset = Subset(dataset, list(range(max_images)))

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=(device.type == "cuda"),
    )

    rows = []
    model.eval()
    amp_enabled = use_amp and device.type == "cuda"

    for x, paths in loader:
        x = x.to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=amp_enabled,
        ):
            pred = model(x)

        pred = torch.clamp(pred, 0.0, 1.0)

        x_cpu = x.float().cpu().numpy()
        pred_cpu = pred.float().cpu().numpy()

        for i in range(x_cpu.shape[0]):
            target = x_cpu[i, 0]
            recon = pred_cpu[i, 0]
            diff = recon - target

            mse = float(np.mean(diff ** 2))
            mae = float(np.mean(np.abs(diff)))
            psnr = float(-10.0 * math.log10(max(mse, 1e-12)))

            ssim = None
            if skimage_ssim is not None:
                ssim = float(
                    skimage_ssim(target, recon, data_range=1.0)
                )

            rows.append({
                "path": paths[i],
                "mse": mse,
                "mae": mae,
                "psnr_db": psnr,
                "ssim": ssim,
                "_target": target,
                "_recon": recon,
            })

    return rows


def summarize_rows(rows):
    mses = np.array([r["mse"] for r in rows], dtype=np.float64)
    maes = np.array([r["mae"] for r in rows], dtype=np.float64)
    psnrs = np.array([r["psnr_db"] for r in rows], dtype=np.float64)
    ssims = np.array(
        [r["ssim"] for r in rows if r["ssim"] is not None],
        dtype=np.float64,
    )

    return {
        "n_images": int(len(rows)),
        "mean_mse": float(mses.mean()),
        "std_mse": float(mses.std()),
        "mean_mae": float(maes.mean()),
        "std_mae": float(maes.std()),
        "mean_psnr_db": float(psnrs.mean()),
        "std_psnr_db": float(psnrs.std()),
        "min_psnr_db": float(psnrs.min()),
        "max_psnr_db": float(psnrs.max()),
        "mean_ssim": float(ssims.mean()) if len(ssims) else None,
        "std_ssim": float(ssims.std()) if len(ssims) else None,
    }


def save_eval(rows, summary, out_dir: Path, prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / f"{prefix}_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["path", "mse", "mae", "psnr_db", "ssim"],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(
                {k: r[k] for k in ["path", "mse", "mae", "psnr_db", "ssim"]}
            )

    json_path = out_dir / f"{prefix}_summary.json"
    json_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    fig_path = None
    if plt is not None and len(rows) >= 3:
        sorted_rows = sorted(rows, key=lambda r: r["psnr_db"])
        n = len(sorted_rows)
        candidate_idxs = [
            0,
            max(1, n // 10),
            n // 2,
            min(n - 1, n // 2 + 1),
            max(0, n - 2),
            n - 1,
        ]
        seen = set()
        candidate_idxs = [
            i for i in candidate_idxs
            if not (i in seen or seen.add(i))
        ]
        selected = [sorted_rows[i] for i in candidate_idxs[:6]]

        fig, axes = plt.subplots(
            len(selected), 2, figsize=(7, 3 * len(selected))
        )
        if len(selected) == 1:
            axes = np.array([axes])

        for i, r in enumerate(selected):
            axes[i, 0].imshow(
                r["_target"], cmap="gray", vmin=0, vmax=1
            )
            axes[i, 0].set_title("Original")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(
                r["_recon"], cmap="gray", vmin=0, vmax=1
            )
            axes[i, 1].set_title(
                f"Reconstructed | PSNR {r['psnr_db']:.2f} dB"
            )
            axes[i, 1].axis("off")

        fig.tight_layout()
        fig_path = out_dir / f"{prefix}_reconstruction_examples.png"
        fig.savefig(fig_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    return csv_path, json_path, fig_path


def save_checkpoint(
    path, model, optimizer, scheduler, epoch, args, best_val_mse
):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": (
            optimizer.state_dict() if optimizer is not None else None
        ),
        "scheduler_state_dict": (
            scheduler.state_dict() if scheduler is not None else None
        ),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "model_name": "UNetPlusPlusAutoencoder",
        "args": vars(args),
    }, path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train + evaluate a compact U-Net++ autoencoder "
            "on BRISC healthy/no_tumor images."
        )
    )
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--base_channels", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scheduler_patience", type=int, default=15)
    parser.add_argument("--scheduler_factor", type=float, default=0.5)
    parser.add_argument("--min_lr", type=float, default=1e-7)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)

    dataset_root = Path(args.dataset_root)
    train_dir = dataset_root / "classification_task" / "train" / "no_tumor"
    test_dir = dataset_root / "classification_task" / "test" / "no_tumor"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    full_train = HealthyImageDataset(train_dir, args.image_size)

    val_len = max(
        1,
        int(round(len(full_train) * args.val_fraction))
    )
    train_len = len(full_train) - val_len

    generator = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(
        full_train,
        [train_len, val_len],
        generator=generator,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    model = UNetPlusPlusAutoencoder(
        base_channels=args.base_channels
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_lr,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(args.amp and device.type == "cuda"),
    )

    config = {
        **vars(args),
        "device": str(device),
        "train_dir": str(train_dir),
        "test_dir": str(test_dir),
        "total_healthy_train_images": len(full_train),
        "train_images": train_len,
        "val_images": val_len,
        "architecture": (
            "Compact U-Net++ / Nested U-Net; dense nested skip pathways; "
            "no BatchNorm; linear output head"
        ),
        "loss": "MSE",
        "optimizer": "Adam",
        "scheduler": "ReduceLROnPlateau(val_mse)",
        "checkpoint_criterion": "lowest validation MSE",
    }

    (out_dir / "config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    print(f"Device: {device}")
    print(f"Healthy train dir: {train_dir}")
    print(f"Healthy test dir: {test_dir}")
    print(f"Total healthy train images: {len(full_train)}")
    print(f"Train images: {train_len} | Val images: {val_len}")
    print(f"Image size: {args.image_size}")
    print(f"Base channels: {args.base_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Initial LR: {args.lr}")
    print(f"AMP: {args.amp and device.type == 'cuda'}")
    print(f"Output: {out_dir}")

    history_path = out_dir / "history.csv"
    best_path = out_dir / "best_autoencoder.pt"
    last_path = out_dir / "last_autoencoder.pt"

    best_val_mse = float("inf")

    with history_path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "lr",
            "train_mse",
            "train_psnr_db",
            "val_mse",
            "val_psnr_db",
            "best_val_mse",
        ])

        for epoch in range(1, args.epochs + 1):
            train_mse, train_psnr = run_epoch(
                model,
                train_loader,
                device,
                optimizer=optimizer,
                scaler=scaler,
                use_amp=args.amp,
            )

            val_mse, val_psnr = run_epoch(
                model,
                val_loader,
                device,
                optimizer=None,
                scaler=None,
                use_amp=args.amp,
            )

            scheduler.step(val_mse)
            current_lr = optimizer.param_groups[0]["lr"]

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    args,
                    best_val_mse,
                )
                best_marker = "  <-- BEST"
            else:
                best_marker = ""

            writer.writerow([
                epoch,
                current_lr,
                train_mse,
                train_psnr,
                val_mse,
                val_psnr,
                best_val_mse,
            ])
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | "
                f"lr={current_lr:.2e} | "
                f"train_mse={train_mse:.8e} | "
                f"train_psnr={train_psnr:.2f} dB | "
                f"val_mse={val_mse:.8e} | "
                f"val_psnr={val_psnr:.2f} dB"
                f"{best_marker}"
            )

    save_checkpoint(
        last_path,
        model,
        optimizer,
        scheduler,
        args.epochs,
        args,
        best_val_mse,
    )

    print("\nTraining complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best validation MSE: {best_val_mse:.8e}")

    if args.skip_eval:
        return

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_ds = HealthyImageDataset(
        test_dir,
        args.image_size,
    )

    test_rows = evaluate_dataset(
        model,
        test_ds,
        device,
        args.eval_batch_size,
        args.workers,
        args.amp,
        max_images=None,
    )
    test_summary = summarize_rows(test_rows)
    test_summary.update({
        "split": "classification_task/test/no_tumor",
        "checkpoint": str(best_path),
        "image_size": args.image_size,
        "base_channels": args.base_channels,
    })
    test_paths = save_eval(
        test_rows,
        test_summary,
        out_dir / "eval_test140",
        "unetpp_test140",
    )

    train1000_rows = evaluate_dataset(
        model,
        full_train,
        device,
        args.eval_batch_size,
        args.workers,
        args.amp,
        max_images=1000,
    )
    train1000_summary = summarize_rows(train1000_rows)
    train1000_summary.update({
        "split": "classification_task/train/no_tumor",
        "checkpoint": str(best_path),
        "max_images": 1000,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "note_generalization": (
            "This is a consistency check on training-pool healthy images, "
            "not an independent generalization test."
        ),
    })
    train1000_paths = save_eval(
        train1000_rows,
        train1000_summary,
        out_dir / "eval_train1000",
        "unetpp_train1000",
    )

    print("\n=== INDEPENDENT HEALTHY TEST (140) ===")
    print(json.dumps(test_summary, indent=2))

    print(
        "\n=== HEALTHY CONSISTENCY CHECK "
        "(up to 1000 train-pool images) ==="
    )
    print(json.dumps(train1000_summary, indent=2))

    print("\nSaved:")
    for p in [*test_paths, *train1000_paths]:
        if p is not None:
            print(p)


if __name__ == "__main__":
    main()
