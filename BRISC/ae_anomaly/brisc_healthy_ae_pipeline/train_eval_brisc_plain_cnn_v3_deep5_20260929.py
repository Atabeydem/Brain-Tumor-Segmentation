import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from train_eval_brisc_plain_cnn_autoencoder_v2 import (
    HealthyImageDataset,
    seed_everything,
    run_epoch,
    evaluate_dataset,
    summarize_rows,
    save_eval,
)

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class PlainCNNDeep5Autoencoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        b = base_channels
        c1, c2, c3, c4, c5 = b, b*2, b*4, b*8, b*8

        self.enc1 = ConvBlock(1, c1)
        self.down1 = nn.Conv2d(c1, c1, 4, 2, 1)
        self.enc2 = ConvBlock(c1, c2)
        self.down2 = nn.Conv2d(c2, c2, 4, 2, 1)
        self.enc3 = ConvBlock(c2, c3)
        self.down3 = nn.Conv2d(c3, c3, 4, 2, 1)
        self.enc4 = ConvBlock(c3, c4)
        self.down4 = nn.Conv2d(c4, c4, 4, 2, 1)
        self.enc5 = ConvBlock(c4, c5)
        self.down5 = nn.Conv2d(c5, c5, 4, 2, 1)

        self.bottleneck = ConvBlock(c5, c5)

        self.up5 = nn.ConvTranspose2d(c5, c5, 4, 2, 1)
        self.dec5 = ConvBlock(c5, c5)
        self.up4 = nn.ConvTranspose2d(c5, c3, 4, 2, 1)
        self.dec4 = ConvBlock(c3, c3)
        self.up3 = nn.ConvTranspose2d(c3, c2, 4, 2, 1)
        self.dec3 = ConvBlock(c2, c2)
        self.up2 = nn.ConvTranspose2d(c2, c1, 4, 2, 1)
        self.dec2 = ConvBlock(c1, c1)

        c_out = max(b // 2, 16)
        self.up1 = nn.ConvTranspose2d(c1, c_out, 4, 2, 1)
        self.dec1 = ConvBlock(c_out, c_out)
        self.out = nn.Conv2d(c_out, 1, 1)

    def forward(self, x):
        x = self.enc1(x); x = torch.relu(self.down1(x))
        x = self.enc2(x); x = torch.relu(self.down2(x))
        x = self.enc3(x); x = torch.relu(self.down3(x))
        x = self.enc4(x); x = torch.relu(self.down4(x))
        x = self.enc5(x); x = torch.relu(self.down5(x))
        x = self.bottleneck(x)
        x = torch.relu(self.up5(x)); x = self.dec5(x)
        x = torch.relu(self.up4(x)); x = self.dec4(x)
        x = torch.relu(self.up3(x)); x = self.dec3(x)
        x = torch.relu(self.up2(x)); x = self.dec2(x)
        x = torch.relu(self.up1(x)); x = self.dec1(x)
        return self.out(x)

def save_checkpoint(path, model, optimizer, scheduler, epoch, args, best_val_mse):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "model_name": "PlainCNNDeep5Autoencoder",
        "architecture_version": "plain_cnn_v3_deep5",
        "args": vars(args),
    }, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--image_size", type=int, default=256)
    ap.add_argument("--base_channels", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler_patience", type=int, default=20)
    ap.add_argument("--scheduler_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval_batch_size", type=int, default=4)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    root = Path(args.dataset_root)
    train_dir = root / "classification_task" / "train" / "no_tumor"
    test_dir = root / "classification_task" / "test" / "no_tumor"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_train = HealthyImageDataset(train_dir, args.image_size)
    val_len = max(1, int(round(len(full_train) * args.val_fraction)))
    train_len = len(full_train) - val_len
    g = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=g)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=(device.type=="cuda"))

    model = PlainCNNDeep5Autoencoder(args.base_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.min_lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type=="cuda"))

    (out_dir / "config.json").write_text(json.dumps({
        **vars(args),
        "device": str(device),
        "train_images": train_len,
        "val_images": val_len,
        "parameter_count": n_params,
        "architecture": "Plain CNN v3 Deep-5, no skip, no BN, linear output, 5 downs"
    }, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Train: {train_len} | Val: {val_len}")
    print(f"Params: {n_params:,}")
    print(f"Image size: {args.image_size} | Base: {args.base_channels}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | Batch: {args.batch_size}")
    print(f"AMP: {args.amp and device.type=='cuda'}")
    print(f"Output: {out_dir}")

    best_path = out_dir / "best_autoencoder.pt"
    last_path = out_dir / "last_autoencoder.pt"
    history_path = out_dir / "history.csv"
    best_val_mse = float("inf")

    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch","lr","train_mse","train_psnr_db","val_mse","val_psnr_db","best_val_mse"])
        for epoch in range(1, args.epochs + 1):
            tr_mse, tr_psnr = run_epoch(model, train_loader, device, optimizer=optimizer,
                                        scaler=scaler, use_amp=args.amp)
            va_mse, va_psnr = run_epoch(model, val_loader, device, optimizer=None,
                                        scaler=None, use_amp=args.amp)
            scheduler.step(va_mse)
            lr = optimizer.param_groups[0]["lr"]
            marker = ""
            if va_mse < best_val_mse:
                best_val_mse = va_mse
                save_checkpoint(best_path, model, optimizer, scheduler, epoch, args, best_val_mse)
                marker = "  <-- BEST"
            w.writerow([epoch, lr, tr_mse, tr_psnr, va_mse, va_psnr, best_val_mse]); f.flush()
            print(f"Epoch {epoch:03d}/{args.epochs} | lr={lr:.2e} | "
                  f"train_mse={tr_mse:.8e} | train_psnr={tr_psnr:.2f} dB | "
                  f"val_mse={va_mse:.8e} | val_psnr={va_psnr:.2f} dB{marker}")

    save_checkpoint(last_path, model, optimizer, scheduler, args.epochs, args, best_val_mse)
    print("\nTraining complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best validation MSE: {best_val_mse:.8e}")

    if args.skip_eval:
        return

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()

    test_ds = HealthyImageDataset(test_dir, args.image_size)
    test_rows = evaluate_dataset(model, test_ds, device, args.eval_batch_size,
                                 args.workers, args.amp, max_images=None)
    test_summary = summarize_rows(test_rows)
    test_summary.update({
        "split": "classification_task/test/no_tumor",
        "checkpoint": str(best_path),
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "architecture_version": "plain_cnn_v3_deep5",
        "parameter_count": n_params,
    })
    test_paths = save_eval(test_rows, test_summary, out_dir/"eval_test140", "plain_cnn_v3_test140")

    train1000_rows = evaluate_dataset(model, full_train, device, args.eval_batch_size,
                                      args.workers, args.amp, max_images=1000)
    train1000_summary = summarize_rows(train1000_rows)
    train1000_summary.update({
        "split": "classification_task/train/no_tumor",
        "checkpoint": str(best_path),
        "max_images": 1000,
        "image_size": args.image_size,
        "base_channels": args.base_channels,
        "architecture_version": "plain_cnn_v3_deep5",
        "parameter_count": n_params,
        "note_generalization": "Consistency check; not independent generalization."
    })
    train1000_paths = save_eval(train1000_rows, train1000_summary,
                                out_dir/"eval_train1000", "plain_cnn_v3_train1000")

    print("\n=== INDEPENDENT HEALTHY TEST (140) ===")
    print(json.dumps(test_summary, indent=2))
    print("\n=== HEALTHY CONSISTENCY CHECK (up to 1000) ===")
    print(json.dumps(train1000_summary, indent=2))
    print("\nSaved:")
    for p in [*test_paths, *train1000_paths]:
        if p is not None:
            print(p)

if __name__ == "__main__":
    main()
