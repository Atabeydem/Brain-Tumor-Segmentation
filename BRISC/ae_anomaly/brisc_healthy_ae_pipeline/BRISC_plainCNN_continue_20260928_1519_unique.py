import argparse
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from train_eval_brisc_plain_cnn_autoencoder_v2 import (
    HealthyImageDataset,
    PlainCNNAutoencoder,
    seed_everything,
    run_epoch,
    evaluate_dataset,
    summarize_rows,
    save_eval,
)


def save_ckpt(path, model, optimizer, scheduler, epoch, best_val_mse,
              image_size, base_channels, source_checkpoint, args):
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "best_val_mse": best_val_mse,
        "image_size": image_size,
        "base_channels": base_channels,
        "model_name": "PlainCNNAutoencoder",
        "source_checkpoint": str(source_checkpoint),
        "finetune_args": vars(args),
    }, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scheduler_patience", type=int, default=20)
    ap.add_argument("--scheduler_factor", type=float, default=0.5)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--eval_batch_size", type=int, default=8)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_checkpoint = Path(args.checkpoint)
    ckpt = torch.load(source_checkpoint, map_location=device)
    image_size = int(ckpt.get("image_size", 256))
    base_channels = int(ckpt.get("base_channels", 64))

    model = PlainCNNAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    root = Path(args.dataset_root)
    train_dir = root / "classification_task" / "train" / "no_tumor"
    test_dir = root / "classification_task" / "test" / "no_tumor"

    full_train = HealthyImageDataset(train_dir, image_size)
    val_len = max(1, int(round(len(full_train) * args.val_fraction)))
    train_len = len(full_train) - val_len
    gen = torch.Generator().manual_seed(args.seed)
    train_ds, val_ds = random_split(full_train, [train_len, val_len], generator=gen)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.scheduler_factor,
        patience=args.scheduler_patience, min_lr=args.min_lr
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(args.amp and device.type == "cuda")
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "config.json").write_text(json.dumps({
        **vars(args),
        "device": str(device),
        "image_size": image_size,
        "base_channels": base_channels,
        "source_checkpoint": str(source_checkpoint),
        "train_images": train_len,
        "val_images": val_len,
        "architecture": "Same Plain CNN v2; only continued low-LR fine-tuning",
        "checkpoint_criterion": "lowest validation MSE including source baseline"
    }, indent=2), encoding="utf-8")

    print(f"Device: {device}")
    print(f"Source checkpoint: {source_checkpoint}")
    print(f"Image size: {image_size} | Base channels: {base_channels}")
    print(f"Train: {train_len} | Val: {val_len}")
    print(f"Epochs: {args.epochs} | LR: {args.lr} | AMP: {args.amp and device.type == 'cuda'}")

    baseline_val_mse, baseline_val_psnr = run_epoch(
        model, val_loader, device, optimizer=None, scaler=None, use_amp=args.amp
    )
    print(
        f"Epoch 000/{args.epochs} | SOURCE BASELINE | "
        f"val_mse={baseline_val_mse:.8e} | val_psnr={baseline_val_psnr:.2f} dB"
    )

    best_val_mse = baseline_val_mse
    best_path = out_dir / "best_autoencoder_finetuned.pt"
    last_path = out_dir / "last_autoencoder_finetuned.pt"
    history_path = out_dir / "history.csv"

    save_ckpt(best_path, model, optimizer, scheduler, 0, best_val_mse,
              image_size, base_channels, source_checkpoint, args)

    with history_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "lr", "train_mse", "train_psnr_db",
                    "val_mse", "val_psnr_db", "best_val_mse"])
        w.writerow([0, args.lr, "", "", baseline_val_mse,
                    baseline_val_psnr, best_val_mse])
        f.flush()

        for epoch in range(1, args.epochs + 1):
            train_mse, train_psnr = run_epoch(
                model, train_loader, device,
                optimizer=optimizer, scaler=scaler, use_amp=args.amp
            )
            val_mse, val_psnr = run_epoch(
                model, val_loader, device,
                optimizer=None, scaler=None, use_amp=args.amp
            )

            scheduler.step(val_mse)
            lr = optimizer.param_groups[0]["lr"]

            marker = ""
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                save_ckpt(best_path, model, optimizer, scheduler, epoch,
                          best_val_mse, image_size, base_channels,
                          source_checkpoint, args)
                marker = "  <-- BEST"

            w.writerow([epoch, lr, train_mse, train_psnr,
                        val_mse, val_psnr, best_val_mse])
            f.flush()

            print(
                f"Epoch {epoch:03d}/{args.epochs} | lr={lr:.2e} | "
                f"train_mse={train_mse:.8e} | train_psnr={train_psnr:.2f} dB | "
                f"val_mse={val_mse:.8e} | val_psnr={val_psnr:.2f} dB{marker}"
            )

    save_ckpt(last_path, model, optimizer, scheduler, args.epochs,
              best_val_mse, image_size, base_channels,
              source_checkpoint, args)

    print("\nFine-tuning complete.")
    print(f"Best checkpoint: {best_path}")
    print(f"Best validation MSE: {best_val_mse:.8e}")

    if args.skip_eval:
        return

    best_ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    model.eval()

    test_ds = HealthyImageDataset(test_dir, image_size)
    test_rows = evaluate_dataset(
        model, test_ds, device, args.eval_batch_size,
        args.workers, args.amp, max_images=None
    )
    test_summary = summarize_rows(test_rows)
    test_summary.update({
        "split": "classification_task/test/no_tumor",
        "checkpoint": str(best_path),
        "source_checkpoint": str(source_checkpoint),
        "image_size": image_size,
        "base_channels": base_channels,
    })
    test_paths = save_eval(
        test_rows, test_summary, out_dir / "eval_test140",
        "plain_cnn_finetuned_test140"
    )

    train1000_rows = evaluate_dataset(
        model, full_train, device, args.eval_batch_size,
        args.workers, args.amp, max_images=1000
    )
    train1000_summary = summarize_rows(train1000_rows)
    train1000_summary.update({
        "split": "classification_task/train/no_tumor",
        "checkpoint": str(best_path),
        "source_checkpoint": str(source_checkpoint),
        "max_images": 1000,
        "image_size": image_size,
        "base_channels": base_channels,
        "note_generalization": "Consistency check; not an independent generalization test."
    })
    train1000_paths = save_eval(
        train1000_rows, train1000_summary, out_dir / "eval_train1000",
        "plain_cnn_finetuned_train1000"
    )

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
