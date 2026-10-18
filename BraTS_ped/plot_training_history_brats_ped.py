import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def find_col(df, candidates):
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    history_path = run_dir / "training_history.csv"

    if not history_path.exists():
        raise FileNotFoundError(f"training_history.csv not found: {history_path}")

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)

    epoch_col = find_col(df, ["epoch", "Epoch"])
    train_loss_col = find_col(df, ["train_loss", "loss_train"])
    val_loss_col = find_col(df, ["val_loss", "valid_loss", "validation_loss"])
    train_dice_col = find_col(df, ["train_dice", "dice_train"])
    val_dice_col = find_col(df, ["val_dice", "valid_dice", "validation_dice"])

    if epoch_col is None:
        df["epoch"] = range(1, len(df) + 1)
        epoch_col = "epoch"

    if train_loss_col and val_loss_col:
        plt.figure(figsize=(10, 5.5))
        plt.plot(df[epoch_col], df[train_loss_col], marker="o", label="train_loss")
        plt.plot(df[epoch_col], df[val_loss_col], marker="o", label="val_loss")
        plt.title("BraTS-PED AE-pretrained U-Net Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "loss_curve.png", dpi=250)
        plt.close()

    if train_dice_col and val_dice_col:
        plt.figure(figsize=(10, 5.5))
        plt.plot(df[epoch_col], df[train_dice_col], marker="o", label="train_dice")
        plt.plot(df[epoch_col], df[val_dice_col], marker="o", label="val_dice")
        plt.title("BraTS-PED AE-pretrained U-Net Training and Validation Dice")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "dice_curve.png", dpi=250)
        plt.close()

    summary = {"epochs": int(df[epoch_col].max())}
    if val_dice_col:
        best_idx = df[val_dice_col].idxmax()
        summary["best_val_dice"] = float(df.loc[best_idx, val_dice_col])
        summary["best_val_dice_epoch"] = int(df.loc[best_idx, epoch_col])
    if val_loss_col:
        best_loss_idx = df[val_loss_col].idxmin()
        summary["best_val_loss"] = float(df.loc[best_loss_idx, val_loss_col])
        summary["best_val_loss_epoch"] = int(df.loc[best_loss_idx, epoch_col])

    pd.Series(summary).to_json(out_dir / "training_summary.json", indent=2)

    print(f"Saved figures to: {out_dir}")
    print(summary)


if __name__ == "__main__":
    main()
