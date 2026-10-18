r"""
Plot training curves from a BRISC experiment folder.

Run from:
    C:\Users\Atabey\Brain-Tumor-Segmentation\BRISC\segmentation

Example:
    python .\plot_training_history.py --run_dir "..\outputs\unet_baseline_128_e30_gpu_bs8"
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def save_line_plot(df: pd.DataFrame, y_columns, title: str, y_label: str, save_path: Path):
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for col in y_columns:
        if col in df.columns:
            ax.plot(df["epoch"], df[col], marker="o", label=col)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot BRISC training history")
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    history_path = run_dir / "training_history.csv"
    if not history_path.exists():
        raise FileNotFoundError(f"training_history.csv not found: {history_path}")

    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(history_path)
    if "epoch" not in df.columns:
        df.insert(0, "epoch", range(1, len(df) + 1))

    save_line_plot(
        df,
        ["train_loss", "val_loss"],
        "Training and Validation Loss",
        "Loss",
        output_dir / "loss_curve.png",
    )
    save_line_plot(
        df,
        ["train_dice", "val_dice"],
        "Training and Validation Dice",
        "Dice",
        output_dir / "dice_curve.png",
    )

    summary = {
        "epochs": int(df["epoch"].max()),
    }
    if "val_dice" in df.columns:
        best_idx = df["val_dice"].idxmax()
        summary["best_val_dice"] = float(df.loc[best_idx, "val_dice"])
        summary["best_val_dice_epoch"] = int(df.loc[best_idx, "epoch"])
    if "val_loss" in df.columns:
        best_loss_idx = df["val_loss"].idxmin()
        summary["best_val_loss"] = float(df.loc[best_loss_idx, "val_loss"])
        summary["best_val_loss_epoch"] = int(df.loc[best_loss_idx, "epoch"])

    pd.DataFrame([summary]).to_csv(output_dir / "run_summary.csv", index=False)
    print("Saved figures to:", output_dir.resolve())
    print(summary)


if __name__ == "__main__":
    main()
