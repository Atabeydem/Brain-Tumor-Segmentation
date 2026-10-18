r"""
Evaluate threshold values for a trained BRISC U-Net checkpoint.

Run from:
    C:\Users\Atabey\Brain-Tumor-Segmentation\BRISC\segmentation

Example:
    python .\evaluate_thresholds.py --checkpoint "..\outputs\unet_baseline_128_e30_gpu_bs8\best_unet.pt" --output_dir "..\outputs\unet_baseline_128_e30_gpu_bs8\threshold_eval"
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# This script is intended to live in the same folder as train_brisc_unet.py
from train_brisc_unet import DEFAULT_DATASET_PATH, BRISCSegmentationDataset, UNet2D


def parse_thresholds(value: str):
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def compute_batch_metrics(probs: torch.Tensor, masks: torch.Tensor, threshold: float, eps: float = 1e-6):
    preds = (probs >= threshold).float()
    dims = (1, 2, 3)

    tp = torch.sum(preds * masks, dims)
    fp = torch.sum(preds * (1.0 - masks), dims)
    fn = torch.sum((1.0 - preds) * masks, dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice_sum": dice.sum().item(),
        "iou_sum": iou.sum().item(),
        "precision_sum": precision.sum().item(),
        "recall_sum": recall.sum().item(),
        "n": masks.size(0),
    }


def save_prediction_grid(model, loader, device, save_path: Path, threshold: float, max_items: int = 6):
    import matplotlib.pyplot as plt

    model.eval()
    images, masks = next(iter(loader))
    images = images[:max_items].to(device)
    masks = masks[:max_items].to(device)

    with torch.no_grad():
        probs = torch.sigmoid(model(images))
        preds = (probs >= threshold).float()

    images = images.cpu().numpy()
    masks = masks.cpu().numpy()
    preds = preds.cpu().numpy()
    probs = probs.cpu().numpy()

    rows = min(max_items, images.shape[0])
    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(rows):
        axes[i, 0].imshow(images[i, 0], cmap="gray")
        axes[i, 0].set_title("MRI image")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(probs[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title("Probability map")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(preds[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(f"Predicted mask\nth={threshold:.2f}")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 3].set_title("Ground truth mask")
        axes[i, 3].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate BRISC U-Net thresholds")
    parser.add_argument("--checkpoint", type=str, default="../outputs/unet_baseline_128_e30_gpu_bs8/best_unet.pt")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--val_split", type=str, default=None)
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--base_channels", type=int, default=None)
    parser.add_argument("--thresholds", type=str, default="0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if args.output_dir is None:
        output_dir = checkpoint_path.parent / "threshold_eval"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {})

    image_size = args.image_size or int(config.get("image_size", 128))
    base_channels = args.base_channels or int(config.get("base_channels", 32))
    val_split = args.val_split or str(config.get("val_split", "test"))
    dataset_path = Path(args.dataset_path)

    print("=== BRISC Threshold Evaluation ===")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Using device: {device}")
    print(f"Image size: {image_size}")
    print(f"Base channels: {base_channels}")
    print(f"Validation split: {val_split}")

    val_ds = BRISCSegmentationDataset(
        dataset_path=str(dataset_path),
        split=val_split,
        image_size=image_size,
        augment=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet2D(in_channels=1, out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    thresholds = parse_thresholds(args.thresholds)
    accum = {
        th: {"dice_sum": 0.0, "iou_sum": 0.0, "precision_sum": 0.0, "recall_sum": 0.0, "n": 0}
        for th in thresholds
    }

    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="evaluating"):
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images))

            for th in thresholds:
                batch_metrics = compute_batch_metrics(probs, masks, threshold=th)
                for key, value in batch_metrics.items():
                    accum[th][key] += value

    records = []
    for th in thresholds:
        n = max(accum[th]["n"], 1)
        records.append({
            "threshold": th,
            "dice": accum[th]["dice_sum"] / n,
            "iou": accum[th]["iou_sum"] / n,
            "precision": accum[th]["precision_sum"] / n,
            "recall": accum[th]["recall_sum"] / n,
            "n_images": n,
        })

    results = pd.DataFrame(records).sort_values("threshold")
    results.to_csv(output_dir / "threshold_results.csv", index=False)

    best_row = results.loc[results["dice"].idxmax()].to_dict()
    (output_dir / "best_threshold.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBest threshold by Dice:")
    print(json.dumps(best_row, indent=2))

    save_prediction_grid(
        model,
        val_loader,
        device,
        output_dir / f"best_threshold_predictions_th_{best_row['threshold']:.2f}.png",
        threshold=float(best_row["threshold"]),
    )

    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
