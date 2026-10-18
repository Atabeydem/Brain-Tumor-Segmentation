r"""
Evaluate probability thresholds for a trained BraTS-PED 2D U-Net model.

This script is designed to be placed inside:
    C:\Users\Atabey\Brain-Tumor-Segmentation\BraTS_ped

It reuses the dataset/model utilities from train_brats_ped_2d_unet.py, rebuilds the
same internal validation slice subset using the saved checkpoint config, then tests
multiple binary-mask thresholds.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Reuse the exact same definitions used during training.
from train_brats_ped_2d_unet import (
    DEFAULT_DATASET_PATH,
    UNet2D,
    PreloadedBraTSPEDSlices,
    build_case_slices,
    get_file,
    sample_list,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate thresholds for BraTS-PED 2D U-Net")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to best_brats_ped_2d_unet.pt. If omitted, defaults to the "
            "multimodal pilot run checkpoint."
        ),
    )
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_visual_items", type=int, default=6)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> Dict:
    # weights_only=False is important for newer PyTorch versions when loading full checkpoints.
    return torch.load(str(path), map_location=device, weights_only=False)


def rebuild_validation_dataset(config: Dict, dataset_path: Path):
    seed = int(config.get("seed", 42))
    val_ratio = float(config.get("val_ratio", 0.20))
    modalities = list(config.get("modalities", ["t1n", "t1c", "t2w", "t2f"]))
    image_size = int(config.get("image_size", 128))
    max_val_slices = int(config.get("max_val_slices", 800))
    include_neighbor_slices = int(config.get("include_neighbor_slices", 0))

    training_dir = dataset_path / "Training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {training_dir}")

    all_cases = sorted([p for p in training_dir.iterdir() if p.is_dir()])
    complete_cases = []
    for case_dir in all_cases:
        required = [get_file(case_dir, mod) for mod in modalities] + [get_file(case_dir, "seg")]
        if all(p.exists() for p in required):
            complete_cases.append(case_dir)

    rng = random.Random(seed)
    rng.shuffle(complete_cases)
    val_count = max(1, int(len(complete_cases) * val_ratio))
    val_cases = sorted(complete_cases[:val_count])

    val_samples = build_case_slices(
        val_cases,
        tumor_only=True,
        include_neighbor_slices=include_neighbor_slices,
    )
    val_samples = sample_list(val_samples, max_val_slices, seed + 1)

    val_ds = PreloadedBraTSPEDSlices(
        val_samples,
        modalities,
        image_size=image_size,
        desc="preloading val for threshold evaluation",
    )
    return val_ds, val_cases, val_samples


def compute_metrics_for_threshold(probs: torch.Tensor, targets: torch.Tensor, threshold: float, eps: float = 1e-6):
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()

    dims = (1, 2, 3)
    tp = (preds * targets).sum(dim=dims)
    fp = (preds * (1.0 - targets)).sum(dim=dims)
    fn = ((1.0 - preds) * targets).sum(dim=dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return {
        "dice": float(dice.mean().item()),
        "iou": float(iou.mean().item()),
        "precision": float(precision.mean().item()),
        "recall": float(recall.mean().item()),
    }


def save_prediction_grid(model, dataset, device, save_path: Path, threshold: float, max_items: int = 6):
    import matplotlib.pyplot as plt

    model.eval()
    count = min(max_items, len(dataset))
    fig, axes = plt.subplots(count, 4, figsize=(12, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for i in range(count):
            image, mask = dataset[i]
            logits = model(image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob >= threshold).astype(np.float32)

            # If multimodal, display t1c as second channel; otherwise display the only channel.
            display_channel = 1 if image.shape[0] > 1 else 0
            axes[i, 0].imshow(image[display_channel].numpy(), cmap="gray")
            axes[i, 0].set_title("MRI slice")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(prob, cmap="gray", vmin=0, vmax=1)
            axes[i, 1].set_title("Probability map")
            axes[i, 1].axis("off")

            axes[i, 2].imshow(pred, cmap="gray")
            axes[i, 2].set_title(f"Predicted mask\nth={threshold:.2f}")
            axes[i, 2].axis("off")

            axes[i, 3].imshow(mask[0].numpy(), cmap="gray")
            axes[i, 3].set_title("Ground truth mask")
            axes[i, 3].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    if args.checkpoint is None:
        checkpoint_path = script_dir / "outputs" / "brats_ped_2d_unet_pilot_e10_s3000" / "best_brats_ped_2d_unet.pt"
    else:
        checkpoint_path = Path(args.checkpoint)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint.get("config", {})

    dataset_path = Path(args.dataset_path) if args.dataset_path else Path(config.get("dataset_path", DEFAULT_DATASET_PATH))
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent / "threshold_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(int(config.get("seed", 42)))

    modalities = list(config.get("modalities", ["t1n", "t1c", "t2w", "t2f"]))
    image_size = int(config.get("image_size", 128))
    base_channels = int(config.get("base_channels", 24))
    batch_size = int(config.get("batch_size", 8))

    print("=== BraTS-PED Threshold Evaluation ===")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Dataset path: {dataset_path}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Using device: {device}")
    print(f"Modalities: {modalities}")
    print(f"Image size: {image_size}")
    print(f"Base channels: {base_channels}")

    val_ds, val_cases, val_samples = rebuild_validation_dataset(config, dataset_path)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet2D(in_channels=len(modalities), out_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_probs: List[torch.Tensor] = []
    all_targets: List[torch.Tensor] = []
    with torch.no_grad():
        for images, masks in tqdm(val_loader, desc="evaluating"):
            images = images.to(device, non_blocking=True)
            logits = model(images)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)
            all_targets.append(masks.cpu())

    probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)

    rows = []
    for th in args.thresholds:
        metrics = compute_metrics_for_threshold(probs, targets, float(th))
        rows.append({
            "threshold": float(th),
            **metrics,
            "n_images": int(targets.shape[0]),
        })

    results_df = pd.DataFrame(rows)
    best_row = results_df.sort_values("dice", ascending=False).iloc[0].to_dict()

    results_df.to_csv(output_dir / "threshold_results.csv", index=False)
    (output_dir / "best_threshold.json").write_text(json.dumps(best_row, indent=2), encoding="utf-8")

    save_prediction_grid(
        model,
        val_ds,
        device,
        output_dir / f"best_threshold_predictions_th_{best_row['threshold']:.2f}.png",
        threshold=float(best_row["threshold"]),
        max_items=args.max_visual_items,
    )

    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBest threshold by Dice:")
    print(json.dumps(best_row, indent=2))
    print(f"Saved results to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
