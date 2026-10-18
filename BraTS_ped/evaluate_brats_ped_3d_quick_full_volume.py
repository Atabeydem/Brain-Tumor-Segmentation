import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("nibabel is required. Install it with: pip install nibabel") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required. Install it with: pip install matplotlib") from exc

from train_brats_ped_3d_unet_sanity import SmallUNet3D, normalize_volume


DEFAULT_SPLIT_JSON = Path(r".\outputs\splits\brats_ped_split_seed42.json")
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def load_case(case_item: Dict, modalities: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    images = []
    for mod in modalities:
        path = Path(case_item["files"][mod])
        img = nib.load(str(path)).get_fdata(dtype=np.float32)
        images.append(normalize_volume(img))
    image = np.stack(images, axis=0).astype(np.float32)

    seg_path = Path(case_item["files"]["seg"])
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
    mask = (seg > 0).astype(np.float32)

    return image, mask


def dice_iou_precision_recall(pred: np.ndarray, target: np.ndarray, eps: float = 1e-6):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, np.logical_not(target)).sum()
    fn = np.logical_and(np.logical_not(pred), target).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return float(dice), float(iou), float(precision), float(recall)


def pad_to_multiple(volume: np.ndarray, multiple: int = 16):
    # volume: [C,H,W,D] or [H,W,D]
    spatial = volume.shape[-3:]
    pad_width = []
    for dim in spatial:
        remainder = dim % multiple
        pad_after = 0 if remainder == 0 else multiple - remainder
        pad_width.append((0, pad_after))

    if volume.ndim == 4:
        full_pad = [(0, 0)] + pad_width
    else:
        full_pad = pad_width

    padded = np.pad(volume, full_pad, mode="constant", constant_values=0)
    return padded, spatial


def crop_to_shape(volume: np.ndarray, shape: Tuple[int, int, int]):
    if volume.ndim == 4:
        return volume[:, :shape[0], :shape[1], :shape[2]]
    return volume[:shape[0], :shape[1], :shape[2]]


def resize_image_and_mask(image: np.ndarray, mask: np.ndarray, target_shape: Tuple[int, int, int]):
    """Resize full volume to target_shape for quick full-case evaluation.

    image: [C,H,W,D], mask: [H,W,D]
    """
    image_t = torch.from_numpy(image).unsqueeze(0).float()
    mask_t = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float()

    image_resized = F.interpolate(
        image_t,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )[0].numpy()

    mask_resized = F.interpolate(
        mask_t,
        size=target_shape,
        mode="nearest",
    )[0, 0].numpy()

    return image_resized.astype(np.float32), mask_resized.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Quick full-volume evaluation for BraTS-PED 3D U-Net sanity/baseline checkpoints.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--eval_split", choices=["train", "val"], default="val")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--eval_shape", nargs=3, type=int, default=[128, 128, 96])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    args = parser.parse_args()

    split_json = Path(args.split_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(split_json.read_text(encoding="utf-8"))
    cases = split[f"{args.eval_split}_cases"][: args.max_cases]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    model = SmallUNet3D(
        in_channels=len(args.modalities),
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    target_shape = tuple(args.eval_shape)

    print("=== BraTS-PED 3D Quick Full-Volume Evaluation ===")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Split: {args.eval_split}")
    print(f"Cases evaluated: {len(cases)}")
    print(f"Eval shape: {target_shape}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Device: {device}")

    rows = []
    example_payloads = []

    with torch.no_grad():
        for idx, case_item in enumerate(cases, start=1):
            case_id = case_item["case_id"]
            image, mask = load_case(case_item, args.modalities)
            image_resized, mask_resized = resize_image_and_mask(image, mask, target_shape)

            x = torch.from_numpy(image_resized).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)[0, 0].cpu().numpy()

            for th in args.thresholds:
                pred = (probs >= th).astype(np.uint8)
                dice, iou, precision, recall = dice_iou_precision_recall(pred, mask_resized > 0.5)
                rows.append({
                    "case_id": case_id,
                    "threshold": th,
                    "dice": dice,
                    "iou": iou,
                    "precision": precision,
                    "recall": recall,
                    "target_voxels": int((mask_resized > 0.5).sum()),
                    "pred_voxels": int(pred.sum()),
                })

            if len(example_payloads) < 4:
                z_sums = mask_resized.sum(axis=(0, 1))
                z = int(np.argmax(z_sums)) if z_sums.max() > 0 else mask_resized.shape[-1] // 2
                example_payloads.append({
                    "case_id": case_id,
                    "z": z,
                    "t2f": image_resized[3, :, :, z],
                    "mask": mask_resized[:, :, z],
                    "prob": probs[:, :, z],
                    "pred": (probs[:, :, z] >= 0.5).astype(np.float32),
                })

            print(f"{idx:03d}/{len(cases)} {case_id} done")

    # aggregate
    summary = []
    for th in args.thresholds:
        subset = [r for r in rows if abs(r["threshold"] - th) < 1e-8]
        summary.append({
            "threshold": th,
            "mean_dice": float(np.mean([r["dice"] for r in subset])),
            "mean_iou": float(np.mean([r["iou"] for r in subset])),
            "mean_precision": float(np.mean([r["precision"] for r in subset])),
            "mean_recall": float(np.mean([r["recall"] for r in subset])),
            "n_cases": len(subset),
        })

    best = max(summary, key=lambda x: x["mean_dice"])

    (output_dir / "case_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "threshold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "best_threshold.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    # CSV files
    import csv
    with (output_dir / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    # figure
    if example_payloads:
        fig, axes = plt.subplots(len(example_payloads), 4, figsize=(16, 4 * len(example_payloads)))
        if len(example_payloads) == 1:
            axes = np.expand_dims(axes, axis=0)

        for r, item in enumerate(example_payloads):
            panels = [
                (item["t2f"], "T2F input"),
                (item["mask"], "Ground truth"),
                (item["prob"], "Probability"),
                (item["pred"], "Prediction th=0.5"),
            ]
            for c, (arr, title) in enumerate(panels):
                ax = axes[r, c]
                ax.imshow(arr.T, origin="lower", cmap="gray")
                ax.set_title(f"{title}\n{item['case_id']}, z={item['z']}", fontsize=9)
                ax.axis("off")

        fig.tight_layout()
        fig.savefig(output_dir / "full_volume_eval_examples.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    print("\nBest threshold by mean Dice:")
    print(json.dumps(best, indent=2))
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
