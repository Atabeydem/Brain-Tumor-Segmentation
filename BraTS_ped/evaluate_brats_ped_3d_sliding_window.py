import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

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
        img = normalize_volume(img)
        images.append(img)

    image = np.stack(images, axis=0).astype(np.float32)

    seg_path = Path(case_item["files"]["seg"])
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
    mask = (seg > 0).astype(np.float32)

    return image, mask


def make_starts(dim: int, patch: int, stride: int):
    if dim <= patch:
        return [0]

    starts = list(range(0, dim - patch + 1, stride))
    last = dim - patch
    if starts[-1] != last:
        starts.append(last)

    return sorted(set(starts))


def sliding_window_predict(
    model,
    image: np.ndarray,
    device,
    patch_size: Tuple[int, int, int],
    stride: Tuple[int, int, int],
    patch_batch_size: int = 4,
):
    """Native-resolution sliding-window inference.

    image shape: [C, H, W, D]
    returns probability map [H, W, D]
    """
    model.eval()

    _, h, w, d = image.shape
    ph, pw, pd = patch_size
    sh, sw, sd = stride

    h_starts = make_starts(h, ph, sh)
    w_starts = make_starts(w, pw, sw)
    d_starts = make_starts(d, pd, sd)

    prob_sum = np.zeros((h, w, d), dtype=np.float32)
    count_sum = np.zeros((h, w, d), dtype=np.float32)

    coords = [(y, x, z) for y in h_starts for x in w_starts for z in d_starts]

    with torch.no_grad():
        for batch_start in range(0, len(coords), patch_batch_size):
            batch_coords = coords[batch_start: batch_start + patch_batch_size]
            patches = []

            for y, x, z in batch_coords:
                patch = image[:, y:y + ph, x:x + pw, z:z + pd]
                patches.append(patch)

            batch = torch.from_numpy(np.stack(patches, axis=0)).float().to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()[:, 0]

            for prob, (y, x, z) in zip(probs, batch_coords):
                prob_sum[y:y + ph, x:x + pw, z:z + pd] += prob
                count_sum[y:y + ph, x:x + pw, z:z + pd] += 1.0

    count_sum[count_sum == 0] = 1.0
    return prob_sum / count_sum, len(coords)


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

    return float(dice), float(iou), float(precision), float(recall), int(tp), int(fp), int(fn)


def save_examples(examples, output_path: Path):
    if not examples:
        return

    rows = len(examples)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, item in enumerate(examples):
        panels = [
            (item["t2f"], "T2F input"),
            (item["mask"], "Ground truth"),
            (item["prob"], "Probability"),
            (item["pred"], f"Prediction th={item['threshold']:.2f}"),
        ]

        for c, (arr, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(arr.T, origin="lower", cmap="gray")
            ax.set_title(f"{title}\n{item['case_id']}, z={item['z']}", fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Native-resolution 3D sliding-window evaluation for BraTS-PED.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--eval_split", choices=["train", "val"], default="val")
    parser.add_argument("--max_cases", type=int, default=10)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--stride", nargs=3, type=int, default=[32, 32, 32])
    parser.add_argument("--patch_batch_size", type=int, default=4)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50])
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

    patch_size = tuple(args.patch_size)
    stride = tuple(args.stride)

    config = {
        "checkpoint": str(Path(args.checkpoint)),
        "split_json": str(split_json),
        "output_dir": str(output_dir),
        "modalities": args.modalities,
        "base_channels": args.base_channels,
        "eval_split": args.eval_split,
        "max_cases": args.max_cases,
        "patch_size": list(patch_size),
        "stride": list(stride),
        "patch_batch_size": args.patch_batch_size,
        "thresholds": args.thresholds,
        "device": str(device),
        "note": "Native-resolution sliding-window evaluation. More reliable than quick resized full-volume evaluation.",
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("=== BraTS-PED 3D Native Sliding-Window Evaluation ===")
    print(f"Checkpoint: {Path(args.checkpoint).resolve()}")
    print(f"Split: {args.eval_split}")
    print(f"Cases evaluated: {len(cases)}")
    print(f"Patch size: {patch_size}")
    print(f"Stride: {stride}")
    print(f"Patch batch size: {args.patch_batch_size}")
    print(f"Thresholds: {args.thresholds}")
    print(f"Device: {device}")

    rows = []
    examples_raw = []

    for idx, case_item in enumerate(cases, start=1):
        case_id = case_item["case_id"]
        image, mask = load_case(case_item, args.modalities)

        prob, n_patches = sliding_window_predict(
            model=model,
            image=image,
            device=device,
            patch_size=patch_size,
            stride=stride,
            patch_batch_size=args.patch_batch_size,
        )

        for th in args.thresholds:
            pred = prob >= th
            dice, iou, precision, recall, tp, fp, fn = dice_iou_precision_recall(pred, mask > 0.5)
            rows.append({
                "case_id": case_id,
                "threshold": th,
                "dice": dice,
                "iou": iou,
                "precision": precision,
                "recall": recall,
                "target_voxels": int((mask > 0.5).sum()),
                "pred_voxels": int(pred.sum()),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "n_patches": n_patches,
            })

        if len(examples_raw) < 4:
            z_sums = mask.sum(axis=(0, 1))
            z = int(np.argmax(z_sums)) if z_sums.max() > 0 else mask.shape[-1] // 2
            examples_raw.append({
                "case_id": case_id,
                "z": z,
                "t2f": image[3, :, :, z],
                "mask": mask[:, :, z],
                "prob": prob[:, :, z],
            })

        print(f"{idx:03d}/{len(cases)} {case_id} done | patches={n_patches}")

    # summary
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
    best_threshold = float(best["threshold"])

    # save examples at best threshold
    examples = []
    for item in examples_raw:
        pred = (item["prob"] >= best_threshold).astype(np.float32)
        examples.append({
            **item,
            "pred": pred,
            "threshold": best_threshold,
        })

    save_examples(examples, output_dir / "sliding_window_examples.png")

    # write outputs
    (output_dir / "case_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (output_dir / "threshold_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "best_threshold.json").write_text(json.dumps(best, indent=2), encoding="utf-8")

    with (output_dir / "case_results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "threshold_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    print("\nBest threshold by mean Dice:")
    print(json.dumps(best, indent=2))
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
