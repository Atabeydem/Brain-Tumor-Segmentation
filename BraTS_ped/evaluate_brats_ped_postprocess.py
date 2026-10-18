import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Put this file into the same folder as train_brats_ped_2d_unet_ae_pretrained.py
from train_brats_ped_2d_unet_ae_pretrained import (
    DEFAULT_DATASET_PATH,
    DEFAULT_MODALITIES,
    PreloadedBraTSPEDSlices,
    UNet2D,
    build_case_slices,
    get_file,
    sample_list,
    set_seed,
)


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components smaller than min_area from a binary 2D mask."""
    if min_area <= 0:
        return mask.astype(np.uint8)

    mask = mask.astype(np.uint8)
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    out = np.zeros((h, w), dtype=np.uint8)

    neighbors = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for y in range(h):
        for x in range(w):
            if mask[y, x] == 0 or visited[y, x]:
                continue

            stack = [(y, x)]
            visited[y, x] = True
            coords = []

            while stack:
                cy, cx = stack.pop()
                coords.append((cy, cx))

                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] == 1 and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))

            if len(coords) >= min_area:
                for cy, cx in coords:
                    out[cy, cx] = 1

    return out


def dice_iou_precision_recall(pred: np.ndarray, target: np.ndarray, eps: float = 1e-7):
    pred = pred.astype(bool)
    target = target.astype(bool)

    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()

    dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)

    return dice, iou, precision, recall


def collect_complete_cases(dataset_path: Path, modalities):
    training_dir = dataset_path / "Training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {training_dir}")

    all_cases = sorted([p for p in training_dir.iterdir() if p.is_dir()])
    complete_cases = []

    for case_dir in all_cases:
        required = [get_file(case_dir, mod) for mod in modalities] + [get_file(case_dir, "seg")]
        if all(p.exists() for p in required):
            complete_cases.append(case_dir)

    return complete_cases


def split_cases(complete_cases, val_ratio: float, seed: int):
    cases = list(complete_cases)
    rng = random.Random(seed)
    rng.shuffle(cases)

    val_count = max(1, int(len(cases) * val_ratio))
    val_cases = sorted(cases[:val_count])
    train_cases = sorted(cases[val_count:])
    return train_cases, val_cases


def main():
    parser = argparse.ArgumentParser(description="BraTS-PED post-processing evaluation")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES, choices=DEFAULT_MODALITIES)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--max_val_slices", type=int, default=1200)
    parser.add_argument("--include_neighbor_slices", type=int, default=0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.20])
    parser.add_argument("--min_areas", nargs="+", type=int, default=[0, 5, 10, 20, 30, 50, 75, 100])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    complete_cases = collect_complete_cases(dataset_path, args.modalities)
    train_cases, val_cases = split_cases(complete_cases, args.val_ratio, args.seed)

    print("=== BraTS-PED Post-processing Evaluation ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Using device: {device}")
    print(f"Complete cases: {len(complete_cases)}")
    print(f"Internal train cases: {len(train_cases)}")
    print(f"Internal val cases: {len(val_cases)}")
    print(f"Modalities: {args.modalities}")

    val_samples = build_case_slices(
        val_cases,
        tumor_only=True,
        include_neighbor_slices=args.include_neighbor_slices,
    )
    val_samples = sample_list(val_samples, args.max_val_slices, args.seed + 1)

    print(f"Val tumor-containing slices used: {len(val_samples)}")

    val_ds = PreloadedBraTSPEDSlices(
        val_samples,
        args.modalities,
        args.image_size,
        desc="preloading val for postprocess evaluation",
    )

    loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet2D(
        in_channels=len(args.modalities),
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    checkpoint = torch.load(str(checkpoint_path), map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    probs_all = []
    targets_all = []

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="predicting"):
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits).detach().cpu().numpy()[:, 0]
            targets = masks.detach().cpu().numpy()[:, 0]
            probs_all.append(probs)
            targets_all.append(targets)

    probs_all = np.concatenate(probs_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    rows = []
    for th in args.thresholds:
        for min_area in args.min_areas:
            metrics = []
            for prob, target in zip(probs_all, targets_all):
                pred = (prob >= th).astype(np.uint8)
                pred = remove_small_components(pred, min_area=min_area)
                metrics.append(dice_iou_precision_recall(pred, target))

            arr = np.array(metrics)
            rows.append({
                "threshold": th,
                "min_area": min_area,
                "dice": arr[:, 0].mean(),
                "iou": arr[:, 1].mean(),
                "precision": arr[:, 2].mean(),
                "recall": arr[:, 3].mean(),
                "n_images": len(metrics),
            })

    df = pd.DataFrame(rows).sort_values("dice", ascending=False)
    df.to_csv(output_dir / "postprocess_results.csv", index=False)

    best = df.iloc[0].to_dict()
    with open(output_dir / "best_postprocess.json", "w", encoding="utf-8") as f:
        json.dump(best, f, indent=2)

    print(df.to_string(index=False))
    print("\nBest post-processing setting:")
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
