import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import ndimage

from brisc_ae_common import (
    list_images,
    load_grayscale_tensor,
    binary_metrics,
    write_csv,
)


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


class PlainCNNAutoencoder(nn.Module):
    def __init__(self, base_channels=64):
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(1, b)
        self.down1 = nn.Conv2d(b, b, 4, stride=2, padding=1)

        self.enc2 = ConvBlock(b, b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 2, 4, stride=2, padding=1)

        self.enc3 = ConvBlock(b * 2, b * 4)
        self.down3 = nn.Conv2d(b * 4, b * 4, 4, stride=2, padding=1)

        self.enc4 = ConvBlock(b * 4, b * 8)
        self.down4 = nn.Conv2d(b * 8, b * 8, 4, stride=2, padding=1)

        self.bottleneck = ConvBlock(b * 8, b * 8)

        self.up4 = nn.ConvTranspose2d(b * 8, b * 4, 4, stride=2, padding=1)
        self.dec4 = ConvBlock(b * 4, b * 4)

        self.up3 = nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(b * 2, b * 2)

        self.up2 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(b, b)

        self.up1 = nn.ConvTranspose2d(b, max(b // 2, 16), 4, stride=2, padding=1)
        self.dec1 = ConvBlock(max(b // 2, 16), max(b // 2, 16))

        self.out = nn.Conv2d(max(b // 2, 16), 1, kernel_size=1)

    def forward(self, x):
        x = self.enc1(x)
        x = torch.relu(self.down1(x))
        x = self.enc2(x)
        x = torch.relu(self.down2(x))
        x = self.enc3(x)
        x = torch.relu(self.down3(x))
        x = self.enc4(x)
        x = torch.relu(self.down4(x))
        x = self.bottleneck(x)

        x = torch.relu(self.up4(x))
        x = self.dec4(x)
        x = torch.relu(self.up3(x))
        x = self.dec3(x)
        x = torch.relu(self.up2(x))
        x = self.dec2(x)
        x = torch.relu(self.up1(x))
        x = self.dec1(x)
        return self.out(x)


def load_plain_cnn_checkpoint(checkpoint_path: Path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    image_size = int(ckpt.get("image_size", 256))
    base_channels = int(ckpt.get("base_channels", 64))
    model = PlainCNNAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()
    return model, image_size


def match_mask(image_path: Path, mask_files_by_stem: dict):
    if image_path.stem in mask_files_by_stem:
        return mask_files_by_stem[image_path.stem]

    candidates = [
        image_path.stem.replace("_image", ""),
        image_path.stem.replace("-image", ""),
        image_path.stem.replace("image", "mask"),
    ]
    for c in candidates:
        if c in mask_files_by_stem:
            return mask_files_by_stem[c]
    return None


def remove_small_components(mask: np.ndarray, min_area: int) -> np.ndarray:
    if min_area <= 0:
        return mask.astype(bool)

    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    keep = counts >= min_area
    keep[0] = False
    return keep[labels]


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    labels, n = ndimage.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)

    counts = np.bincount(labels.ravel())
    counts[0] = 0
    largest = int(np.argmax(counts))
    return labels == largest


def postprocess(mask: np.ndarray, min_area: int, keep_largest: bool) -> np.ndarray:
    out = remove_small_components(mask, min_area)
    if keep_largest:
        out = keep_largest_component(out)
    return out


def summarize_metrics(metrics):
    return {
        "n_cases": len(metrics),
        "mean_dice": float(np.mean([m["dice"] for m in metrics])),
        "mean_iou": float(np.mean([m["iou"] for m in metrics])),
        "mean_precision": float(np.mean([m["precision"] for m in metrics])),
        "mean_recall": float(np.mean([m["recall"] for m in metrics])),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Plain CNN normalized-error anomaly segmentation with post-processing sweep."
    )
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--dataset_root",
        default=r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025",
    )
    ap.add_argument("--split", default="train", choices=["train", "test"])
    ap.add_argument("--output_dir", required=True)

    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12],
    )
    ap.add_argument(
        "--min_areas",
        type=int,
        nargs="+",
        default=[0, 25, 50, 100, 200],
    )
    ap.add_argument(
        "--keep_largest_options",
        type=int,
        nargs="+",
        default=[0, 1],
    )

    ap.add_argument("--sample_cases", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--locked_threshold", type=float, default=None)
    ap.add_argument("--locked_min_area", type=int, default=None)
    ap.add_argument("--locked_keep_largest", type=int, choices=[0, 1], default=None)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, image_size = load_plain_cnn_checkpoint(Path(args.checkpoint), device)

    root = Path(args.dataset_root)
    img_dir = root / "segmentation_task" / args.split / "images"
    mask_dir = root / "segmentation_task" / args.split / "masks"

    images = list_images(img_dir)
    masks = list_images(mask_dir)
    mask_by_stem = {p.stem: p for p in masks}

    if args.sample_cases and 0 < args.sample_cases < len(images):
        rng = random.Random(args.seed)
        idxs = sorted(rng.sample(range(len(images)), args.sample_cases))
        images = [images[i] for i in idxs]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    locked = (
        args.locked_threshold is not None
        and args.locked_min_area is not None
        and args.locked_keep_largest is not None
    )

    if locked:
        configs = [
            (
                float(args.locked_threshold),
                int(args.locked_min_area),
                bool(args.locked_keep_largest),
            )
        ]
    else:
        configs = [
            (float(th), int(ma), bool(kl))
            for th in args.thresholds
            for ma in args.min_areas
            for kl in args.keep_largest_options
        ]

    metrics_by_cfg = {cfg: [] for cfg in configs}
    per_case_rows = []

    print(f"Device: {device}")
    print(f"Split: {args.split}")
    print(f"Cases: {len(images)}")
    print(f"Image size: {image_size}")
    print(f"Configurations: {len(configs)}")
    print("Error map: per-image normalized absolute reconstruction error")

    model.eval()

    with torch.no_grad():
        for idx, img_path in enumerate(images):
            mask_path = match_mask(img_path, mask_by_stem)
            if mask_path is None:
                print(f"WARNING: no mask match for {img_path.name}; skipped")
                continue

            x = load_grayscale_tensor(
                img_path, image_size, nearest=False
            ).unsqueeze(0).to(device)

            gt = (
                load_grayscale_tensor(
                    mask_path, image_size, nearest=True
                ).squeeze().numpy() > 0.5
            )

            recon = torch.clamp(model(x), 0.0, 1.0)
            recon = recon.detach().cpu().squeeze(0)
            orig = x.detach().cpu().squeeze(0)

            err = torch.abs(orig - recon).squeeze().numpy()
            err_norm = err / (err.max() + 1e-8)

            threshold_masks = {}
            for th, _, _ in configs:
                if th not in threshold_masks:
                    threshold_masks[th] = err_norm >= th

            for th, min_area, keep_largest in configs:
                pred = postprocess(
                    threshold_masks[th],
                    min_area=min_area,
                    keep_largest=keep_largest,
                )

                m = binary_metrics(pred, gt)
                m.update({
                    "image": img_path.name,
                    "threshold": th,
                    "min_area": min_area,
                    "keep_largest": int(keep_largest),
                })
                metrics_by_cfg[(th, min_area, keep_largest)].append(m)
                per_case_rows.append(m)

            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{len(images)}")

    rows = []
    for (th, min_area, keep_largest), ms in metrics_by_cfg.items():
        if not ms:
            continue
        s = summarize_metrics(ms)
        rows.append({
            "threshold": th,
            "min_area": min_area,
            "keep_largest": int(keep_largest),
            **s,
        })

    rows.sort(key=lambda r: r["mean_dice"], reverse=True)
    best = rows[0] if rows else {}

    write_csv(out_dir / "postprocess_sweep_summary.csv", rows)
    write_csv(out_dir / "postprocess_per_case.csv", per_case_rows)

    summary = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "image_size": image_size,
        "sample_cases": args.sample_cases,
        "seed": args.seed,
        "locked_configuration": locked,
        "best": best,
        "note": (
            "Error map = abs(original - clamp(reconstruction,0,1)); "
            "per-image normalization = err/(err.max()+1e-8). "
            "Post-processing removes connected components smaller than min_area "
            "and optionally retains only the largest connected component."
        ),
    }

    (out_dir / "postprocess_sweep_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nTop 10 configurations:")
    for r in rows[:10]:
        print(r)

    print("\nSaved outputs:", out_dir)


if __name__ == "__main__":
    main()
