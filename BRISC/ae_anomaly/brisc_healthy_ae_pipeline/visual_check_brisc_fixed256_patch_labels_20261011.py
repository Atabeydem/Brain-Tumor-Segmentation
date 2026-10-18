#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visual sanity-check for fixed 256x256 BRISC patch labels.

Reads patch_manifest_256.csv and creates contact sheets for:
- tumor-positive patches
- tumor-source mask-negative patches

No resize is used. The valid patch region is cropped directly from the source image/mask.
"""

import argparse
import csv
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))


def read_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def choose(rows, label, n, seed):
    pool = [
        r for r in rows
        if r["split"] == "tumor_test"
        and r["source_class"] == "tumor_seg"
        and int(r["patch_label"]) == label
    ]
    rng = random.Random(seed + label)
    rng.shuffle(pool)
    return pool[:min(n, len(pool))]


def crop_valid(row):
    img = load_gray(Path(row["source_image"]))
    mask = load_gray(Path(row["source_mask"]))

    x0 = int(row["x0"])
    y0 = int(row["y0"])
    x1 = int(row["x1_valid"])
    y1 = int(row["y1_valid"])

    return img[y0:y1, x0:x1], mask[y0:y1, x0:x1]


def make_sheet(rows, title, out_path):
    n = len(rows)
    cols = 4
    rows_n = math.ceil(n / cols)

    fig, axes = plt.subplots(rows_n, cols, figsize=(4.2*cols, 4.2*rows_n))
    axes = np.atleast_1d(axes).ravel()

    for ax in axes:
        ax.axis("off")

    for ax, row in zip(axes, rows):
        img, mask = crop_valid(row)
        tumor = mask > 0

        ax.imshow(img, cmap="gray")
        if tumor.any():
            ax.imshow(np.ma.masked_where(~tumor, tumor), cmap="autumn", alpha=0.45)

        frac = float(row["tumor_fraction"]) * 100.0
        ax.set_title(
            f'{row["patch_id"]} | label={row["patch_label"]}\n'
            f'tumor={int(row["tumor_pixels"])} px ({frac:.3f}%)\n'
            f'valid={row["valid_width"]}x{row["valid_height"]}',
            fontsize=9
        )
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--manifest",
        default=r".\outputs\BRISC_FIXED256_PATCH_AUDIT\patch_manifest_256.csv"
    )
    ap.add_argument(
        "--output_dir",
        default=r".\outputs\BRISC_FIXED256_PATCH_AUDIT\visual_label_check"
    )
    ap.add_argument("--n_each", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_rows(Path(args.manifest))
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pos = choose(rows, 1, args.n_each, args.seed)
    neg = choose(rows, 0, args.n_each, args.seed)

    make_sheet(
        pos,
        "Tumor-positive 256x256 patches (mask overlay)",
        out / "tumor_positive_patch_check.png"
    )
    make_sheet(
        neg,
        "Tumor-source mask-negative 256x256 patches",
        out / "tumor_negative_patch_check.png"
    )

    print("Created:")
    print(out / "tumor_positive_patch_check.png")
    print(out / "tumor_negative_patch_check.png")
    print(f"Positive examples: {len(pos)}")
    print(f"Negative examples: {len(neg)}")


if __name__ == "__main__":
    main()
