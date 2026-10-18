#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def audit_folder(folder: Path, label: str):
    paths = list_images(folder)
    size_counter = Counter()
    mode_counter = Counter()
    ext_counter = Counter()
    corrupted = []
    rows = []
    exact_512 = 0
    even_dims = 0
    divisible_256 = 0

    for i, p in enumerate(paths, 1):
        try:
            with Image.open(p) as im:
                w, h = im.size
                mode = im.mode

            size_counter[(w, h)] += 1
            mode_counter[mode] += 1
            ext_counter[p.suffix.lower()] += 1

            is_512 = (w, h) == (512, 512)
            is_even = (w % 2 == 0 and h % 2 == 0)
            is_div256 = (w % 256 == 0 and h % 256 == 0)

            exact_512 += int(is_512)
            even_dims += int(is_even)
            divisible_256 += int(is_div256)

            rows.append({
                "split": label,
                "filename": p.name,
                "width": w,
                "height": h,
                "mode": mode,
                "extension": p.suffix.lower(),
                "is_512x512": int(is_512),
                "even_width_height": int(is_even),
                "divisible_by_256": int(is_div256),
            })
        except Exception as e:
            corrupted.append({
                "split": label,
                "filename": p.name,
                "path": str(p),
                "error": repr(e),
            })

        if i % 250 == 0:
            print(f"{label}: checked {i}/{len(paths)}")

    summary = {
        "split": label,
        "folder": str(folder),
        "n_images": len(paths),
        "n_readable": len(rows),
        "n_corrupted": len(corrupted),
        "n_exact_512x512": exact_512,
        "pct_exact_512x512": (100.0 * exact_512 / len(rows)) if rows else 0.0,
        "n_even_dimensions": even_dims,
        "n_dimensions_divisible_by_256": divisible_256,
        "size_distribution": [
            {"width": w, "height": h, "count": c}
            for (w, h), c in sorted(size_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
        ],
        "mode_distribution": dict(mode_counter),
        "extension_distribution": dict(ext_counter),
    }
    return summary, rows, corrupted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--output_dir", default="./outputs/HEALTHY_RESOLUTION_AUDIT")
    args = ap.parse_args()

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_dir = root / "classification_task" / "train" / "no_tumor"
    test_dir = root / "classification_task" / "test" / "no_tumor"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train folder not found: {train_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test folder not found: {test_dir}")

    print("=" * 78)
    print("BRISC HEALTHY IMAGE RESOLUTION AUDIT")
    print("NO resize. NO training. Only native image metadata is inspected.")
    print("=" * 78)

    train_summary, train_rows, train_bad = audit_folder(train_dir, "train_no_tumor")
    test_summary, test_rows, test_bad = audit_folder(test_dir, "test_no_tumor")

    all_rows = train_rows + test_rows
    all_bad = train_bad + test_bad

    with open(out / "healthy_image_dimensions_per_file.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["split", "filename", "width", "height", "mode", "extension",
                  "is_512x512", "even_width_height", "divisible_by_256"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    with open(out / "healthy_corrupted_images.csv", "w", newline="", encoding="utf-8") as f:
        fields = ["split", "filename", "path", "error"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_bad)

    combined_size_counter = Counter((r["width"], r["height"]) for r in all_rows)
    combined_mode_counter = Counter(r["mode"] for r in all_rows)

    combined = {
        "train": train_summary,
        "test": test_summary,
        "combined": {
            "n_images": len(all_rows),
            "n_exact_512x512": sum(r["is_512x512"] for r in all_rows),
            "size_distribution": [
                {"width": w, "height": h, "count": c}
                for (w, h), c in sorted(combined_size_counter.items(), key=lambda kv: (-kv[1], kv[0][0], kv[0][1]))
            ],
            "mode_distribution": dict(combined_mode_counter),
        },
        "decision_note": (
            "Do not resize any image. If healthy images are 512x512, next preprocessing is fixed 2x2 tiling "
            "into four native 256x256 patches. Other dimensions must be reviewed separately first."
        )
    }

    (out / "healthy_resolution_audit_summary.json").write_text(
        json.dumps(combined, indent=2), encoding="utf-8"
    )

    with open(out / "healthy_resolution_audit_report.txt", "w", encoding="utf-8") as f:
        for s in [train_summary, test_summary]:
            f.write(f"\n=== {s['split']} ===\n")
            f.write(f"Images: {s['n_images']}\n")
            f.write(f"Readable: {s['n_readable']}\n")
            f.write(f"Corrupted: {s['n_corrupted']}\n")
            f.write(f"Exact 512x512: {s['n_exact_512x512']} ({s['pct_exact_512x512']:.2f}%)\n")
            f.write(f"Even dimensions: {s['n_even_dimensions']}\n")
            f.write(f"Dimensions divisible by 256: {s['n_dimensions_divisible_by_256']}\n")
            f.write("Top size distribution:\n")
            for x in s["size_distribution"][:25]:
                f.write(f"  {x['width']}x{x['height']}: {x['count']}\n")
            f.write("Modes:\n")
            for k, v in s["mode_distribution"].items():
                f.write(f"  {k}: {v}\n")

        f.write("\n=== NEXT STEP RULE ===\n")
        f.write("No resize. If healthy training images are 512x512, split each into a fixed 2x2 grid of four 256x256 native-resolution patches.\n")

    def print_summary(s):
        print(f"\n=== {s['split']} ===")
        print(f"Images                : {s['n_images']}")
        print(f"Readable              : {s['n_readable']}")
        print(f"Corrupted             : {s['n_corrupted']}")
        print(f"Exact 512x512         : {s['n_exact_512x512']} ({s['pct_exact_512x512']:.2f}%)")
        print(f"Even dimensions       : {s['n_even_dimensions']}")
        print(f"Divisible by 256      : {s['n_dimensions_divisible_by_256']}")
        print("Top dimensions:")
        for x in s["size_distribution"][:15]:
            print(f"  {x['width']}x{x['height']}: {x['count']}")
        print("Modes:")
        for k, v in s["mode_distribution"].items():
            print(f"  {k}: {v}")

    print_summary(train_summary)
    print_summary(test_summary)
    print("\nSaved outputs:", out)
    print("IMPORTANT: No image was resized or modified.")

if __name__ == "__main__":
    main()
