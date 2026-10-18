#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS)

def normalize_key(stem: str) -> str:
    s = stem.lower()
    for token in ["_image", "-image", "_img", "-img", "_mask", "-mask"]:
        s = s.replace(token, "")
    s = s.replace("image", "").replace("mask", "")
    return s.strip("_- ")

def build_mask_index(mask_paths: Iterable[Path]) -> Dict[str, Path]:
    out = {}
    for p in mask_paths:
        out[p.stem.lower()] = p
        out[normalize_key(p.stem)] = p
    return out

def match_mask(image_path: Path, mask_index: Dict[str, Path]) -> Optional[Path]:
    for c in [
        image_path.stem.lower(),
        normalize_key(image_path.stem),
        image_path.stem.lower().replace("_image", ""),
        image_path.stem.lower().replace("-image", ""),
        image_path.stem.lower().replace("image", "mask"),
    ]:
        if c in mask_index:
            return mask_index[c]
    return None

def load_gray(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("L"))

def pad_replicate_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape
    return np.pad(arr, ((0, target_h-h), (0, target_w-w)), mode="edge")

def pad_zero_2d(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape
    out = np.zeros((target_h, target_w), dtype=arr.dtype)
    out[:h, :w] = arr
    return out

def safe_name(path: Path) -> str:
    return f"{path.parent.name}__{path.stem}".replace(" ", "_")

def extract_patch_records(image_path, patch_size, split_name, source_class, image_label,
                          mask_path, output_root, save_patches):
    img = load_gray(image_path)
    h, w = img.shape
    if mask_path is not None:
        mask = load_gray(mask_path)
        if mask.shape != img.shape:
            raise RuntimeError(f"Image/mask size mismatch; resize is forbidden:\n{image_path} {img.shape}\n{mask_path} {mask.shape}")
        mask_bin = mask > 0
    else:
        mask_bin = np.zeros_like(img, dtype=bool)

    n_rows = math.ceil(h / patch_size)
    n_cols = math.ceil(w / patch_size)
    image_patch_count = n_rows * n_cols
    records = []
    tumor_patch_count = 0

    for r in range(n_rows):
        y0, y1 = r*patch_size, min((r+1)*patch_size, h)
        for c in range(n_cols):
            x0, x1 = c*patch_size, min((c+1)*patch_size, w)
            patch = img[y0:y1, x0:x1]
            mask_patch = mask_bin[y0:y1, x0:x1]
            valid_h, valid_w = patch.shape
            pad_bottom, pad_right = patch_size-valid_h, patch_size-valid_w
            tumor_pixels = int(mask_patch.sum())
            valid_pixels = int(valid_h*valid_w)
            tumor_fraction = tumor_pixels/valid_pixels if valid_pixels else 0.0
            patch_label = int(tumor_pixels > 0)
            tumor_patch_count += patch_label
            patch_id = f"r{r:02d}_c{c:02d}"
            base = safe_name(image_path)
            patch_out = ""
            mask_out = ""

            if save_patches:
                img_dir = output_root/"patches"/split_name/source_class/"images"
                msk_dir = output_root/"patches"/split_name/source_class/"masks"
                img_dir.mkdir(parents=True, exist_ok=True)
                if mask_path is not None:
                    msk_dir.mkdir(parents=True, exist_ok=True)
                patch_file = img_dir/f"{base}__{patch_id}.png"
                Image.fromarray(pad_replicate_2d(patch, patch_size, patch_size).astype(np.uint8), mode="L").save(patch_file)
                patch_out = str(patch_file)
                if mask_path is not None:
                    mask_file = msk_dir/f"{base}__{patch_id}_mask.png"
                    mask_u8 = mask_patch.astype(np.uint8)*255
                    Image.fromarray(pad_zero_2d(mask_u8, patch_size, patch_size), mode="L").save(mask_file)
                    mask_out = str(mask_file)

            records.append({
                "split": split_name,
                "source_class": source_class,
                "source_image": str(image_path),
                "source_mask": str(mask_path) if mask_path else "",
                "image_id": base,
                "image_label": int(image_label),
                "patch_id": patch_id,
                "row_idx": r, "col_idx": c,
                "x0": x0, "y0": y0, "x1_valid": x1, "y1_valid": y1,
                "original_width": w, "original_height": h,
                "patch_size": patch_size,
                "valid_width": valid_w, "valid_height": valid_h,
                "pad_right": pad_right, "pad_bottom": pad_bottom,
                "image_patch_count": image_patch_count,
                "tumor_pixels": tumor_pixels,
                "valid_pixels": valid_pixels,
                "tumor_fraction": tumor_fraction,
                "patch_label": patch_label,
                "patch_image_path": patch_out,
                "patch_mask_path": mask_out,
            })

    return records, {
        "source_image": str(image_path),
        "source_class": source_class,
        "split": split_name,
        "original_height": h,
        "original_width": w,
        "n_patches": image_patch_count,
        "n_tumor_patches": tumor_patch_count,
        "n_clean_patches": image_patch_count-tumor_patch_count,
    }

def write_csv(path: Path, rows: List[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

def summarize(rows, image_summaries):
    by_split = defaultdict(lambda: {"images":0,"patches":0,"tumor_patches":0,"clean_patches":0,"padded_patches":0})
    seen_images = defaultdict(set)
    source_class_counts = Counter()
    shape_counts = Counter()
    tumor_source_patch_counts = Counter()

    for row in rows:
        s = row["split"]
        seen_images[s].add(row["source_image"])
        by_split[s]["patches"] += 1
        if int(row["patch_label"]):
            by_split[s]["tumor_patches"] += 1
        else:
            by_split[s]["clean_patches"] += 1
        if int(row["pad_right"]) or int(row["pad_bottom"]):
            by_split[s]["padded_patches"] += 1
        source_class_counts[f'{row["split"]}/{row["source_class"]}'] += 1
        shape_counts[f'{row["original_width"]}x{row["original_height"]}'] += 1
        if row["source_class"] == "tumor_seg":
            tumor_source_patch_counts["tumor_positive" if int(row["patch_label"]) else "tumor_source_mask_negative"] += 1

    for s, imgs in seen_images.items():
        by_split[s]["images"] = len(imgs)

    dist = Counter(int(x["n_tumor_patches"]) for x in image_summaries if x["source_class"] == "tumor_seg")

    return {
        "by_split": dict(by_split),
        "patch_counts_by_source_class": dict(source_class_counts),
        "original_shape_frequency_top20": dict(shape_counts.most_common(20)),
        "tumor_source_patch_label_counts": dict(tumor_source_patch_counts),
        "tumor_images_by_number_of_positive_patches": {str(k): v for k, v in sorted(dist.items())},
        "important_note": "Fixed 256x256, stride 256, no resize. Edge image areas use replicate padding; mask padding is zero. Labels come only from valid mask pixels."
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")
    ap.add_argument("--output_dir", default=r".\outputs\BRISC_FIXED256_PATCH_AUDIT")
    ap.add_argument("--patch_size", type=int, default=256)
    ap.add_argument("--save_patches", action="store_true")
    ap.add_argument("--include_seg_train", action="store_true")
    args = ap.parse_args()

    root, out = Path(args.dataset_root), Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    datasets = [
        ("healthy_train","healthy",0,root/"classification_task"/"train"/"no_tumor",None),
        ("healthy_test","healthy",0,root/"classification_task"/"test"/"no_tumor",None),
        ("tumor_test","tumor_seg",1,root/"segmentation_task"/"test"/"images",root/"segmentation_task"/"test"/"masks"),
    ]
    if args.include_seg_train:
        datasets.append(("tumor_train_seg_only_for_analysis","tumor_seg",1,root/"segmentation_task"/"train"/"images",root/"segmentation_task"/"train"/"masks"))

    all_rows, all_image_summaries, missing_masks = [], [], []

    print("="*90)
    print("BRISC FIXED 256 PATCH AUDIT")
    print("="*90)
    print("Spatial resize : NONE")
    print("Patch/stride   : 256 / 256")
    print("Edge handling  : image replicate padding, mask zero padding")
    print("Save PNGs      :", args.save_patches)
    print()

    for split_name, source_class, image_label, image_dir, mask_dir in datasets:
        images = list_images(image_dir)
        print(f"[{split_name}] images={len(images)} | {image_dir}")
        mask_index = {}
        if mask_dir is not None:
            masks = list_images(mask_dir)
            mask_index = build_mask_index(masks)
            print(f"[{split_name}] masks ={len(masks)} | {mask_dir}")

        for i, image_path in enumerate(images, 1):
            mask_path = None
            if mask_dir is not None:
                mask_path = match_mask(image_path, mask_index)
                if mask_path is None:
                    missing_masks.append(str(image_path))
                    continue

            rows, imsum = extract_patch_records(
                image_path, args.patch_size, split_name, source_class, image_label,
                mask_path, out, args.save_patches
            )
            all_rows.extend(rows)
            all_image_summaries.append(imsum)
            if i % 100 == 0 or i == len(images):
                print(f"  processed {i}/{len(images)}")

    if missing_masks:
        p = out/"missing_masks.txt"
        p.write_text("\n".join(missing_masks), encoding="utf-8")
        raise RuntimeError(f"{len(missing_masks)} image(s) had no matching mask. See {p}")

    manifest = out/"patch_manifest_256.csv"
    image_summary = out/"image_patch_summary_256.csv"
    summary_json = out/"patch_audit_summary_256.json"
    write_csv(manifest, all_rows)
    write_csv(image_summary, all_image_summaries)
    summary = summarize(all_rows, all_image_summaries)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n"+"="*90)
    print("SUMMARY")
    print("="*90)
    for s, x in summary["by_split"].items():
        print(f"{s:34s} images={x['images']:4d} | patches={x['patches']:5d} | tumor+={x['tumor_patches']:5d} | clean={x['clean_patches']:5d} | padded={x['padded_patches']:5d}")

    print("\nTumor-source patch labels:")
    for k,v in summary["tumor_source_patch_label_counts"].items():
        print(f"  {k}: {v}")

    print("\nTumor images by number of positive patches:")
    for k,v in summary["tumor_images_by_number_of_positive_patches"].items():
        print(f"  {k} positive patch(es): {v} image(s)")

    print("\nSaved:")
    print(" ", manifest)
    print(" ", image_summary)
    print(" ", summary_json)
    if not args.save_patches:
        print("\nAudit-only run. After we verify counts, rerun with --save_patches if needed.")

if __name__ == "__main__":
    main()
