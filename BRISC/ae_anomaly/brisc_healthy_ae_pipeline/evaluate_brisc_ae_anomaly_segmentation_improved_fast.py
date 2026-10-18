import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from brisc_ae_common import (
    list_images,
    load_grayscale_tensor,
    load_checkpoint_model,
    binary_metrics,
    write_csv,
    tensor_to_pil,
)


def match_mask(image_path: Path, mask_files_by_stem: dict):
    if image_path.stem in mask_files_by_stem:
        return mask_files_by_stem[image_path.stem]
    candidates = [
        image_path.stem.replace("_image", ""),
        image_path.stem.replace("-image", ""),
        image_path.stem.replace("image", "mask"),
        image_path.stem.replace("_img", ""),
        image_path.stem.replace("-img", ""),
    ]
    for c in candidates:
        if c in mask_files_by_stem:
            return mask_files_by_stem[c]
    return None


def make_foreground_mask(x: np.ndarray, fixed_threshold: float = 0.03) -> np.ndarray:
    """Approximate head/brain foreground. Removes pure-black background."""
    mask = x > fixed_threshold
    try:
        from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
        mask = remove_small_objects(mask.astype(bool), min_size=128)
        mask = remove_small_holes(mask.astype(bool), area_threshold=128)
        try:
            mask = binary_closing(mask.astype(bool), footprint=disk(3))
        except TypeError:
            mask = binary_closing(mask.astype(bool), disk(3))
    except Exception:
        mask = mask.astype(bool)
    return mask.astype(bool)


def normalize_error(err: np.ndarray, fg: np.ndarray, low_p=1.0, high_p=99.0) -> np.ndarray:
    vals = err[fg]
    if vals.size < 10:
        return np.zeros_like(err, dtype=np.float32)
    lo = np.percentile(vals, low_p)
    hi = np.percentile(vals, high_p)
    if hi <= lo + 1e-8:
        return np.zeros_like(err, dtype=np.float32)
    out = (err - lo) / (hi - lo + 1e-8)
    out = np.clip(out, 0.0, 1.0).astype(np.float32)
    out[~fg] = 0.0
    return out


def postprocess(pred: np.ndarray, min_area: int = 0, keep_largest: int = 0) -> np.ndarray:
    pred = pred.astype(bool)
    try:
        from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
        from skimage.measure import label, regionprops
        if min_area > 0:
            pred = remove_small_objects(pred, min_size=int(min_area))
        try:
            pred = binary_opening(pred, footprint=disk(1))
            pred = binary_closing(pred, footprint=disk(2))
        except TypeError:
            pred = binary_opening(pred, disk(1))
            pred = binary_closing(pred, disk(2))
        if keep_largest > 0:
            lab = label(pred)
            props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
            keep = np.zeros_like(pred, dtype=bool)
            for r in props[:int(keep_largest)]:
                keep[lab == r.label] = True
            pred = keep
    except Exception:
        pass
    return pred.astype(bool)


def pil_from_array(arr: np.ndarray, normalize=False) -> Image.Image:
    a = arr.astype(np.float32)
    if normalize:
        a = a - a.min()
        a = a / (a.max() + 1e-8)
    a = np.clip(a, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8)).convert("L")


def save_contact_sheet(rows, labels, out_path: Path, cell_size=120):
    if not rows:
        return
    margin = 18
    label_h = 28
    title_w = 230
    cols = len(labels)
    w = title_w + cols * cell_size + margin * 2
    h = margin + label_h + len(rows) * (cell_size + label_h) + margin
    canvas = Image.new("RGB", (w, h), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
        small = ImageFont.truetype("arial.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
        small = ImageFont.load_default()
    y = margin
    for j, lab in enumerate(labels):
        draw.text((title_w + j * cell_size + 4, y), lab, fill="black", font=font)
    y += label_h
    for row_name, imgs in rows:
        draw.text((margin, y + 8), row_name[:30], fill="black", font=small)
        for j, img in enumerate(imgs):
            im = img.convert("RGB").resize((cell_size, cell_size), Image.Resampling.BILINEAR)
            canvas.paste(im, (title_w + j * cell_size, y))
        y += cell_size + label_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def predict_from_case(case, cfg):
    err = case["norm_err"] if cfg["use_normalized_error"] else case["raw_err"]
    fg = case["fg"]
    vals = err[fg]
    if vals.size < 10:
        pred = np.zeros_like(err, dtype=bool)
    elif cfg["strategy"] == "percentile":
        th_value = np.percentile(vals, cfg["percentile"])
        pred = err >= th_value
    else:
        pred = err >= cfg["threshold"]
    pred[~fg] = False
    pred = postprocess(pred, cfg["min_area"], cfg["keep_largest"])
    pred[~fg] = False
    return pred


def evaluate_cfg(cases, cfg):
    ms = []
    for case in cases:
        pred = predict_from_case(case, cfg)
        ms.append(binary_metrics(pred, case["gt"]))
    return {
        "n_cases": len(ms),
        "mean_dice": float(np.mean([m["dice"] for m in ms])),
        "mean_iou": float(np.mean([m["iou"] for m in ms])),
        "mean_precision": float(np.mean([m["precision"] for m in ms])),
        "mean_recall": float(np.mean([m["recall"] for m in ms])),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=0, help="0 = all cases")
    parser.add_argument("--save_examples", type=int, default=12)
    parser.add_argument("--use_normalized_error", action="store_true")
    parser.add_argument("--foreground_threshold", type=float, default=0.03)
    parser.add_argument("--norm_low_percentile", type=float, default=1.0)
    parser.add_argument("--norm_high_percentile", type=float, default=99.0)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[90, 92, 94, 95, 96, 97, 98, 99, 99.2, 99.5, 99.7, 99.9])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90])
    parser.add_argument("--min_areas", type=int, nargs="+", default=[0, 20, 50, 100, 200, 500])
    parser.add_argument("--keep_largest_values", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt_config, image_size = load_checkpoint_model(Path(args.checkpoint), device)

    root = Path(args.dataset_root)
    img_dir = root / "segmentation_task" / args.split / "images"
    mask_dir = root / "segmentation_task" / args.split / "masks"
    images = list_images(img_dir)
    masks = list_images(mask_dir)
    mask_by_stem = {p.stem: p for p in masks}

    pairs = []
    for img in images:
        m = match_mask(img, mask_by_stem)
        if m is not None:
            pairs.append((img, m))
    if args.max_cases > 0:
        pairs = pairs[:args.max_cases]

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Pairs:", len(pairs))
    print("Image size:", image_size)
    print("Step 1/2: computing reconstructions and error maps once...")

    cases = []
    model.eval()
    with torch.no_grad():
        for i, (img_path, mask_path) in enumerate(pairs, 1):
            x = load_grayscale_tensor(img_path, image_size, nearest=False).unsqueeze(0).to(device)
            gt = load_grayscale_tensor(mask_path, image_size, nearest=True).squeeze().numpy() > 0.5
            recon = model(x).detach().cpu().squeeze(0)
            orig = x.detach().cpu().squeeze(0)
            orig_np = orig.squeeze().numpy().clip(0, 1)
            recon_np = recon.squeeze().numpy().clip(0, 1)
            raw_err = np.abs(orig_np - recon_np).astype(np.float32)
            fg = make_foreground_mask(orig_np, fixed_threshold=args.foreground_threshold)
            norm_err = normalize_error(raw_err, fg, args.norm_low_percentile, args.norm_high_percentile)
            cases.append({
                "name": img_path.name,
                "orig": orig_np,
                "recon": recon_np,
                "raw_err": raw_err,
                "norm_err": norm_err,
                "fg": fg,
                "gt": gt,
            })
            if i % 100 == 0 or i == len(pairs):
                print(f"  computed {i}/{len(pairs)}")

    print("Step 2/2: sweeping thresholds/post-processing configs...")
    configs = []
    for p in args.percentiles:
        for area in args.min_areas:
            for keep in args.keep_largest_values:
                configs.append({
                    "strategy": "percentile",
                    "percentile": float(p),
                    "threshold": "",
                    "min_area": int(area),
                    "keep_largest": int(keep),
                    "use_normalized_error": bool(args.use_normalized_error),
                })
    for th in args.thresholds:
        for area in args.min_areas:
            for keep in args.keep_largest_values:
                configs.append({
                    "strategy": "threshold",
                    "percentile": "",
                    "threshold": float(th),
                    "min_area": int(area),
                    "keep_largest": int(keep),
                    "use_normalized_error": bool(args.use_normalized_error),
                })

    rows = []
    for i, cfg in enumerate(configs, 1):
        stats = evaluate_cfg(cases, cfg)
        rows.append({**cfg, **stats})
        if i % 50 == 0 or i == len(configs):
            best_so_far = max(rows, key=lambda r: r["mean_dice"])
            print(f"  evaluated {i}/{len(configs)} | best Dice so far: {best_so_far['mean_dice']:.4f}")

    rows = sorted(rows, key=lambda r: r["mean_dice"], reverse=True)
    write_csv(out_dir / "improved_error_map_results.csv", rows)
    best = rows[0] if rows else {}

    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_cases": len(cases),
        "image_size": image_size,
        "foreground_threshold": args.foreground_threshold,
        "use_normalized_error": args.use_normalized_error,
        "norm_low_percentile": args.norm_low_percentile,
        "norm_high_percentile": args.norm_high_percentile,
        "best": best,
        "note": "Fast version: model inference is done once per image; then threshold/postprocess configs are swept.",
    }
    (out_dir / "improved_error_map_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if best:
        example_rows = []
        for case in cases[:args.save_examples]:
            pred = predict_from_case(case, best)
            example_rows.append((case["name"], [
                pil_from_array(case["orig"]),
                pil_from_array(case["recon"]),
                pil_from_array(case["raw_err"], normalize=True),
                pil_from_array(case["norm_err"], normalize=True),
                pil_from_array(case["fg"].astype(np.float32)),
                pil_from_array(pred.astype(np.float32)),
                pil_from_array(case["gt"].astype(np.float32)),
            ]))
        save_contact_sheet(
            example_rows,
            ["Original", "Recon", "Raw err", "Norm err", "FG mask", "Pred", "GT"],
            out_dir / "best_config_examples_contact_sheet.png",
            cell_size=120,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Saved outputs:", out_dir)


if __name__ == "__main__":
    main()
