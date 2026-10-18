import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from brisc_ae_common import (
    list_images,
    load_grayscale_tensor,
    load_checkpoint_model,
    binary_metrics,
    write_csv,
    tensor_to_pil,
)


def mask_from_image(x: np.ndarray, method: str = "otsu", fixed_threshold: float = 0.03) -> np.ndarray:
    """Approximate foreground/brain mask from 0-1 grayscale image."""
    if method == "none":
        return np.ones_like(x, dtype=bool)

    if method == "fixed":
        mask = x > fixed_threshold
    else:
        try:
            from skimage.filters import threshold_otsu
            vals = x[x > fixed_threshold]
            if vals.size < 100:
                mask = x > fixed_threshold
            else:
                th = threshold_otsu(vals)
                mask = x > max(float(th) * 0.35, fixed_threshold)
        except Exception:
            mask = x > fixed_threshold

    try:
        from skimage.morphology import remove_small_objects, remove_small_holes, binary_closing, disk
        mask = remove_small_objects(mask.astype(bool), min_size=128)
        mask = remove_small_holes(mask.astype(bool), area_threshold=128)
        mask = binary_closing(mask.astype(bool), disk(3))
    except Exception:
        mask = mask.astype(bool)

    return mask.astype(bool)


def normalize_error(err: np.ndarray, brain_mask: np.ndarray, low_p: float = 1.0, high_p: float = 99.0) -> np.ndarray:
    """Percentile-normalize error map inside foreground mask."""
    e = err.copy().astype(np.float32)
    vals = e[brain_mask]
    if vals.size < 10:
        return np.zeros_like(e, dtype=np.float32)
    lo = np.percentile(vals, low_p)
    hi = np.percentile(vals, high_p)
    if hi <= lo + 1e-8:
        return np.zeros_like(e, dtype=np.float32)
    e = (e - lo) / (hi - lo + 1e-8)
    e = np.clip(e, 0.0, 1.0)
    e[~brain_mask] = 0.0
    return e


def postprocess(pred: np.ndarray, min_area: int = 0, keep_largest: int = 0) -> np.ndarray:
    """Remove small components and optionally keep largest N components."""
    pred = pred.astype(bool)
    try:
        from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
        from skimage.measure import label, regionprops
        if min_area and min_area > 0:
            pred = remove_small_objects(pred, min_size=int(min_area))
        pred = binary_opening(pred, disk(1))
        pred = binary_closing(pred, disk(2))
        if keep_largest and keep_largest > 0:
            lab = label(pred)
            props = sorted(regionprops(lab), key=lambda r: r.area, reverse=True)
            keep = np.zeros_like(pred, dtype=bool)
            for r in props[:int(keep_largest)]:
                keep[lab == r.label] = True
            pred = keep
    except Exception:
        pass
    return pred.astype(bool)


def pil_from_array(arr: np.ndarray, normalize: bool = False) -> Image.Image:
    a = arr.astype(np.float32)
    if normalize:
        a = a - a.min()
        a = a / (a.max() + 1e-8)
    a = np.clip(a, 0, 1)
    return Image.fromarray((a * 255).astype(np.uint8)).convert("L")


def save_contact_sheet(rows, labels, out_path: Path, cell_size: int = 120):
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
        draw.text((title_w + j * cell_size + 5, y), lab, fill="black", font=font)
    y += label_h
    for row_name, imgs in rows:
        draw.text((margin, y + 8), row_name[:30], fill="black", font=small)
        for j, img in enumerate(imgs):
            im = img.convert("RGB").resize((cell_size, cell_size), Image.Resampling.BILINEAR)
            canvas.paste(im, (title_w + j * cell_size, y))
        y += cell_size + label_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


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


def make_prediction(orig_np, recon_np, config):
    raw_err = np.abs(orig_np - recon_np)
    brain = mask_from_image(
        orig_np,
        method=config["brain_mask_method"],
        fixed_threshold=config["brain_fixed_threshold"],
    )
    if config["use_normalized_error"]:
        err_used = normalize_error(
            raw_err,
            brain,
            low_p=config["norm_low_percentile"],
            high_p=config["norm_high_percentile"],
        )
    else:
        err_used = raw_err.copy()
        err_used[~brain] = 0.0

    if config["strategy"] == "percentile":
        vals = err_used[brain]
        if vals.size < 10:
            pred = np.zeros_like(err_used, dtype=bool)
        else:
            th_value = np.percentile(vals, config["percentile"])
            pred = err_used >= th_value
    elif config["strategy"] == "threshold":
        pred = err_used >= config["threshold"]
    else:
        raise ValueError("strategy must be percentile or threshold")

    pred[~brain] = False
    pred = postprocess(pred, min_area=config["min_area"], keep_largest=config["keep_largest"])
    pred[~brain] = False
    return pred, raw_err, err_used, brain


def evaluate_config(cases, model, image_size, device, config):
    metrics = []
    with torch.no_grad():
        for img_path, mask_path in cases:
            x = load_grayscale_tensor(img_path, image_size, nearest=False).unsqueeze(0).to(device)
            gt = load_grayscale_tensor(mask_path, image_size, nearest=True).squeeze().numpy() > 0.5
            recon = model(x).detach().cpu().squeeze(0)
            orig = x.detach().cpu().squeeze(0)
            orig_np = orig.squeeze().numpy().clip(0, 1)
            recon_np = recon.squeeze().numpy().clip(0, 1)
            pred, _, _, _ = make_prediction(orig_np, recon_np, config)
            metrics.append(binary_metrics(pred, gt))
    if not metrics:
        return None
    return {
        "n_cases": len(metrics),
        "mean_dice": float(np.mean([m["dice"] for m in metrics])),
        "mean_iou": float(np.mean([m["iou"] for m in metrics])),
        "mean_precision": float(np.mean([m["precision"] for m in metrics])),
        "mean_recall": float(np.mean([m["recall"] for m in metrics])),
    }


def bool_from_any(v):
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"true", "1", "yes"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, default=r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_cases", type=int, default=0, help="0 means all cases")
    parser.add_argument("--save_examples", type=int, default=12)
    parser.add_argument("--brain_mask_method", type=str, default="otsu", choices=["otsu", "fixed", "none"])
    parser.add_argument("--brain_fixed_threshold", type=float, default=0.03)
    parser.add_argument("--use_normalized_error", action="store_true")
    parser.add_argument("--norm_low_percentile", type=float, default=1.0)
    parser.add_argument("--norm_high_percentile", type=float, default=99.0)
    parser.add_argument("--percentiles", type=float, nargs="+", default=[90, 92, 94, 95, 96, 97, 98, 99, 99.2, 99.5, 99.7, 99.9])
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])
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

    cases = []
    for img in images:
        m = match_mask(img, mask_by_stem)
        if m is not None:
            cases.append((img, m))
    if args.max_cases and args.max_cases > 0:
        cases = cases[:args.max_cases]

    print("Device:", device)
    print("Checkpoint:", args.checkpoint)
    print("Cases:", len(cases))
    print("Image size:", image_size)

    rows = []
    total = 0

    for p in args.percentiles:
        for min_area in args.min_areas:
            for keep_largest in args.keep_largest_values:
                cfg = {
                    "strategy": "percentile",
                    "percentile": float(p),
                    "threshold": "",
                    "min_area": int(min_area),
                    "keep_largest": int(keep_largest),
                    "brain_mask_method": args.brain_mask_method,
                    "brain_fixed_threshold": float(args.brain_fixed_threshold),
                    "use_normalized_error": bool(args.use_normalized_error),
                    "norm_low_percentile": float(args.norm_low_percentile),
                    "norm_high_percentile": float(args.norm_high_percentile),
                }
                stats = evaluate_config(cases, model, image_size, device, cfg)
                if stats is not None:
                    rows.append({**cfg, **stats})
                total += 1
                if total % 20 == 0:
                    print(f"Evaluated configs: {total}")

    for th in args.thresholds:
        for min_area in args.min_areas:
            for keep_largest in args.keep_largest_values:
                cfg = {
                    "strategy": "threshold",
                    "percentile": "",
                    "threshold": float(th),
                    "min_area": int(min_area),
                    "keep_largest": int(keep_largest),
                    "brain_mask_method": args.brain_mask_method,
                    "brain_fixed_threshold": float(args.brain_fixed_threshold),
                    "use_normalized_error": bool(args.use_normalized_error),
                    "norm_low_percentile": float(args.norm_low_percentile),
                    "norm_high_percentile": float(args.norm_high_percentile),
                }
                stats = evaluate_config(cases, model, image_size, device, cfg)
                if stats is not None:
                    rows.append({**cfg, **stats})
                total += 1
                if total % 20 == 0:
                    print(f"Evaluated configs: {total}")

    rows = sorted(rows, key=lambda r: r["mean_dice"], reverse=True)
    write_csv(out_dir / "improved_error_map_results.csv", rows)
    best = rows[0] if rows else {}
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "n_cases": len(cases),
        "image_size": image_size,
        "best": best,
        "note": "Improved evaluation: brain mask, optional normalized error, percentile/fixed threshold sweep, min-area and connected-component post-processing.",
    }
    (out_dir / "improved_error_map_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if best:
        best_cfg = dict(best)
        best_cfg["use_normalized_error"] = bool_from_any(best_cfg["use_normalized_error"])
        best_cfg["percentile"] = None if best_cfg["percentile"] == "" else float(best_cfg["percentile"])
        best_cfg["threshold"] = None if best_cfg["threshold"] == "" else float(best_cfg["threshold"])
        best_cfg["min_area"] = int(best_cfg["min_area"])
        best_cfg["keep_largest"] = int(best_cfg["keep_largest"])
        best_cfg["brain_fixed_threshold"] = float(best_cfg["brain_fixed_threshold"])
        best_cfg["norm_low_percentile"] = float(best_cfg["norm_low_percentile"])
        best_cfg["norm_high_percentile"] = float(best_cfg["norm_high_percentile"])

        example_rows = []
        with torch.no_grad():
            for img_path, mask_path in cases[:args.save_examples]:
                x = load_grayscale_tensor(img_path, image_size, nearest=False).unsqueeze(0).to(device)
                gt = load_grayscale_tensor(mask_path, image_size, nearest=True).squeeze().numpy() > 0.5
                recon = model(x).detach().cpu().squeeze(0)
                orig = x.detach().cpu().squeeze(0)
                orig_np = orig.squeeze().numpy().clip(0, 1)
                recon_np = recon.squeeze().numpy().clip(0, 1)
                pred, raw_err, err_used, brain = make_prediction(orig_np, recon_np, best_cfg)
                example_rows.append((
                    img_path.name,
                    [
                        tensor_to_pil(orig),
                        tensor_to_pil(recon),
                        pil_from_array(raw_err, normalize=True),
                        pil_from_array(err_used, normalize=True),
                        pil_from_array(brain.astype(np.float32)),
                        pil_from_array(pred.astype(np.float32)),
                        pil_from_array(gt.astype(np.float32)),
                    ],
                ))
        save_contact_sheet(
            example_rows,
            ["Original", "Recon", "Raw err", "Err used", "Brain", "Pred", "GT"],
            out_dir / "best_config_examples_contact_sheet.png",
            cell_size=120,
        )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved:", out_dir)


if __name__ == "__main__":
    main()
