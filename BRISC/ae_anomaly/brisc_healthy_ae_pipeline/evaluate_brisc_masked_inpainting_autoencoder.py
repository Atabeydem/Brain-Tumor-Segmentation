import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    from skimage.metrics import structural_similarity as ssim_fn
except Exception:
    ssim_fn = None

from brisc_ae_common import (
    psnr_from_mse,
    tensor_to_pil,
    save_contact_sheet,
)

from train_brisc_masked_inpainting_autoencoder import (
    BRISCMaskedInpaintingDataset,
    build_pairs,
    UNetMaskedInpaintingAE,
    weighted_masked_mse_loss,
)


def compute_ssim(target_np, recon_np):
    if ssim_fn is None:
        return None

    target_np = np.squeeze(target_np)
    recon_np = np.squeeze(recon_np)

    return float(
        ssim_fn(
            target_np,
            recon_np,
            data_range=1.0
        )
    )


def evaluate(model, loader, device, mask_weight):
    model.eval()

    rows = []

    weighted_losses = []
    full_mses = []
    mask_mses = []
    nonmask_mses = []
    psnrs = []
    ssims = []

    with torch.no_grad():
        for batch_idx, (inp, target, mask, img_paths, mask_paths) in enumerate(loader):
            inp = inp.to(device)
            target = target.to(device)
            mask = mask.to(device)

            recon = model(inp)

            weighted_loss = weighted_masked_mse_loss(
                recon,
                target,
                mask,
                mask_weight=mask_weight
            )
            weighted_losses.append(float(weighted_loss.item()))

            diff2 = (recon - target) ** 2

            bsz = inp.size(0)

            for b in range(bsz):
                full_mse = float(diff2[b].mean().item())
                psnr = float(psnr_from_mse(full_mse))

                mask_bool = mask[b] > 0.5
                nonmask_bool = mask[b] <= 0.5

                if mask_bool.sum() > 0:
                    mask_mse = float(diff2[b][mask_bool].mean().item())
                else:
                    mask_mse = float("nan")

                if nonmask_bool.sum() > 0:
                    nonmask_mse = float(diff2[b][nonmask_bool].mean().item())
                else:
                    nonmask_mse = float("nan")

                target_np = target[b].detach().cpu().numpy()
                recon_np = recon[b].detach().cpu().numpy()
                ssim_value = compute_ssim(target_np, recon_np)

                full_mses.append(full_mse)
                mask_mses.append(mask_mse)
                nonmask_mses.append(nonmask_mse)
                psnrs.append(psnr)
                if ssim_value is not None:
                    ssims.append(ssim_value)

                rows.append({
                    "index": len(rows),
                    "image_path": img_paths[b],
                    "mask_path": mask_paths[b],
                    "weighted_loss_batch": float(weighted_loss.item()),
                    "full_mse": full_mse,
                    "mask_region_mse": mask_mse,
                    "nonmask_region_mse": nonmask_mse,
                    "psnr_db": psnr,
                    "ssim": "" if ssim_value is None else ssim_value,
                })

    summary = {
        "n_images": len(rows),
        "mean_weighted_mse_loss": float(np.mean(weighted_losses)) if weighted_losses else None,

        "mean_full_mse": float(np.nanmean(full_mses)),
        "std_full_mse": float(np.nanstd(full_mses)),

        "mean_mask_region_mse": float(np.nanmean(mask_mses)),
        "std_mask_region_mse": float(np.nanstd(mask_mses)),

        "mean_nonmask_region_mse": float(np.nanmean(nonmask_mses)),
        "std_nonmask_region_mse": float(np.nanstd(nonmask_mses)),

        "mean_psnr_db": float(np.nanmean(psnrs)),
        "std_psnr_db": float(np.nanstd(psnrs)),
        "min_psnr_db": float(np.nanmin(psnrs)),
        "max_psnr_db": float(np.nanmax(psnrs)),

        "mean_ssim": None if len(ssims) == 0 else float(np.mean(ssims)),
        "std_ssim": None if len(ssims) == 0 else float(np.std(ssims)),
        "note": "SSIM is None if scikit-image is not installed. Install with: pip install scikit-image",
    }

    return rows, summary


def save_sample_contact_sheet(model, dataset, device, out_path, max_samples=8):
    model.eval()

    sample_rows = []

    with torch.no_grad():
        for i in range(min(max_samples, len(dataset))):
            inp, target, mask, img_path, _ = dataset[i]

            recon = model(inp.unsqueeze(0).to(device)).cpu().squeeze(0)

            masked_img = inp[0:1]
            mask_img = inp[1:2]
            diff = torch.abs(recon - target)

            sample_rows.append((
                Path(img_path).name,
                [
                    tensor_to_pil(masked_img),
                    tensor_to_pil(mask_img),
                    tensor_to_pil(recon),
                    tensor_to_pil(target),
                    tensor_to_pil(diff),
                ]
            ))

    save_contact_sheet(
        sample_rows,
        ["Masked Input", "Mask", "Reconstructed", "Target", "Abs Diff"],
        out_path
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--sample_count", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device)

    config = ckpt.get("config", {})
    image_size = int(config.get("image_size", 256))
    base_channels = int(config.get("base_channels", 64))
    mask_weight = float(config.get("mask_weight", 20.0))

    dataset_root = Path(args.dataset_root)

    if args.split == "train":
        images_dir = dataset_root / "segmentation_task" / "train" / "images"
        masks_dir = dataset_root / "segmentation_task" / "train" / "masks"
    else:
        images_dir = dataset_root / "segmentation_task" / "test" / "images"
        masks_dir = dataset_root / "segmentation_task" / "test" / "masks"

    pairs = build_pairs(images_dir, masks_dir)

    if args.max_images and args.max_images > 0:
        pairs = pairs[:args.max_images]

    dataset = BRISCMaskedInpaintingDataset(pairs, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = UNetMaskedInpaintingAE(
        in_channels=2,
        out_channels=1,
        base_channels=base_channels
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Device:", device)
    print("Checkpoint:", checkpoint_path)
    print("Split:", args.split)
    print("Images:", images_dir)
    print("Masks:", masks_dir)
    print("Pairs:", len(pairs))
    print("Image size:", image_size)
    print("Base channels:", base_channels)
    print("Mask weight:", mask_weight)
    print("Output:", out_dir)

    rows, summary = evaluate(
        model=model,
        loader=loader,
        device=device,
        mask_weight=mask_weight
    )

    summary.update({
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "image_size": image_size,
        "base_channels": base_channels,
        "mask_weight": mask_weight,
    })

    with open(out_dir / "masked_inpainting_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(out_dir / "masked_inpainting_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    save_sample_contact_sheet(
        model=model,
        dataset=dataset,
        device=device,
        out_path=out_dir / "masked_inpainting_examples_contact_sheet.png",
        max_samples=args.sample_count
    )

    print()
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print()
    print("Saved:")
    print(out_dir / "masked_inpainting_metrics.csv")
    print(out_dir / "masked_inpainting_summary.json")
    print(out_dir / "masked_inpainting_examples_contact_sheet.png")


if __name__ == "__main__":
    main()