import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError("nibabel is required. Install it with: pip install nibabel") from exc

try:
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise ImportError("matplotlib is required. Install it with: pip install matplotlib") from exc


DEFAULT_SPLIT_JSON = Path(r".\outputs\splits\brats_ped_split_seed42.json")
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    """Nonzero z-score normalization + clipping.

    This is similar to the 2D pipeline:
    - use nonzero brain/tissue voxels for statistics
    - clip to a stable range
    - keep float32
    """
    volume = volume.astype(np.float32)
    nonzero = volume[volume != 0]

    if nonzero.size > 0:
        mean = nonzero.mean()
        std = nonzero.std()
        if std < 1e-6:
            std = 1.0
        volume = (volume - mean) / std
        volume = np.clip(volume, -5.0, 5.0)
    else:
        volume = np.zeros_like(volume, dtype=np.float32)

    return volume.astype(np.float32)


def load_case(case_item: Dict, modalities: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Load one case as 4D image tensor [C, H, W, D] and mask [1, H, W, D]."""
    images = []

    for mod in modalities:
        path = Path(case_item["files"][mod])
        img = nib.load(str(path)).get_fdata(dtype=np.float32)
        img = normalize_volume(img)
        images.append(img)

    image = np.stack(images, axis=0).astype(np.float32)

    seg_path = Path(case_item["files"]["seg"])
    seg = nib.load(str(seg_path)).get_fdata(dtype=np.float32)
    mask = (seg > 0).astype(np.float32)[None, ...]

    return image, mask


def crop_or_pad_3d(arr: np.ndarray, center: Tuple[int, int, int], patch_size: Tuple[int, int, int]) -> np.ndarray:
    """Crop C,H,W,D array around center; pad if the crop crosses volume boundaries."""
    c, h, w, d = arr.shape
    ph, pw, pd = patch_size
    cy, cx, cz = center

    starts = [cy - ph // 2, cx - pw // 2, cz - pd // 2]
    ends = [starts[0] + ph, starts[1] + pw, starts[2] + pd]
    dims = [h, w, d]
    sizes = [ph, pw, pd]

    src_slices = []
    dst_slices = []

    for start, end, dim, size in zip(starts, ends, dims, sizes):
        src_start = max(0, start)
        src_end = min(dim, end)

        dst_start = max(0, -start)
        dst_end = dst_start + (src_end - src_start)

        src_slices.append(slice(src_start, src_end))
        dst_slices.append(slice(dst_start, dst_end))

    out = np.zeros((c, ph, pw, pd), dtype=arr.dtype)
    out[:, dst_slices[0], dst_slices[1], dst_slices[2]] = arr[:, src_slices[0], src_slices[1], src_slices[2]]
    return out


def choose_patch_center(mask: np.ndarray, patch_size: Tuple[int, int, int], rng: random.Random, tumor_prob: float) -> Tuple[int, int, int]:
    """Choose a patch center.

    mask shape: [1, H, W, D]
    Returns center in H,W,D order.
    """
    _, h, w, d = mask.shape
    ph, pw, pd = patch_size

    tumor_voxels = np.argwhere(mask[0] > 0.5)

    if tumor_voxels.size > 0 and rng.random() < tumor_prob:
        idx = rng.randrange(len(tumor_voxels))
        cy, cx, cz = tumor_voxels[idx].tolist()

        # Add small jitter so model sees surrounding context too.
        cy += rng.randint(-ph // 8, ph // 8)
        cx += rng.randint(-pw // 8, pw // 8)
        cz += rng.randint(-pd // 8, pd // 8)
    else:
        cy = rng.randint(0, h - 1)
        cx = rng.randint(0, w - 1)
        cz = rng.randint(0, d - 1)

    cy = int(np.clip(cy, 0, h - 1))
    cx = int(np.clip(cx, 0, w - 1))
    cz = int(np.clip(cz, 0, d - 1))

    return cy, cx, cz


class BraTSPED3DPatchDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        cases: List[Dict],
        modalities: List[str],
        patch_size: Tuple[int, int, int],
        num_patches: int,
        seed: int,
        tumor_prob: float,
        deterministic: bool = False,
    ):
        self.cases = cases
        self.modalities = modalities
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.seed = seed
        self.tumor_prob = tumor_prob
        self.deterministic = deterministic
        self.cache = {}

    def __len__(self):
        return self.num_patches

    def _get_case(self, case_idx: int):
        if case_idx not in self.cache:
            self.cache[case_idx] = load_case(self.cases[case_idx], self.modalities)

            # keep cache tiny to avoid RAM blow-up
            if len(self.cache) > 2:
                oldest = next(iter(self.cache.keys()))
                if oldest != case_idx:
                    self.cache.pop(oldest, None)

        return self.cache[case_idx]

    def __getitem__(self, idx: int):
        rng = random.Random(self.seed + idx) if self.deterministic else random.Random(random.randint(0, 10**9))

        case_idx = idx % len(self.cases) if self.deterministic else rng.randrange(len(self.cases))
        image, mask = self._get_case(case_idx)

        center = choose_patch_center(mask, self.patch_size, rng, self.tumor_prob)
        image_patch = crop_or_pad_3d(image, center, self.patch_size)
        mask_patch = crop_or_pad_3d(mask, center, self.patch_size)

        return (
            torch.from_numpy(image_patch).float(),
            torch.from_numpy(mask_patch).float(),
            self.cases[case_idx]["case_id"],
        )


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class SmallUNet3D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 8):
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base_channels)
        self.enc2 = ConvBlock3D(base_channels, base_channels * 2)
        self.enc3 = ConvBlock3D(base_channels * 2, base_channels * 4)
        self.bottleneck = ConvBlock3D(base_channels * 4, base_channels * 8)

        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(base_channels * 8, base_channels * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base_channels * 8, base_channels * 4)

        self.up2 = nn.ConvTranspose3d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base_channels * 4, base_channels * 2)

        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base_channels * 2, base_channels)

        self.out = nn.Conv3d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out(d1)


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = (targets > 0.5).float()

    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = (targets > 0.5).float()

    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def hard_dice_score_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()

    dims = tuple(range(1, preds.ndim))
    intersection = (preds * targets).sum(dim=dims)
    denominator = preds.sum(dim=dims) + targets.sum(dim=dims)

    dice = (2.0 * intersection + eps) / (denominator + eps)
    return dice.mean()


def segmentation_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    return bce + dice


def run_one_epoch(model, loader, optimizer, device, train: bool):
    model.train(train)

    total_loss = 0.0
    total_soft_dice = 0.0
    total_hard_dice = 0.0
    n_batches = 0

    for images, masks, _case_ids in loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = segmentation_loss(logits, masks)

            if train:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            soft_dice = soft_dice_score_from_logits(logits, masks)
            hard_dice = hard_dice_score_from_logits(logits, masks)

        total_loss += float(loss.item())
        total_soft_dice += float(soft_dice.item())
        total_hard_dice += float(hard_dice.item())
        n_batches += 1

    return {
        "loss": total_loss / max(1, n_batches),
        "soft_dice": total_soft_dice / max(1, n_batches),
        "hard_dice": total_hard_dice / max(1, n_batches),
    }


def save_prediction_figure(model, dataset, device, output_path: Path, n_examples: int = 3):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    examples = []
    with torch.no_grad():
        for i in range(min(n_examples, len(dataset))):
            image, mask, case_id = dataset[i]
            logits = model(image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
            pred = (prob >= 0.5).astype(np.float32)

            image_np = image.numpy()
            mask_np = mask.numpy()[0]

            # choose z slice with most target tumor voxels
            z_sum = mask_np.sum(axis=(0, 1))
            z = int(np.argmax(z_sum)) if z_sum.max() > 0 else mask_np.shape[-1] // 2

            examples.append((case_id, image_np[3, :, :, z], mask_np[:, :, z], prob[:, :, z], pred[:, :, z], z))

    rows = len(examples)
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for r, (case_id, img, mask, prob, pred, z) in enumerate(examples):
        items = [
            (img, "T2F input"),
            (mask, "Ground truth"),
            (prob, "Probability"),
            (pred, "Prediction th=0.5"),
        ]

        for c, (arr, title) in enumerate(items):
            ax = axes[r, c]
            ax.imshow(arr.T, origin="lower", cmap="gray")
            ax.set_title(f"{title}\n{case_id}, z={z}", fontsize=9)
            ax.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Small 3D U-Net sanity training for BraTS-PED.")
    parser.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--output_dir", type=str, default=r".\outputs\brats_ped_3d_unet_sanity_e1")
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--patch_size", nargs=3, type=int, default=[64, 64, 64])
    parser.add_argument("--base_channels", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_train_patches", type=int, default=96)
    parser.add_argument("--max_val_patches", type=int, default=24)
    parser.add_argument("--tumor_prob", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)

    split_json = Path(args.split_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split = json.loads(split_json.read_text(encoding="utf-8"))
    train_cases = split["train_cases"]
    val_cases = split["val_cases"]

    patch_size = tuple(args.patch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== BraTS-PED 3D U-Net Sanity Run ===")
    print(f"Split JSON: {split_json.resolve()}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Device: {device}")
    print(f"Modalities: {args.modalities}")
    print(f"Patch size: {patch_size}")
    print(f"Base channels: {args.base_channels}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Val cases: {len(val_cases)}")
    print(f"Train patches per epoch: {args.max_train_patches}")
    print(f"Val patches: {args.max_val_patches}")

    train_ds = BraTSPED3DPatchDataset(
        cases=train_cases,
        modalities=args.modalities,
        patch_size=patch_size,
        num_patches=args.max_train_patches,
        seed=args.seed,
        tumor_prob=args.tumor_prob,
        deterministic=False,
    )

    val_ds = BraTSPED3DPatchDataset(
        cases=val_cases,
        modalities=args.modalities,
        patch_size=patch_size,
        num_patches=args.max_val_patches,
        seed=args.seed + 1000,
        tumor_prob=1.0,
        deterministic=True,
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = SmallUNet3D(
        in_channels=len(args.modalities),
        out_channels=1,
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    config = vars(args).copy()
    config.update({
        "device": str(device),
        "model": "SmallUNet3D",
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "trainable_parameters": int(n_params),
        "task": "binary tumor segmentation, seg > 0",
        "note": "Sanity run. Patch-level validation is not final case-level evaluation.",
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_val_loss = float("inf")
    best_val_hard_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_metrics = run_one_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_soft_dice": train_metrics["soft_dice"],
            "train_hard_dice": train_metrics["hard_dice"],
            "val_loss": val_metrics["loss"],
            "val_soft_dice": val_metrics["soft_dice"],
            "val_hard_dice": val_metrics["hard_dice"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={row['train_loss']:.5f} | train_soft_dice={row['train_soft_dice']:.4f} | train_hard_dice={row['train_hard_dice']:.4f} | "
            f"val_loss={row['val_loss']:.5f} | val_soft_dice={row['val_soft_dice']:.4f} | val_hard_dice={row['val_hard_dice']:.4f}"
        )

        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            best_val_hard_dice = row["val_hard_dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "best_val_loss": best_val_loss,
                    "best_val_hard_dice": best_val_hard_dice,
                    "config": config,
                },
                output_dir / "best_3d_unet.pt",
            )
            print(f"  Saved best checkpoint. best_val_loss={best_val_loss:.5f}, best_val_hard_dice={best_val_hard_dice:.4f}")

    torch.save(model.state_dict(), output_dir / "last_3d_unet_state_dict.pt")

    history_path = output_dir / "training_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    save_prediction_figure(
        model,
        val_ds,
        device,
        output_dir / "prediction_examples.png",
        n_examples=min(3, args.max_val_patches),
    )

    print("=== 3D sanity training completed ===")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best validation hard Dice: {best_val_hard_dice:.4f}")
    print(f"Saved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
