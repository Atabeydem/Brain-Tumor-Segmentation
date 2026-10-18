r"""
Train a quick 2D slice-based U-Net pilot on BraTS-PEDs-v1.

This is intentionally a lightweight pilot:
- Uses only Training cases, because official Validation cases do not include segmentation masks.
- Splits Training cases into internal train/val sets by case id.
- Converts 3D multimodal NIfTI volumes into 2D axial slices.
- Uses 4-channel input by default: t1n, t1c, t2w, t2f.
- Uses binary tumor mask: seg > 0.

Default dataset path:
    C:\Users\Atabey\Downloads\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit(
        "nibabel is required for reading .nii.gz files. Install it with:\n"
        "    pip install nibabel\n"
    ) from exc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "PKG - BraTS-PEDs-v1" / "BraTS-PEDs-v1"
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_file(case_dir: Path, suffix: str) -> Path:
    return case_dir / f"{case_dir.name}-{suffix}.nii.gz"


def zscore_clip_scale(volume: np.ndarray) -> np.ndarray:
    volume = volume.astype(np.float32)
    mask = volume != 0
    if mask.sum() > 0:
        mean = float(volume[mask].mean())
        std = float(volume[mask].std())
        if std < 1e-6:
            std = 1.0
        volume = (volume - mean) / std
    volume = np.clip(volume, -5.0, 5.0)
    volume = (volume + 5.0) / 10.0
    return volume.astype(np.float32)


def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def build_case_slices(
    case_dirs: Sequence[Path],
    tumor_only: bool = True,
    include_neighbor_slices: int = 0,
) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    for case_dir in tqdm(case_dirs, desc="indexing cases"):
        seg_path = get_file(case_dir, "seg")
        if not seg_path.exists():
            continue
        seg = load_volume(seg_path)
        if tumor_only:
            tumor_slices = np.where((seg > 0).any(axis=(0, 1)))[0].tolist()
            if include_neighbor_slices > 0:
                expanded = set()
                depth = int(seg.shape[2])
                for s in tumor_slices:
                    for ds in range(-include_neighbor_slices, include_neighbor_slices + 1):
                        ns = s + ds
                        if 0 <= ns < depth:
                            expanded.add(ns)
                tumor_slices = sorted(expanded)
            for s in tumor_slices:
                samples.append((case_dir, int(s)))
        else:
            for s in range(int(seg.shape[2])):
                samples.append((case_dir, int(s)))
    return samples


def sample_list(samples: List[Tuple[Path, int]], max_items: int | None, seed: int) -> List[Tuple[Path, int]]:
    if max_items is None or max_items <= 0 or len(samples) <= max_items:
        return samples
    rng = random.Random(seed)
    sampled = rng.sample(samples, max_items)
    return sorted(sampled, key=lambda x: (x[0].name, x[1]))


class PreloadedBraTSPEDSlices(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        modalities: Sequence[str],
        image_size: int = 128,
        desc: str = "loading slices",
    ):
        self.modalities = list(modalities)
        self.image_size = int(image_size)
        self.images: List[torch.Tensor] = []
        self.masks: List[torch.Tensor] = []
        self.meta: List[Dict] = []

        grouped: Dict[Path, List[int]] = {}
        for case_dir, slice_idx in samples:
            grouped.setdefault(case_dir, []).append(int(slice_idx))

        for case_dir, slice_indices in tqdm(grouped.items(), desc=desc):
            volumes = []
            for mod in self.modalities:
                path = get_file(case_dir, mod)
                if not path.exists():
                    raise FileNotFoundError(f"Missing modality file: {path}")
                volumes.append(zscore_clip_scale(load_volume(path)))

            seg_path = get_file(case_dir, "seg")
            if not seg_path.exists():
                raise FileNotFoundError(f"Missing segmentation file: {seg_path}")
            seg = (load_volume(seg_path) > 0).astype(np.float32)

            for slice_idx in slice_indices:
                channels = [vol[:, :, slice_idx] for vol in volumes]
                image_np = np.stack(channels, axis=0).astype(np.float32)  # [C,H,W]
                mask_np = seg[:, :, slice_idx][None, :, :].astype(np.float32)  # [1,H,W]

                image = torch.from_numpy(image_np).unsqueeze(0)
                mask = torch.from_numpy(mask_np).unsqueeze(0)

                image = F.interpolate(
                    image,
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
                mask = F.interpolate(
                    mask,
                    size=(self.image_size, self.image_size),
                    mode="nearest",
                ).squeeze(0)

                self.images.append(image.contiguous())
                self.masks.append(mask.contiguous())
                self.meta.append({"case_id": case_dir.name, "slice_idx": int(slice_idx)})

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.images[idx], self.masks[idx]


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet2D(nn.Module):
    def __init__(self, in_channels: int = 4, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels
        self.enc1 = DoubleConv(in_channels, c)
        self.enc2 = DoubleConv(c, c * 2)
        self.enc3 = DoubleConv(c * 2, c * 4)
        self.enc4 = DoubleConv(c * 4, c * 8)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(c * 8, c * 16)

        self.up4 = nn.ConvTranspose2d(c * 16, c * 8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(c * 16, c * 8)
        self.up3 = nn.ConvTranspose2d(c * 8, c * 4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(c * 8, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(c * 2, c)
        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.up4(b)
        if d4.shape[-2:] != e4.shape[-2:]:
            d4 = F.interpolate(d4, size=e4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))

        d3 = self.up3(d4)
        if d3.shape[-2:] != e3.shape[-2:]:
            d3 = F.interpolate(d3, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[-2:] != e2.shape[-2:]:
            d2 = F.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = F.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.out_conv(d1)


def dice_from_logits(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return dice.mean()


def dice_loss_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    targets = (targets > 0.5).float()
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def tversky_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Tversky loss.

    alpha controls false positives, beta controls false negatives.
    For small lesion segmentation, beta > alpha usually encourages recall.
    """
    probs = torch.sigmoid(logits)
    targets = (targets > 0.5).float()

    tp = (probs * targets).sum(dim=(1, 2, 3))
    fp = (probs * (1.0 - targets)).sum(dim=(1, 2, 3))
    fn = ((1.0 - probs) * targets).sum(dim=(1, 2, 3))

    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky.mean()


def focal_tversky_loss_from_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    gamma: float = 0.75,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Focal Tversky loss.

    gamma < 1.0 is commonly used in medical segmentation to focus optimization
    on hard examples while keeping gradients stable.
    """
    tversky_loss = tversky_loss_from_logits(logits, targets, alpha=alpha, beta=beta, eps=eps)
    return torch.pow(tversky_loss, gamma)


def segmentation_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "bce_dice",
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    focal_gamma: float = 0.75,
) -> torch.Tensor:
    """Configurable segmentation loss.

    Available options:
    - bce_dice: current baseline loss, BCEWithLogits + Dice
    - tversky: pure Tversky loss
    - focal_tversky: pure Focal Tversky loss
    - bce_tversky: BCEWithLogits + Tversky
    - bce_focal_tversky: BCEWithLogits + Focal Tversky
    """
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    dice = dice_loss_from_logits(logits, targets)
    tversky = tversky_loss_from_logits(
        logits,
        targets,
        alpha=tversky_alpha,
        beta=tversky_beta,
    )
    focal_tversky = focal_tversky_loss_from_logits(
        logits,
        targets,
        alpha=tversky_alpha,
        beta=tversky_beta,
        gamma=focal_gamma,
    )

    if loss_type == "bce_dice":
        return bce + dice
    if loss_type == "tversky":
        return tversky
    if loss_type == "focal_tversky":
        return focal_tversky
    if loss_type == "bce_tversky":
        return bce + tversky
    if loss_type == "bce_focal_tversky":
        return bce + focal_tversky

    raise ValueError(f"Unknown loss_type: {loss_type}")


def run_one_epoch(
    model,
    loader,
    optimizer,
    device,
    train: bool,
    loss_type: str,
    tversky_alpha: float,
    tversky_beta: float,
    focal_gamma: float,
) -> Tuple[float, float]:
    model.train(train)
    total_loss = 0.0
    total_dice = 0.0
    total_items = 0
    desc = "train" if train else "val"
    for images, masks in tqdm(loader, desc=desc, leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = segmentation_loss(logits, masks, loss_type=loss_type, tversky_alpha=tversky_alpha, tversky_beta=tversky_beta, focal_gamma=focal_gamma)
            dice = dice_from_logits(logits, masks)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        bs = images.size(0)
        total_loss += float(loss.item()) * bs
        total_dice += float(dice.item()) * bs
        total_items += bs
    return total_loss / max(total_items, 1), total_dice / max(total_items, 1)


def save_prediction_grid(model, dataset: Dataset, device: torch.device, save_path: Path, max_items: int = 6) -> None:
    import matplotlib.pyplot as plt

    model.eval()
    count = min(max_items, len(dataset))
    fig, axes = plt.subplots(count, 3, figsize=(9, 3 * count))
    if count == 1:
        axes = np.expand_dims(axes, axis=0)

    with torch.no_grad():
        for i in range(count):
            image, mask = dataset[i]
            logits = model(image.unsqueeze(0).to(device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
            pred = (prob >= 0.5).astype(np.float32)
            # Display t1c if available as second modality, otherwise first channel.
            display_channel = 1 if image.shape[0] > 1 else 0
            axes[i, 0].imshow(image[display_channel].numpy(), cmap="gray")
            axes[i, 0].set_title("MRI slice")
            axes[i, 0].axis("off")
            axes[i, 1].imshow(pred, cmap="gray")
            axes[i, 1].set_title("Predicted mask")
            axes[i, 1].axis("off")
            axes[i, 2].imshow(mask[0].numpy(), cmap="gray")
            axes[i, 2].set_title("Ground truth mask")
            axes[i, 2].axis("off")

    fig.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)



def load_autoencoder_encoder_weights(seg_model: nn.Module, ae_checkpoint_path: Path, device: torch.device) -> int:
    """Load matching encoder weights from a BraTS-PED UNetAutoencoder2D checkpoint.

    Expected matching modules:
    enc1, enc2, enc3, enc4, bottleneck

    This works because train_brats_ped_unet_autoencoder.py uses the same encoder block names
    and the same base_channels value as the segmentation U-Net when configured consistently.
    """
    if not ae_checkpoint_path.exists():
        raise FileNotFoundError(f"Autoencoder checkpoint does not exist: {ae_checkpoint_path}")

    checkpoint = torch.load(str(ae_checkpoint_path), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        ae_state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        ae_state = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format.")

    seg_state = seg_model.state_dict()
    transfer_prefixes = ("enc1.", "enc2.", "enc3.", "enc4.", "bottleneck.")
    loaded = {}

    for key, value in ae_state.items():
        if key.startswith(transfer_prefixes) and key in seg_state and seg_state[key].shape == value.shape:
            loaded[key] = value

    if not loaded:
        print("WARNING: No matching encoder weights were loaded.")
        print("Possible reasons: base_channels mismatch, modality/channel mismatch, or incompatible checkpoint.")
        return 0

    seg_state.update(loaded)
    seg_model.load_state_dict(seg_state)
    return len(loaded)


def set_encoder_trainable(model: nn.Module, trainable: bool) -> None:
    encoder_names = ("enc1", "enc2", "enc3", "enc4", "bottleneck")
    for name, module in model.named_children():
        if name in encoder_names:
            for param in module.parameters():
                param.requires_grad = trainable


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def parse_args():
    parser = argparse.ArgumentParser(description="BraTS-PED 2D multimodal U-Net pilot")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES, choices=DEFAULT_MODALITIES)
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--loss_type",
        type=str,
        default="bce_dice",
        choices=["bce_dice", "tversky", "focal_tversky", "bce_tversky", "bce_focal_tversky"],
        help="Segmentation loss. bce_dice matches the previous experiments.",
    )
    parser.add_argument("--tversky_alpha", type=float, default=0.3, help="Tversky FP weight.")
    parser.add_argument("--tversky_beta", type=float, default=0.7, help="Tversky FN weight. Larger beta emphasizes recall.")
    parser.add_argument("--focal_gamma", type=float, default=0.75, help="Focal Tversky gamma.")
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--max_train_slices", type=int, default=1500)
    parser.add_argument("--max_val_slices", type=int, default=400)
    parser.add_argument("--include_neighbor_slices", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained_autoencoder", type=str, default=str(Path(__file__).resolve().parent / "outputs" / "brats_ped_ae_denoise_4ch_e5_s3000" / "best_autoencoder.pt"))
    parser.add_argument("--freeze_encoder_epochs", type=int, default=0, help="Freeze transferred encoder for the first N epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path)
    training_dir = dataset_path / "Training"
    if not training_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {training_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "outputs" / "brats_ped_2d_unet_pilot"
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_cases = sorted([p for p in training_dir.iterdir() if p.is_dir()])
    complete_cases = []
    for case_dir in all_cases:
        required = [get_file(case_dir, mod) for mod in args.modalities] + [get_file(case_dir, "seg")]
        if all(p.exists() for p in required):
            complete_cases.append(case_dir)

    rng = random.Random(args.seed)
    rng.shuffle(complete_cases)
    val_count = max(1, int(len(complete_cases) * args.val_ratio))
    val_cases = sorted(complete_cases[:val_count])
    train_cases = sorted(complete_cases[val_count:])

    print("=== BraTS-PED 2D U-Net with Autoencoder Pretraining ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Output folder: {output_dir.resolve()}")
    print(f"Using device: {device}")
    print(f"Complete training cases with selected modalities+seg: {len(complete_cases)}")
    print(f"Internal train cases: {len(train_cases)}")
    print(f"Internal val cases: {len(val_cases)}")
    print(f"Modalities: {args.modalities}")
    print(f"Loss type: {args.loss_type} | alpha={args.tversky_alpha} | beta={args.tversky_beta} | gamma={args.focal_gamma}")

    train_samples = build_case_slices(train_cases, tumor_only=True, include_neighbor_slices=args.include_neighbor_slices)
    val_samples = build_case_slices(val_cases, tumor_only=True, include_neighbor_slices=args.include_neighbor_slices)
    train_samples = sample_list(train_samples, args.max_train_slices, args.seed)
    val_samples = sample_list(val_samples, args.max_val_slices, args.seed + 1)

    print(f"Train tumor-containing slices used: {len(train_samples)}")
    print(f"Val tumor-containing slices used: {len(val_samples)}")

    train_ds = PreloadedBraTSPEDSlices(train_samples, args.modalities, args.image_size, desc="preloading train")
    val_ds = PreloadedBraTSPEDSlices(val_samples, args.modalities, args.image_size, desc="preloading val")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = UNet2D(in_channels=len(args.modalities), out_channels=1, base_channels=args.base_channels).to(device)

    ae_checkpoint_path = Path(args.pretrained_autoencoder).expanduser().resolve()
    loaded_keys = load_autoencoder_encoder_weights(model, ae_checkpoint_path, device)
    print(f"Pretrained autoencoder: {ae_checkpoint_path}")
    print(f"Loaded autoencoder encoder weights: {loaded_keys} keys from ['enc1', 'enc2', 'enc3', 'enc4', 'bottleneck']")

    if args.freeze_encoder_epochs > 0:
        set_encoder_trainable(model, trainable=False)
        print(f"Encoder frozen for the first {args.freeze_encoder_epochs} epoch(s).")

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    config = vars(args).copy()
    config.update({
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir.resolve()),
        "device": str(device),
        "model": "UNet2D_AEPretrained",
        "pretrained_autoencoder": str(ae_checkpoint_path),
        "loaded_encoder_keys": loaded_keys,
        "freeze_encoder_epochs": args.freeze_encoder_epochs,
        "task": "binary tumor segmentation, seg > 0",
        "slice_axis": "axial / z-axis",
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
    })
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_val_loss = float("inf")
    best_val_dice = 0.0

    for epoch in range(1, args.epochs + 1):
        if args.freeze_encoder_epochs > 0 and epoch == args.freeze_encoder_epochs + 1:
            print("Unfreezing encoder and rebuilding optimizer.")
            set_encoder_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=3, factor=0.5)
            print(f"Trainable parameters: {count_trainable_params(model):,}")
        train_loss, train_dice = run_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            loss_type=args.loss_type,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            focal_gamma=args.focal_gamma,
        )
        val_loss, val_dice = run_one_epoch(
            model,
            val_loader,
            optimizer,
            device,
            train=False,
            loss_type=args.loss_type,
            tversky_alpha=args.tversky_alpha,
            tversky_beta=args.tversky_beta,
            focal_gamma=args.focal_gamma,
        )
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_dice": val_dice,
            "lr": lr,
        }
        history.append(record)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | train_dice={train_dice:.4f} | "
            f"val_loss={val_loss:.5f} | val_dice={val_dice:.4f} | lr={lr:.6f}"
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_dice = val_dice
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "best_val_dice": best_val_dice,
                "config": config,
            }
            torch.save(checkpoint, output_dir / "best_brats_ped_2d_unet.pt")
            save_prediction_grid(model, val_ds, device, output_dir / "best_prediction_examples.png")
            print(f"  Saved best checkpoint. best_val_loss={best_val_loss:.5f}, best_val_dice={best_val_dice:.4f}")

    torch.save(model.state_dict(), output_dir / "last_brats_ped_2d_unet_state_dict.pt")
    print("=== Training completed ===")
    print(f"Best validation loss: {best_val_loss:.5f}")
    print(f"Best validation Dice: {best_val_dice:.4f}")
    print(f"Outputs saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
