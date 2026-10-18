r"""
Train a 2D U-Net autoencoder / reconstruction model on BraTS-PEDs-v1.

Supported modes:
1) denoise:
   - Input: corrupted multimodal MRI slice, e.g. [t1n, t1c, t2w, t2f]
   - Target: clean multimodal MRI slice
   - This is the closest BraTS-PED equivalent of the BRISC denoising autoencoder.

2) missing:
   - Input: selected available modalities, e.g. [t1n, t1c, t2w]
   - Target: one missing modality, e.g. t2f
   - This is useful for missing-modality reconstruction experiments.

Default dataset path:
    C:\Users\Atabey\Downloads\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1
"""

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
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


def load_volume(path: Path) -> np.ndarray:
    return np.asarray(nib.load(str(path)).dataobj)


def zscore_clip_scale(volume: np.ndarray) -> np.ndarray:
    """Same normalization idea as the segmentation pilot: z-score on nonzero voxels, clip, scale to [0,1]."""
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


def resize_chw(array_chw: np.ndarray, image_size: int) -> np.ndarray:
    """Resize CxHxW numpy array to Cximage_sizeximage_size."""
    tensor = torch.from_numpy(array_chw).unsqueeze(0).float()
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    return tensor.squeeze(0).numpy().astype(np.float32)


def list_complete_training_cases(dataset_path: Path, required_suffixes: Sequence[str]) -> List[Path]:
    train_dir = dataset_path / "Training"
    case_dirs = sorted([p for p in train_dir.iterdir() if p.is_dir()])
    complete = []
    for case_dir in case_dirs:
        if all(get_file(case_dir, s).exists() for s in required_suffixes):
            complete.append(case_dir)
    return complete


def split_cases(case_dirs: Sequence[Path], val_ratio: float, seed: int) -> Tuple[List[Path], List[Path]]:
    case_dirs = list(case_dirs)
    rng = random.Random(seed)
    rng.shuffle(case_dirs)
    n_val = max(1, int(round(len(case_dirs) * val_ratio)))
    val_cases = sorted(case_dirs[:n_val])
    train_cases = sorted(case_dirs[n_val:])
    return train_cases, val_cases


def build_slice_index(
    case_dirs: Sequence[Path],
    slice_filter: str,
    max_slices: int | None,
    seed: int,
) -> List[Tuple[Path, int]]:
    """Create (case_dir, axial_slice_index) list.

    slice_filter='tumor' uses seg > 0 slices.
    slice_filter='all' uses all axial slices.
    """
    all_items: List[Tuple[Path, int]] = []
    for case_dir in tqdm(case_dirs, desc="indexing cases"):
        seg_path = get_file(case_dir, "seg")
        seg = load_volume(seg_path)
        if slice_filter == "tumor":
            mask = seg > 0
            z_indices = np.where(mask.any(axis=(0, 1)))[0].tolist()
        elif slice_filter == "all":
            z_indices = list(range(seg.shape[2]))
        else:
            raise ValueError(f"Unknown slice_filter: {slice_filter}")
        all_items.extend((case_dir, int(z)) for z in z_indices)

    rng = random.Random(seed)
    rng.shuffle(all_items)
    if max_slices is not None and max_slices > 0:
        all_items = all_items[:max_slices]
    return all_items


class BraTSPEDAutoencoderSlices(Dataset):
    def __init__(
        self,
        slice_index: Sequence[Tuple[Path, int]],
        input_modalities: Sequence[str],
        target_modalities: Sequence[str],
        image_size: int,
        mode: str,
        noise_std: float,
        random_erasing_prob: float,
    ):
        self.slice_index = list(slice_index)
        self.input_modalities = list(input_modalities)
        self.target_modalities = list(target_modalities)
        self.image_size = image_size
        self.mode = mode
        self.noise_std = noise_std
        self.random_erasing_prob = random_erasing_prob

        self.inputs: List[np.ndarray] = []
        self.targets: List[np.ndarray] = []
        self.meta: List[Dict[str, str | int]] = []
        self._preload()

    def _preload(self) -> None:
        case_to_slices: Dict[Path, List[int]] = {}
        for case_dir, z in self.slice_index:
            case_to_slices.setdefault(case_dir, []).append(z)

        required_modalities = sorted(set(self.input_modalities + self.target_modalities))

        for case_dir, z_list in tqdm(case_to_slices.items(), desc=f"preloading {self.mode}"):
            volumes = {
                mod: zscore_clip_scale(load_volume(get_file(case_dir, mod)))
                for mod in required_modalities
            }

            for z in z_list:
                input_chw = np.stack([volumes[m][:, :, z] for m in self.input_modalities], axis=0)
                target_chw = np.stack([volumes[m][:, :, z] for m in self.target_modalities], axis=0)

                input_chw = resize_chw(input_chw, self.image_size)
                target_chw = resize_chw(target_chw, self.image_size)

                self.inputs.append(input_chw.astype(np.float32))
                self.targets.append(target_chw.astype(np.float32))
                self.meta.append({"case_id": case_dir.name, "slice": int(z)})

    def __len__(self) -> int:
        return len(self.inputs)

    def _corrupt(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "denoise":
            if self.noise_std > 0:
                x = x + torch.randn_like(x) * self.noise_std
            if self.random_erasing_prob > 0 and random.random() < self.random_erasing_prob:
                _, h, w = x.shape
                erase_h = max(4, int(h * random.uniform(0.05, 0.15)))
                erase_w = max(4, int(w * random.uniform(0.05, 0.15)))
                y0 = random.randint(0, max(0, h - erase_h))
                x0 = random.randint(0, max(0, w - erase_w))
                x[:, y0 : y0 + erase_h, x0 : x0 + erase_w] = 0.0
        return torch.clamp(x, 0.0, 1.0)

    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.targets[idx]).float()
        x = self._corrupt(x)
        return x, y


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


class UNetAutoencoder2D(nn.Module):
    """U-Net style reconstruction model.

    Encoder block names match the segmentation UNet2D:
    enc1, enc2, enc3, enc4, bottleneck.
    This makes encoder transfer straightforward later.
    """

    def __init__(self, in_channels: int = 4, out_channels: int = 4, base_channels: int = 24):
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


def reconstruction_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(logits)
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    return mse + 0.25 * l1


def run_one_epoch(model, loader, optimizer, device, train: bool) -> float:
    model.train(train)
    total_loss = 0.0
    n_samples = 0

    for x, y in tqdm(loader, desc="train" if train else "val", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = reconstruction_loss(logits, y)
            if train:
                loss.backward()
                optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        n_samples += bs

    return total_loss / max(1, n_samples)


@torch.no_grad()
def save_reconstruction_grid(
    model,
    loader,
    device,
    output_path: Path,
    mode: str,
    input_modalities: Sequence[str],
    target_modalities: Sequence[str],
    visual_channel: str,
    n_examples: int = 6,
):
    model.eval()
    x, y = next(iter(loader))
    x = x.to(device)
    y = y.to(device)
    pred = torch.sigmoid(model(x))

    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    p_np = pred.cpu().numpy()

    if visual_channel in target_modalities:
        target_idx = list(target_modalities).index(visual_channel)
    else:
        target_idx = 0

    if mode == "denoise" and visual_channel in input_modalities:
        input_idx = list(input_modalities).index(visual_channel)
        input_label = f"Corrupted input ({visual_channel})"
    else:
        input_idx = 0
        input_label = f"Input ({input_modalities[0]})"

    target_label = f"Clean target ({target_modalities[target_idx]})"
    pred_label = f"Reconstruction ({target_modalities[target_idx]})"

    n = min(n_examples, x_np.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n):
        axes[i, 0].imshow(x_np[i, input_idx], cmap="gray", vmin=0, vmax=1)
        axes[i, 0].set_title(input_label)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(p_np[i, target_idx], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].set_title(pred_label)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(y_np[i, target_idx], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].set_title(target_label)
        axes[i, 2].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a BraTS-PED 2D U-Net autoencoder/reconstruction model.")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default="./outputs/brats_ped_unet_autoencoder_denoise_e5")
    parser.add_argument("--mode", choices=["denoise", "missing"], default="denoise")
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--target_modality", type=str, default="t2f", help="Used only in missing mode.")
    parser.add_argument("--slice_filter", choices=["tumor", "all"], default="tumor")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base_channels", type=int, default=24)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--max_train_slices", type=int, default=3000)
    parser.add_argument("--max_val_slices", type=int, default=800)
    parser.add_argument("--noise_std", type=float, default=0.03)
    parser.add_argument("--random_erasing_prob", type=float, default=0.0)
    parser.add_argument("--visual_channel", type=str, default="t2f")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "denoise":
        input_modalities = list(args.modalities)
        target_modalities = list(args.modalities)
    else:
        input_modalities = list(args.modalities)
        target_modalities = [args.target_modality]
        if args.target_modality in input_modalities:
            raise ValueError("For missing mode, target_modality should not be included in --modalities.")

    required = sorted(set(input_modalities + target_modalities + ["seg"]))
    case_dirs = list_complete_training_cases(dataset_path, required)

    train_cases, val_cases = split_cases(case_dirs, args.val_ratio, args.seed)
    train_index = build_slice_index(train_cases, args.slice_filter, args.max_train_slices, args.seed)
    val_index = build_slice_index(val_cases, args.slice_filter, args.max_val_slices, args.seed + 1)

    print("=== BraTS-PED 2D U-Net Autoencoder / Reconstruction ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Output folder: {output_dir}")
    print(f"Mode: {args.mode}")
    print(f"Input modalities: {input_modalities}")
    print(f"Target modalities: {target_modalities}")
    print(f"Complete labeled cases: {len(case_dirs)}")
    print(f"Internal train cases: {len(train_cases)}")
    print(f"Internal val cases: {len(val_cases)}")
    print(f"Train slices used: {len(train_index)}")
    print(f"Val slices used: {len(val_index)}")

    train_ds = BraTSPEDAutoencoderSlices(
        train_index,
        input_modalities=input_modalities,
        target_modalities=target_modalities,
        image_size=args.image_size,
        mode=args.mode,
        noise_std=args.noise_std,
        random_erasing_prob=args.random_erasing_prob,
    )
    val_ds = BraTSPEDAutoencoderSlices(
        val_index,
        input_modalities=input_modalities,
        target_modalities=target_modalities,
        image_size=args.image_size,
        mode=args.mode,
        noise_std=0.0 if args.mode == "missing" else args.noise_std,
        random_erasing_prob=0.0,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = UNetAutoencoder2D(
        in_channels=len(input_modalities),
        out_channels=len(target_modalities),
        base_channels=args.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    config = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "mode": args.mode,
        "input_modalities": input_modalities,
        "target_modalities": target_modalities,
        "slice_filter": args.slice_filter,
        "image_size": args.image_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "base_channels": args.base_channels,
        "val_ratio": args.val_ratio,
        "max_train_slices": args.max_train_slices,
        "max_val_slices": args.max_val_slices,
        "noise_std": args.noise_std,
        "random_erasing_prob": args.random_erasing_prob,
        "visual_channel": args.visual_channel,
        "seed": args.seed,
        "device": str(device),
        "model": "UNetAutoencoder2D",
        "train_cases": len(train_cases),
        "val_cases": len(val_cases),
        "train_slices": len(train_ds),
        "val_slices": len(val_ds),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = run_one_epoch(model, train_loader, optimizer, device, train=True)
        val_loss = run_one_epoch(model, val_loader, optimizer, device, train=False)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_val_loss": best_val,
                    "epoch": epoch,
                },
                output_dir / "best_autoencoder.pt",
            )
            print(f"  Saved best checkpoint. best_val_loss={best_val:.5f}")

    torch.save(model.state_dict(), output_dir / "last_autoencoder_state_dict.pt")

    save_reconstruction_grid(
        model,
        val_loader,
        device,
        output_dir / "best_reconstruction_examples.png",
        mode=args.mode,
        input_modalities=input_modalities,
        target_modalities=target_modalities,
        visual_channel=args.visual_channel,
        n_examples=6,
    )

    print("=== Training completed ===")
    print(f"Best validation loss: {best_val:.5f}")
    print(f"Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
