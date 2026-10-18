import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

try:
    from skimage.metrics import structural_similarity
except Exception:
    structural_similarity = None

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

IMG_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(folder):
    return sorted(
        p for p in Path(folder).rglob('*')
        if p.is_file() and p.suffix.lower() in IMG_EXTS
    )


def preprocessed_gray_hash(path: Path, image_size: int):
    """Hash the exact 8-bit grayscale+resize image seen before ToTensor()."""
    with Image.open(path) as im:
        im = im.convert('L').resize((image_size, image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('utf-8'))
    h.update(arr.tobytes())
    return h.hexdigest()


class TumorPairDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=256):
        images = list_images(image_dir)
        masks = {p.stem: p for p in list_images(mask_dir)}
        self.pairs = [(p, masks[p.stem]) for p in images if p.stem in masks]
        if not self.pairs:
            raise FileNotFoundError('No matched tumor image-mask pairs found.')

        self.image_tf = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.ToTensor(),
        ])
        self.mask_tf = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        ip, mp = self.pairs[idx]
        with Image.open(ip) as im:
            x = self.image_tf(im.convert('L'))
        with Image.open(mp) as mm:
            mask = (self.mask_tf(mm.convert('L')) > 0.5).float()
        return x, mask, str(ip)


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


class UNetPlusPlusAutoencoder(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels
        self.pool = nn.MaxPool2d(2, 2)

        self.conv0_0 = ConvBlock(1, b)
        self.conv1_0 = ConvBlock(b, b * 2)
        self.conv2_0 = ConvBlock(b * 2, b * 4)
        self.conv3_0 = ConvBlock(b * 4, b * 8)

        self.conv0_1 = ConvBlock(b + b * 2, b)
        self.conv1_1 = ConvBlock(b * 2 + b * 4, b * 2)
        self.conv2_1 = ConvBlock(b * 4 + b * 8, b * 4)

        self.conv0_2 = ConvBlock(b * 2 + b * 2, b)
        self.conv1_2 = ConvBlock(b * 4 + b * 4, b * 2)

        self.conv0_3 = ConvBlock(b * 3 + b * 2, b)
        self.out = nn.Conv2d(b, 1, kernel_size=1)

    @staticmethod
    def up_to(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x2_0 = self.conv2_0(self.pool(x1_0))
        x3_0 = self.conv3_0(self.pool(x2_0))

        x0_1 = self.conv0_1(torch.cat([x0_0, self.up_to(x1_0, x0_0)], dim=1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up_to(x2_0, x1_0)], dim=1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up_to(x3_0, x2_0)], dim=1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up_to(x1_1, x0_0)], dim=1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up_to(x2_1, x1_0)], dim=1))

        x0_3 = self.conv0_3(
            torch.cat([x0_0, x0_1, x0_2, self.up_to(x1_2, x0_0)], dim=1)
        )
        return self.out(x0_3)


@torch.no_grad()
def evaluate(model, loader, device, collect_examples=False):
    rows = []
    examples = []
    model.eval()

    for x, mask, paths in loader:
        x = x.to(device, non_blocking=True)
        pred = torch.clamp(model(x), 0.0, 1.0).cpu().numpy()
        x_np = x.cpu().numpy()
        mask_np = mask.numpy()

        for i in range(x_np.shape[0]):
            target = x_np[i, 0]
            recon = pred[i, 0]
            m = mask_np[i, 0] > 0.5
            nm = ~m

            sq = (recon - target) ** 2
            ae = np.abs(recon - target)
            full_mse = float(sq.mean())
            mask_mse = float(sq[m].mean())
            nonmask_mse = float(sq[nm].mean())

            row = {
                'image': paths[i],
                'full_mse': full_mse,
                'full_mae': float(ae.mean()),
                'full_psnr_db': float(-10.0 * math.log10(max(full_mse, 1e-12))),
                'mask_mse': mask_mse,
                'mask_mae': float(ae[m].mean()),
                'mask_psnr_db': float(-10.0 * math.log10(max(mask_mse, 1e-12))),
                'nonmask_mse': nonmask_mse,
                'nonmask_mae': float(ae[nm].mean()),
                'nonmask_psnr_db': float(-10.0 * math.log10(max(nonmask_mse, 1e-12))),
                'mask_area_ratio': float(m.mean()),
                'mask_to_nonmask_mse_ratio': float(mask_mse / max(nonmask_mse, 1e-12)),
                'ssim': None,
            }
            if structural_similarity is not None:
                row['ssim'] = float(structural_similarity(target, recon, data_range=1.0))
            rows.append(row)

            if collect_examples and len(examples) < 8:
                examples.append((target, recon, np.abs(target-recon), m, row['full_psnr_db']))

    return rows, examples


def summarize(rows):
    keys = [
        'full_mse', 'full_mae', 'full_psnr_db',
        'mask_mse', 'mask_mae', 'mask_psnr_db',
        'nonmask_mse', 'nonmask_mae', 'nonmask_psnr_db',
        'mask_area_ratio', 'mask_to_nonmask_mse_ratio',
    ]
    out = {'n_images': len(rows)}
    for key in keys:
        vals = np.asarray([r[key] for r in rows], dtype=np.float64)
        out[f'mean_{key}'] = float(vals.mean())
        out[f'std_{key}'] = float(vals.std())
    ssims = [r['ssim'] for r in rows if r['ssim'] is not None]
    out['mean_ssim'] = float(np.mean(ssims)) if ssims else None
    out['std_ssim'] = float(np.std(ssims)) if ssims else None
    return out


def save_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def save_examples(path, examples):
    if plt is None or not examples:
        return
    fig, axes = plt.subplots(len(examples), 4, figsize=(12, 3 * len(examples)))
    if len(examples) == 1:
        axes = np.array([axes])
    for i, (target, recon, err, mask, psnr) in enumerate(examples):
        axes[i, 0].imshow(target, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title('Original tumor image')
        axes[i, 1].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'Reconstructed | {psnr:.2f} dB')
        axes[i, 2].imshow(err, cmap='gray')
        axes[i, 2].set_title('Absolute error')
        axes[i, 3].imshow(mask, cmap='gray', vmin=0, vmax=1)
        axes[i, 3].set_title('GT tumor mask')
        for j in range(4):
            axes[i, j].axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--eval_batch_size', type=int, default=4)
    ap.add_argument('--workers', type=int, default=2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.checkpoint, map_location=device)
    image_size = int(ckpt.get('image_size', 256))
    base_channels = int(ckpt.get('base_channels', 32))

    root = Path(args.dataset_root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_img_dir = root / 'segmentation_task' / 'train' / 'images'
    test_img_dir = root / 'segmentation_task' / 'test' / 'images'
    test_mask_dir = root / 'segmentation_task' / 'test' / 'masks'

    train_imgs = list_images(train_img_dir)
    test_ds = TumorPairDataset(test_img_dir, test_mask_dir, image_size=image_size)

    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Image size from checkpoint: {image_size}')
    print(f'Base channels from checkpoint: {base_channels}')
    print(f'Train images: {len(train_imgs)} | Test pairs: {len(test_ds)}')
    print('\nBuilding post-preprocessing hashes (grayscale + resize)...')

    train_hash_to_paths = {}
    for i, p in enumerate(train_imgs, 1):
        h = preprocessed_gray_hash(p, image_size)
        train_hash_to_paths.setdefault(h, []).append(p)
        if i % 500 == 0:
            print(f'  train hashed {i}/{len(train_imgs)}')

    clean_indices = []
    excluded_rows = []
    for idx, (ip, mp) in enumerate(test_ds.pairs):
        h = preprocessed_gray_hash(ip, image_size)
        if h in train_hash_to_paths:
            excluded_rows.append({
                'test_image': str(ip),
                'test_mask': str(mp),
                'matching_train_images': ' | '.join(str(p) for p in train_hash_to_paths[h]),
                'n_matching_train_images': len(train_hash_to_paths[h]),
                'preprocessed_sha256': h,
            })
        else:
            clean_indices.append(idx)

    print(f'\nPost-preprocessing overlap test images: {len(excluded_rows)}')
    print(f'Clean test images: {len(clean_indices)}')

    if excluded_rows:
        save_csv(out / 'excluded_preprocessed_overlap_images.csv', excluded_rows)

    model = UNetPlusPlusAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    all_loader = DataLoader(
        test_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda')
    )
    clean_ds = Subset(test_ds, clean_indices)
    clean_loader = DataLoader(
        clean_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == 'cuda')
    )

    print('\nEvaluating ALL provided test images...')
    all_rows, _ = evaluate(model, all_loader, device, collect_examples=False)
    all_summary = summarize(all_rows)

    print('Evaluating CLEAN non-overlap test subset...')
    clean_rows, examples = evaluate(model, clean_loader, device, collect_examples=True)
    clean_summary = summarize(clean_rows)

    comparison = {
        'checkpoint': str(args.checkpoint),
        'overlap_definition': 'pixel-identical after the exact grayscale + bilinear resize preprocessing used by the model',
        'image_size': image_size,
        'n_train_images': len(train_imgs),
        'n_test_images_total': len(test_ds),
        'n_test_images_excluded_overlap': len(excluded_rows),
        'n_test_images_clean': len(clean_indices),
        'all_test': all_summary,
        'clean_test': clean_summary,
        'delta_clean_minus_all': {
            'mean_full_psnr_db': clean_summary['mean_full_psnr_db'] - all_summary['mean_full_psnr_db'],
            'mean_mask_psnr_db': clean_summary['mean_mask_psnr_db'] - all_summary['mean_mask_psnr_db'],
            'mean_nonmask_psnr_db': clean_summary['mean_nonmask_psnr_db'] - all_summary['mean_nonmask_psnr_db'],
            'mean_ssim': (clean_summary['mean_ssim'] - all_summary['mean_ssim']) if clean_summary['mean_ssim'] is not None and all_summary['mean_ssim'] is not None else None,
        },
        'note': 'No retraining was performed. The same best checkpoint was evaluated on the provided test set and on a leakage-filtered subset.'
    }

    save_csv(out / 'all_test_metrics.csv', all_rows)
    save_csv(out / 'clean_test_metrics.csv', clean_rows)
    (out / 'clean_test_comparison_summary.json').write_text(
        json.dumps(comparison, indent=2), encoding='utf-8'
    )
    save_examples(out / 'clean_test_reconstruction_examples.png', examples)

    print('\n=== ALL TEST ===')
    print(json.dumps(all_summary, indent=2))
    print('\n=== CLEAN TEST ===')
    print(json.dumps(clean_summary, indent=2))
    print('\n=== CLEAN - ALL DELTA ===')
    print(json.dumps(comparison['delta_clean_minus_all'], indent=2))
    print(f'\nSaved outputs: {out}')


if __name__ == '__main__':
    main()
