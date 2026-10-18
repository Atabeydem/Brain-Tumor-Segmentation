#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True), nn.ReLU(inplace=True),
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
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up_to(x1_2, x0_0)], dim=1))
        return self.out(x0_3)

def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device)
    image_size = int(ckpt.get('image_size', 256))
    base_channels = int(ckpt.get('base_channels', 32))
    model = UNetPlusPlusAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()
    return model, image_size, base_channels

def list_images(folder: Path):
    return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def load_gray_tensor(path: Path, size: int, nearest=False):
    with Image.open(path) as im:
        im = im.convert('L')
        resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
        im = im.resize((size, size), resample=resample)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

def processed_hash(path: Path, size: int):
    with Image.open(path) as im:
        im = im.convert('L').resize((size, size), resample=Image.Resampling.BILINEAR)
        arr = np.asarray(im, dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('utf-8'))
    h.update(arr.tobytes())
    return h.hexdigest()

def match_by_stem(images, masks):
    mask_map = {p.stem: p for p in masks}
    return [(p, mask_map[p.stem]) for p in images if p.stem in mask_map]

def binary_metrics(pred, gt, eps=1e-8):
    pred = pred.astype(bool); gt = gt.astype(bool)
    tp = np.logical_and(pred, gt).sum(dtype=np.float64)
    fp = np.logical_and(pred, ~gt).sum(dtype=np.float64)
    fn = np.logical_and(~pred, gt).sum(dtype=np.float64)
    return {
        'dice': float((2*tp + eps)/(2*tp + fp + fn + eps)),
        'iou': float((tp + eps)/(tp + fp + fn + eps)),
        'precision': float((tp + eps)/(tp + fp + eps)),
        'recall': float((tp + eps)/(tp + fn + eps)),
    }

def write_csv(path, rows):
    if not rows: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def arr_to_pil(arr01):
    return Image.fromarray((np.clip(arr01,0,1)*255).astype(np.uint8), mode='L')

def save_contact_sheet(rows, labels, out_path, cell=160):
    if not rows: return
    margin, label_h, name_h = 10, 24, 20
    cols = len(labels)
    width = margin + cols*(cell+margin)
    height = label_h + margin + len(rows)*(cell+name_h+margin)
    canvas = Image.new('RGB',(width,height),'white'); draw = ImageDraw.Draw(canvas)
    for c, lab in enumerate(labels): draw.text((margin+c*(cell+margin),3), lab, fill='black')
    y = label_h + margin
    for name, ims in rows:
        draw.text((margin,y), name, fill='black'); yy=y+name_h
        for c, im in enumerate(ims):
            canvas.paste(im.convert('RGB').resize((cell,cell)), (margin+c*(cell+margin),yy))
        y=yy+cell+margin
    canvas.save(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--thresholds', nargs='+', type=float, default=[0.005,0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.14,0.16,0.18,0.20,0.25,0.30])
    ap.add_argument('--save_examples', type=int, default=12)
    args = ap.parse_args()

    root = Path(args.dataset_root); out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, image_size, base_channels = load_checkpoint(Path(args.checkpoint), device)

    train_images = list_images(root/'segmentation_task'/'train'/'images')
    test_images = list_images(root/'segmentation_task'/'test'/'images')
    test_masks = list_images(root/'segmentation_task'/'test'/'masks')
    test_pairs = match_by_stem(test_images, test_masks)

    print(f'Device: {device}')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'Image size: {image_size}')
    print(f'Base channels: {base_channels}')
    print(f'Train images: {len(train_images)} | Test pairs: {len(test_pairs)}')
    print('\nBuilding post-preprocessing train hashes (grayscale + resize)...')
    train_hashes=set()
    for i,p in enumerate(train_images,1):
        train_hashes.add(processed_hash(p,image_size))
        if i%500==0: print(f'  train hashed {i}/{len(train_images)}')

    clean_pairs=[]; overlap_pairs=[]
    for img,msk in test_pairs:
        (overlap_pairs if processed_hash(img,image_size) in train_hashes else clean_pairs).append((img,msk))
    print(f'\nPost-preprocessing overlap test images: {len(overlap_pairs)}')
    print(f'Clean test images: {len(clean_pairs)}')
    write_csv(out_dir/'excluded_overlap_test_images.csv',[{'image':i.name,'mask':m.name} for i,m in overlap_pairs])

    metrics_by_th={th:[] for th in args.thresholds}; per_case=[]; examples=[]
    print('\nEvaluating CLEAN subset with PER-IMAGE NORMALIZED residual...')
    with torch.no_grad():
        for idx,(img_path,mask_path) in enumerate(clean_pairs):
            x = load_gray_tensor(img_path,image_size,False).unsqueeze(0).to(device)
            gt = load_gray_tensor(mask_path,image_size,True).squeeze().numpy()>0.5
            recon = torch.clamp(model(x),0,1)
            orig_np = x.detach().cpu().squeeze().numpy(); recon_np = recon.detach().cpu().squeeze().numpy()
            err = np.abs(orig_np-recon_np); err_norm = err/(float(err.max())+1e-8)
            for th in args.thresholds:
                m=binary_metrics(err_norm>=th,gt); metrics_by_th[th].append(m)
                per_case.append({'image':img_path.name,'threshold':th,'raw_error_max':float(err.max()),**m})
            if len(examples)<args.save_examples:
                examples.append((img_path.name,orig_np,recon_np,err_norm,gt))
            if (idx+1)%100==0: print(f'Processed {idx+1}/{len(clean_pairs)}')

    threshold_rows=[]
    for th,ms in metrics_by_th.items():
        threshold_rows.append({'threshold':th,'n_cases':len(ms),'mean_dice':float(np.mean([m['dice'] for m in ms])),'mean_iou':float(np.mean([m['iou'] for m in ms])),'mean_precision':float(np.mean([m['precision'] for m in ms])),'mean_recall':float(np.mean([m['recall'] for m in ms]))})
    threshold_rows=sorted(threshold_rows,key=lambda r:r['mean_dice'],reverse=True)
    best=threshold_rows[0]
    write_csv(out_dir/'clean_residual_per_threshold.csv',threshold_rows)
    write_csv(out_dir/'clean_residual_per_case.csv',per_case)

    best_th=float(best['threshold']); contact=[]
    for name,orig,recon,err_norm,gt in examples:
        pred=err_norm>=best_th
        contact.append((name,[arr_to_pil(orig),arr_to_pil(recon),arr_to_pil(err_norm),Image.fromarray((pred.astype(np.uint8)*255),mode='L'),Image.fromarray((gt.astype(np.uint8)*255),mode='L')]))
    save_contact_sheet(contact,['Original','Reconstructed','Norm Error',f'Pred th={best_th:g}','GT Mask'],out_dir/'clean_residual_examples_contact_sheet.png')

    summary={
        'experiment':'tumor_self_reconstruction_clean_residual_localization',
        'checkpoint':str(args.checkpoint),'model':'UNetPlusPlusAutoencoder','image_size':image_size,'base_channels':base_channels,
        'provided_test_n':len(test_pairs),'excluded_post_preprocessing_overlap_n':len(overlap_pairs),'clean_test_n':len(clean_pairs),
        'residual_definition':'abs(original - clamp(reconstruction,0,1))',
        'normalization':'per-image error / (per-image max error + 1e-8)',
        'best':best,
        'note':'No retraining. Same tumor self-reconstruction checkpoint; only post-preprocessing non-overlap test subset. Threshold is selected on this same clean subset, so this is exploratory localization analysis, not a locked external-test estimate.'
    }
    (out_dir/'clean_residual_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\n=== CLEAN RESIDUAL LOCALIZATION ===')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    print('\nTop 5 thresholds by mean Dice:')
    for row in threshold_rows[:5]: print(row)
    print(f'\nSaved outputs: {out_dir}')

if __name__=='__main__': main()
