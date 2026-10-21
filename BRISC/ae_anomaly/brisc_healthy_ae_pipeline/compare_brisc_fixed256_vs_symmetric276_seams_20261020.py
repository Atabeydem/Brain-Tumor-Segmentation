#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Paired healthy-test audit: Fixed-256 vs Symmetric-276 reconstruction seam error.

READ ONLY: No training, no model edits, no dataset writes.
Compares the same healthy TEST images using their original native geometry.
Reports pixelwise squared error near internal 256px tile boundaries AND
how accurately the two reconstructed pixel values across the seam preserve
an actual native-image adjacent-pixel difference (jump error).
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

import train_eval_brisc_healthy_fixed256_resnet_20261011 as old
import train_eval_brisc_healthy_symmetric276_resnet_20261019 as sym


def arguments():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--fixed_checkpoint', type=Path, required=True)
    ap.add_argument('--symmetric_checkpoint', type=Path, required=True)
    ap.add_argument('--output_dir', type=Path, required=True)
    ap.add_argument('--band', type=int, default=10, help='Seam band in pixels on each side')
    ap.add_argument('--max_images', type=int, default=0, help='0 = all healthy TEST images')
    ap.add_argument('--examples', type=int, default=8)
    return ap.parse_args()


def load_model(module, filename, device):
    ckpt = torch.load(filename, map_location='cpu', weights_only=True)
    if 'model_state_dict' not in ckpt:
        raise KeyError(f'Not a compatible checkpoint: {filename}')
    base = int(ckpt.get('base_channels', 64))
    if base != 64:
        raise ValueError(f'Expected base_channels=64, got {base} in {filename}')
    model = module.NativeResNetAutoencoder(base_channels=base)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def reconstruct(image, rows, model, device, symmetric):
    out = np.full(image.shape, np.nan, dtype=np.float32)
    for row in rows:
        x0, y0 = int(row['x0']), int(row['y0'])
        x1, y1 = int(row['x1_valid']), int(row['y1_valid'])
        patch = image[y0:y1, x0:x1]
        ph, pw = patch.shape
        if (ph, pw) != (int(row['valid_height']), int(row['valid_width'])):
            raise AssertionError('Manifest/patch geometry mismatch')
        if symmetric:
            tile = sym.pad_context_276(sym.pad_patch_symmetric(patch))
        else:
            tile = old.pad_patch_replicate(patch)
        x = torch.from_numpy(tile.copy()).float().unsqueeze(0).unsqueeze(0).to(device)
        y = sym.forward_276(model, x) if symmetric else model(x)
        y = torch.clamp(y, 0, 1)[0, 0].cpu().numpy()
        pred = y[10:10+ph, 10:10+pw] if symmetric else y[:ph, :pw]
        out[y0:y1, x0:x1] = pred
    if not np.isfinite(out).all():
        raise AssertionError('Some image pixels were not reconstructed')
    return out


def make_seam_masks(shape, band):
    h, w = shape
    seam = np.zeros((h, w), dtype=bool)
    if w > 256:
        seam[:, max(0, 256-band):min(w, 256+band)] = True
    if h > 256:
        seam[max(0, 256-band):min(h, 256+band), :] = True
    return seam


def adjacent_jump_error(original, reconstructed):
    """Mean |pred difference - true difference| at internal tile boundaries.

    This measures extra discontinuity caused by processing tiles separately;
    natural anatomical intensity jumps at the boundary are subtracted out.
    """
    h, w = original.shape
    vals = []
    if w > 256:
        pred = reconstructed[:, 256] - reconstructed[:, 255]
        true = original[:, 256] - original[:, 255]
        vals.append(np.abs(pred - true))
    if h > 256:
        pred = reconstructed[256, :] - reconstructed[255, :]
        true = original[256, :] - original[255, :]
        vals.append(np.abs(pred - true))
    if not vals:
        return None
    return float(sum(v.sum() for v in vals) / sum(v.size for v in vals))


def per_image(original, reconstructed, band):
    err = (original - reconstructed).astype(np.float64)
    sq = err ** 2
    seam = make_seam_masks(original.shape, band)
    valid = bool(np.any(seam))
    inner = ~seam
    result = {
        'whole_mse': float(sq.mean()),
        'whole_psnr_db': float(-10 * np.log10(max(float(sq.mean()), 1e-20))),
        'seam_mse': float(sq[seam].mean()) if valid else None,
        'interior_mse': float(sq[inner].mean()) if np.any(inner) else None,
        'jump_mae': adjacent_jump_error(original, reconstructed),
    }
    return result


def save_examples(rows, out_dir, n):
    if not rows or n <= 0: return
    n = min(len(rows), n)
    panel_size = 256
    scale = 2
    gap = 15
    hrow = panel_size + 40
    canvas = Image.new('RGB', (panel_size * 5 + gap * 6, hrow*n+25), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (name, original, fixed, symmetric) in enumerate(rows[:n]):
        errors = [np.abs(original-fixed), np.abs(original-symmetric)]
        # SAME color scale for both error maps within the same source image.
        cap = max(float(np.percentile(np.concatenate([e.ravel() for e in errors]), 99.5)), 1e-8)
        panels = [original, fixed, symmetric, np.clip(errors[0]/cap, 0, 1), np.clip(errors[1]/cap, 0, 1)]
        ytop = i*hrow+25
        for j, p in enumerate(panels):
            img = Image.fromarray(np.uint8(np.clip(p, 0, 1)*255), mode='L').convert('RGB')
            img.thumbnail((panel_size, panel_size))
            x = gap + j*(panel_size+gap)
            canvas.paste(img, (x, ytop + 20))
            # True 256px tile boundary, if it exists in this image.
            if original.shape[1] > 256:
                bx = int(round(256*img.width/original.shape[1]))
                draw.line((x+bx, ytop+20, x+bx, ytop+20+img.height), fill=(255, 190, 0), width=1)
            if original.shape[0] > 256:
                by = int(round(256*img.height/original.shape[0]))
                draw.line((x, ytop+20+by, x+img.width, ytop+20+by), fill=(255, 190, 0), width=1)
        draw.text((12, ytop+1), name[:65], fill=(20,20,20))
    for j, name in enumerate(('Original', 'Fixed-256', 'Symmetric-276', 'Fixed abs error', 'Symmetric abs error')):
        draw.text((gap+j*(panel_size+gap), 5), name, fill=(20,20,20))
    canvas.save(out_dir/'paired_seam_examples.png')


def avg(items, key):
    x = [r[key] for r in items if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(x)) if x else None


def main():
    a = arguments()
    for path in (a.manifest, a.fixed_checkpoint, a.symmetric_checkpoint):
        if not path.is_file(): raise FileNotFoundError(path)
    if not 1 <= a.band <= 32: raise ValueError('--band must be between 1 and 32')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [r for r in sym.read_manifest(a.manifest) if r['split']=='healthy_test']
    grouped = defaultdict(list)
    for r in rows: grouped[r['source_image']].append(r)
    if not grouped: raise ValueError('No healthy_test in manifest')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fixed_model = load_model(old, a.fixed_checkpoint, device)
    sym_model = load_model(sym, a.symmetric_checkpoint, device)
    items = sorted(grouped.items())
    if a.max_images > 0: items = items[:a.max_images]
    print(f'Paired healthy TEST images: {len(items)} | device={device} | seam half-band={a.band}px')
    output = []
    figures = []
    for i, (filename, image_rows) in enumerate(items):
        source = sym.load_native_gray(Path(filename))
        fixed_recon = reconstruct(source, image_rows, fixed_model, device, symmetric=False)
        sym_recon = reconstruct(source, image_rows, sym_model, device, symmetric=True)
        fix = per_image(source, fixed_recon, a.band)
        sy = per_image(source, sym_recon, a.band)
        row = {'image': Path(filename).name, 'height': source.shape[0], 'width': source.shape[1],
               'has_internal_seam': int(source.shape[0]>256 or source.shape[1]>256)}
        for key, metrics in (('fixed',fix),('symmetric',sy)):
            row.update({f'{key}_{k}':v for k,v in metrics.items()})
        for k in ('whole_mse','whole_psnr_db','seam_mse','interior_mse','jump_mae'):
            f = row[f'fixed_{k}']; s = row[f'symmetric_{k}']
            row[f'delta_{k}'] = s-f if (s is not None and f is not None) else None
        output.append(row)
        if len(figures)<a.examples and row['has_internal_seam']:
            figures.append((row['image'],source,fixed_recon,sym_recon))
        if (i+1)%25==0 or i+1==len(items):print(f'  processed {i+1}/{len(items)}')
    with (a.output_dir/'per_image_seam_comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    seam_items=[r for r in output if r['has_internal_seam']]
    full512_items=[r for r in output if r['height']==512 and r['width']==512]
    summary={'n_healthy_test_images':len(output), 'n_images_with_internal_seam':len(seam_items),
             'seam_half_band_px':a.band, 'delta_definition':'symmetric minus fixed; negative MSE/jump means improvement',
             'seam_jump_definition':'mean abs(error in reconstructed pixel jump across tile seam)',
             'full512_only_n':len(full512_items),
             'full512_only_mean_seam_jump_mae_fixed':avg(full512_items,'fixed_jump_mae'),
             'full512_only_mean_seam_jump_mae_symmetric':avg(full512_items,'symmetric_jump_mae'),
             'full512_only_mean_seam_mse_fixed':avg(full512_items,'fixed_seam_mse'),
             'full512_only_mean_seam_mse_symmetric':avg(full512_items,'symmetric_seam_mse'),
             'whole_image_mean_psnr_db_fixed':avg(output,'fixed_whole_psnr_db'),
             'whole_image_mean_psnr_db_symmetric':avg(output,'symmetric_whole_psnr_db'),
             'mean_seam_mse_fixed':avg(seam_items,'fixed_seam_mse'),
             'mean_seam_mse_symmetric':avg(seam_items,'symmetric_seam_mse'),
             'mean_interior_mse_fixed':avg(seam_items,'fixed_interior_mse'),
             'mean_interior_mse_symmetric':avg(seam_items,'symmetric_interior_mse'),
             'mean_seam_jump_mae_fixed':avg(seam_items,'fixed_jump_mae'),
             'mean_seam_jump_mae_symmetric':avg(seam_items,'symmetric_jump_mae')}
    for metric in ('mean_seam_mse','mean_interior_mse','mean_seam_jump_mae'):
        x=summary[f'{metric}_fixed']; y=summary[f'{metric}_symmetric']
        summary[f'{metric}_relative_change_pct']=float(100*(y/x-1)) if x else None
    (a.output_dir/'seam_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    save_examples(figures,a.output_dir,a.examples)
    print(json.dumps(summary,indent=2))
    print('Saved to:',a.output_dir.resolve())

if __name__=='__main__': main()
