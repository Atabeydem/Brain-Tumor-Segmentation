#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import hashlib
import math
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

try:
    from skimage.metrics import peak_signal_noise_ratio
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted([p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS])


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open('rb') as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def decoded_gray_hash(path: Path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L'), dtype=np.uint8)
    h = hashlib.sha256()
    h.update(str(arr.shape).encode('utf-8'))
    h.update(arr.tobytes())
    return h.hexdigest()


def stem_map(paths):
    d = {}
    dup = []
    for p in paths:
        if p.stem in d:
            dup.append((p.stem, d[p.stem], p))
        else:
            d[p.stem] = p
    return d, dup


def mask_nonempty(path: Path):
    with Image.open(path) as im:
        arr = np.asarray(im.convert('L'))
    return bool(np.any(arr > 0)), float(np.mean(arr > 0)), arr.shape


def image_meta(path: Path):
    with Image.open(path) as im:
        return im.mode, im.size


def manual_psnr(a, b, data_range=1.0):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    if mse == 0.0:
        return float('inf'), mse
    return 10.0 * math.log10((data_range ** 2) / mse), mse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_root', required=True)
    ap.add_argument('--hash_mode', choices=['raw', 'decoded', 'both'], default='both')
    args = ap.parse_args()

    root = Path(args.dataset_root)
    tr_img_dir = root / 'segmentation_task' / 'train' / 'images'
    tr_msk_dir = root / 'segmentation_task' / 'train' / 'masks'
    te_img_dir = root / 'segmentation_task' / 'test' / 'images'
    te_msk_dir = root / 'segmentation_task' / 'test' / 'masks'

    tr_imgs = list_images(tr_img_dir)
    tr_msks = list_images(tr_msk_dir)
    te_imgs = list_images(te_img_dir)
    te_msks = list_images(te_msk_dir)

    print('=' * 80)
    print('BRISC TUMOR PIPELINE - SANITY CHECK')
    print('=' * 80)
    print(f'Dataset root: {root}')

    print('\n[1] FILE COUNTS')
    print(f'Train images : {len(tr_imgs)}')
    print(f'Train masks  : {len(tr_msks)}')
    print(f'Test images  : {len(te_imgs)}')
    print(f'Test masks   : {len(te_msks)}')

    tr_img_map, tr_img_dups = stem_map(tr_imgs)
    tr_msk_map, tr_msk_dups = stem_map(tr_msks)
    te_img_map, te_img_dups = stem_map(te_imgs)
    te_msk_map, te_msk_dups = stem_map(te_msks)

    print('\n[2] IMAGE-MASK PAIRING')
    train_missing_masks = sorted(set(tr_img_map) - set(tr_msk_map))
    train_orphan_masks = sorted(set(tr_msk_map) - set(tr_img_map))
    test_missing_masks = sorted(set(te_img_map) - set(te_msk_map))
    test_orphan_masks = sorted(set(te_msk_map) - set(te_img_map))

    print(f'Train matched pairs : {len(set(tr_img_map) & set(tr_msk_map))}')
    print(f'Train images w/o mask: {len(train_missing_masks)}')
    print(f'Train masks w/o image: {len(train_orphan_masks)}')
    print(f'Test matched pairs  : {len(set(te_img_map) & set(te_msk_map))}')
    print(f'Test images w/o mask: {len(test_missing_masks)}')
    print(f'Test masks w/o image: {len(test_orphan_masks)}')
    print(f'Duplicate train image stems: {len(tr_img_dups)}')
    print(f'Duplicate test image stems : {len(te_img_dups)}')
    print(f'Duplicate train mask stems : {len(tr_msk_dups)}')
    print(f'Duplicate test mask stems  : {len(te_msk_dups)}')

    print('\n[3] TRAIN/TEST NAME OVERLAP')
    exact_name_overlap = sorted(set(p.name for p in tr_imgs) & set(p.name for p in te_imgs))
    stem_overlap = sorted(set(tr_img_map) & set(te_img_map))
    print(f'Exact filename overlap: {len(exact_name_overlap)}')
    print(f'Stem overlap          : {len(stem_overlap)}')
    if exact_name_overlap[:10]:
        print('Examples:', exact_name_overlap[:10])

    print('\n[4] TRAIN/TEST CONTENT HASH OVERLAP')
    raw_overlap = []
    dec_overlap = []
    if args.hash_mode in ('raw', 'both'):
        train_raw = {}
        for i, p in enumerate(tr_imgs, 1):
            train_raw.setdefault(sha256_file(p), []).append(p)
            if i % 500 == 0:
                print(f'  raw train hashed {i}/{len(tr_imgs)}')
        for i, p in enumerate(te_imgs, 1):
            h = sha256_file(p)
            if h in train_raw:
                raw_overlap.append((p, train_raw[h]))
            if i % 250 == 0:
                print(f'  raw test hashed {i}/{len(te_imgs)}')
        print(f'Exact byte-identical train/test images: {len(raw_overlap)}')
        for p, matches in raw_overlap[:10]:
            print(f'  TEST {p.name} == TRAIN {[m.name for m in matches[:3]]}')

    if args.hash_mode in ('decoded', 'both'):
        train_dec = {}
        for i, p in enumerate(tr_imgs, 1):
            train_dec.setdefault(decoded_gray_hash(p), []).append(p)
            if i % 500 == 0:
                print(f'  decoded train hashed {i}/{len(tr_imgs)}')
        for i, p in enumerate(te_imgs, 1):
            h = decoded_gray_hash(p)
            if h in train_dec:
                dec_overlap.append((p, train_dec[h]))
            if i % 250 == 0:
                print(f'  decoded test hashed {i}/{len(te_imgs)}')
        print(f'Pixel-identical grayscale train/test images: {len(dec_overlap)}')
        for p, matches in dec_overlap[:10]:
            print(f'  TEST {p.name} == TRAIN {[m.name for m in matches[:3]]}')

    print('\n[5] MASK SANITY')
    tr_empty = 0
    te_empty = 0
    tr_ratios = []
    te_ratios = []
    shape_mismatch = []

    for stem in sorted(set(tr_img_map) & set(tr_msk_map)):
        nonempty, ratio, mshape = mask_nonempty(tr_msk_map[stem])
        tr_ratios.append(ratio)
        if not nonempty:
            tr_empty += 1
        with Image.open(tr_img_map[stem]) as im:
            ishape = (im.height, im.width)
        if ishape != mshape:
            shape_mismatch.append(('train', stem, ishape, mshape))

    for stem in sorted(set(te_img_map) & set(te_msk_map)):
        nonempty, ratio, mshape = mask_nonempty(te_msk_map[stem])
        te_ratios.append(ratio)
        if not nonempty:
            te_empty += 1
        with Image.open(te_img_map[stem]) as im:
            ishape = (im.height, im.width)
        if ishape != mshape:
            shape_mismatch.append(('test', stem, ishape, mshape))

    print(f'Empty train masks: {tr_empty}')
    print(f'Empty test masks : {te_empty}')
    if tr_ratios:
        print(f'Train mask area mean: {np.mean(tr_ratios):.6f} ({np.mean(tr_ratios)*100:.3f}%)')
        print(f'Train mask area min/max: {np.min(tr_ratios):.6f} / {np.max(tr_ratios):.6f}')
    if te_ratios:
        print(f'Test mask area mean : {np.mean(te_ratios):.6f} ({np.mean(te_ratios)*100:.3f}%)')
        print(f'Test mask area min/max : {np.min(te_ratios):.6f} / {np.max(te_ratios):.6f}')
    print(f'Image/mask spatial shape mismatches: {len(shape_mismatch)}')
    for x in shape_mismatch[:10]:
        print(' ', x)

    print('\n[6] IMAGE FORMAT / SIZE SUMMARY')
    mode_counts = Counter()
    size_counts = Counter()
    for p in tr_imgs + te_imgs:
        mode, size = image_meta(p)
        mode_counts[mode] += 1
        size_counts[size] += 1
    print('Modes:', dict(mode_counts))
    print('Most common sizes:')
    for size, n in size_counts.most_common(10):
        print(f'  {size}: {n}')

    print('\n[7] PSNR FORMULA UNIT TEST')
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.full((64, 64), 0.01, dtype=np.float32)
    psnr_manual, mse = manual_psnr(a, b, data_range=1.0)
    print(f'Synthetic MSE (expected 1e-4): {mse:.12e}')
    print(f'Manual PSNR (expected 40 dB): {psnr_manual:.8f} dB')
    if HAS_SKIMAGE:
        psnr_sk = float(peak_signal_noise_ratio(a, b, data_range=1.0))
        print(f'skimage PSNR                 : {psnr_sk:.8f} dB')
        print(f'Difference                   : {abs(psnr_manual-psnr_sk):.12e}')
    else:
        print('skimage not installed -> manual formula only.')

    print('\n[8] SUMMARY')
    checks = {
        'train_count_3933': len(tr_imgs) == 3933,
        'test_count_860': len(te_imgs) == 860,
        'train_all_paired': len(train_missing_masks) == 0 and len(train_orphan_masks) == 0,
        'test_all_paired': len(test_missing_masks) == 0 and len(test_orphan_masks) == 0,
        'no_name_overlap': len(exact_name_overlap) == 0 and len(stem_overlap) == 0,
        'no_raw_hash_overlap': len(raw_overlap) == 0 if args.hash_mode in ('raw', 'both') else True,
        'no_decoded_hash_overlap': len(dec_overlap) == 0 if args.hash_mode in ('decoded', 'both') else True,
        'no_empty_masks': tr_empty == 0 and te_empty == 0,
        'psnr_formula_ok': abs(psnr_manual - 40.0) < 1e-5,
    }
    for k, v in checks.items():
        print(f"{'[PASS]' if v else '[CHECK]'} {k}")

    print('\nNOTE:')
    print('- This checks obvious file-level leakage, pairing, masks, and the PSNR formula.')
    print('- It does NOT prove patient-level independence unless patient IDs are encoded')
    print('  in filenames/metadata or documented by the dataset.')
    print('- Highly similar slices can still exist without being pixel-identical.')


if __name__ == '__main__':
    main()
