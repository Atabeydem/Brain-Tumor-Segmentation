from pathlib import Path
from PIL import Image
import numpy as np
from collections import Counter
import random

ROOT = Path(r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")

paths = {
    "healthy_train": ROOT / "classification_task" / "train" / "no_tumor",
    "healthy_test": ROOT / "classification_task" / "test" / "no_tumor",
    "seg_train_images": ROOT / "segmentation_task" / "train" / "images",
    "seg_train_masks": ROOT / "segmentation_task" / "train" / "masks",
    "seg_test_images": ROOT / "segmentation_task" / "test" / "images",
    "seg_test_masks": ROOT / "segmentation_task" / "test" / "masks",
}

def image_files(folder):
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])

def inspect_folder(name, folder, max_sample=200):
    files = image_files(folder)
    sample = random.sample(files, min(max_sample, len(files))) if files else []

    sizes = []
    modes = []
    means = []
    stds = []

    for p in sample:
        img = Image.open(p)
        arr = np.array(img).astype(np.float32)
        sizes.append(img.size)
        modes.append(img.mode)
        means.append(arr.mean())
        stds.append(arr.std())

    print(f"\n{name}")
    print("-" * 60)
    print("folder:", folder)
    print("count:", len(files))
    print("size counts:", Counter(sizes).most_common(10))
    print("mode counts:", Counter(modes).most_common(10))
    if means:
        print("mean intensity:", round(float(np.mean(means)), 3))
        print("std intensity :", round(float(np.mean(stds)), 3))

for name, folder in paths.items():
    inspect_folder(name, folder)

# segmentation mask boş mu dolu mu kontrolü
print("\nMask foreground kontrolü")
print("-" * 60)

for split in ["train", "test"]:
    mask_dir = ROOT / "segmentation_task" / split / "masks"
    masks = image_files(mask_dir)
    non_empty = 0
    empty = 0

    for p in masks:
        arr = np.array(Image.open(p))
        if arr.max() > 0:
            non_empty += 1
        else:
            empty += 1

    print(f"{split}: non_empty={non_empty}, empty={empty}, total={len(masks)}")