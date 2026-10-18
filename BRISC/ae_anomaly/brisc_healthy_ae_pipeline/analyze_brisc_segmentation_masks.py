from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image

dataset_root = Path(r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")
img_dir = dataset_root / "segmentation_task" / "train" / "images"
mask_dir = dataset_root / "segmentation_task" / "train" / "masks"

out_dir = Path(r".\outputs\BRISC_MASKED_INPAINTING_AE\data_stats")
out_dir.mkdir(parents=True, exist_ok=True)

exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in exts])
mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in exts])
mask_map = {p.stem: p for p in mask_files}

pairs = []
for img_path in img_files:
    mask_path = mask_map.get(img_path.stem)
    if mask_path is not None:
        pairs.append((img_path, mask_path))

if len(pairs) == 0:
    pairs = list(zip(img_files, mask_files))

rows = []

for img_path, mask_path in pairs:
    mask = Image.open(mask_path).convert("L")
    arr = np.array(mask)
    mask_bin = arr > 0

    h, w = mask_bin.shape
    area = int(mask_bin.sum())
    total = int(h * w)
    area_ratio = area / total

    name = img_path.name.lower()

    if "_gl_" in name:
        tumor_type = "glioma"
    elif "_me_" in name:
        tumor_type = "meningioma"
    elif "_pi_" in name:
        tumor_type = "pituitary"
    else:
        tumor_type = "unknown"

    if "_ax_" in name:
        plane = "axial"
    elif "_co_" in name:
        plane = "coronal"
    elif "_sa_" in name:
        plane = "sagittal"
    else:
        plane = "unknown"

    rows.append({
        "image": img_path.name,
        "mask": mask_path.name,
        "height": h,
        "width": w,
        "mask_area_px": area,
        "image_area_px": total,
        "mask_area_ratio": area_ratio,
        "tumor_type": tumor_type,
        "plane": plane,
    })

df = pd.DataFrame(rows)
df.to_csv(out_dir / "segmentation_train_mask_stats.csv", index=False)

summary = {
    "n_pairs": len(df),
    "mask_area_ratio_mean": df["mask_area_ratio"].mean(),
    "mask_area_ratio_median": df["mask_area_ratio"].median(),
    "mask_area_ratio_min": df["mask_area_ratio"].min(),
    "mask_area_ratio_max": df["mask_area_ratio"].max(),
    "mask_area_ratio_p10": df["mask_area_ratio"].quantile(0.10),
    "mask_area_ratio_p90": df["mask_area_ratio"].quantile(0.90),
}

print("Pair count:", len(df))
print()
print("Mask area ratio summary:")
for k, v in summary.items():
    print(f"{k}: {v}")
print()
print("Tumor type distribution:")
print(df["tumor_type"].value_counts())
print()
print("Plane distribution:")
print(df["plane"].value_counts())
print()
print("Saved:")
print(out_dir / "segmentation_train_mask_stats.csv")
