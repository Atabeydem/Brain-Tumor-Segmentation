from pathlib import Path
import random
import numpy as np
from PIL import Image, ImageDraw, ImageOps

dataset_root = Path(r"C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025")
img_dir = dataset_root / "segmentation_task" / "train" / "images"
mask_dir = dataset_root / "segmentation_task" / "train" / "masks"

out_dir = Path(r".\outputs\BRISC_MASKED_INPAINTING_AE\preview")
out_dir.mkdir(parents=True, exist_ok=True)

image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

img_files = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in image_exts])
mask_files = sorted([p for p in mask_dir.iterdir() if p.suffix.lower() in image_exts])

mask_map = {p.stem: p for p in mask_files}

pairs = []
for img_path in img_files:
    mask_path = mask_map.get(img_path.stem)
    if mask_path is not None:
        pairs.append((img_path, mask_path))

# Eğer stem eşleşmesi yoksa sıralı eşleştirme fallback
if len(pairs) == 0:
    print("Stem-based eşleşme bulunamadı. Sorted-order fallback kullanılıyor.")
    pairs = list(zip(img_files, mask_files))

print(f"Images found: {len(img_files)}")
print(f"Masks found: {len(mask_files)}")
print(f"Pairs found: {len(pairs)}")

if len(pairs) == 0:
    raise RuntimeError("Hiç image-mask pair bulunamadı. Klasör yollarını kontrol et.")

random.seed(42)
selected = random.sample(pairs, min(8, len(pairs)))

cell = 256
label_h = 36
cols = 4
rows = len(selected)

sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
draw = ImageDraw.Draw(sheet)

headers = ["Original", "GT Mask", "Masked Input", "Overlay"]
for c, h in enumerate(headers):
    draw.text((c * cell + 8, 6), h, fill="black")

def prepare_gray(path):
    img = Image.open(path).convert("L")
    img = ImageOps.contain(img, (cell, cell))
    canvas = Image.new("L", (cell, cell), 0)
    x = (cell - img.width) // 2
    y = (cell - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas

for r, (img_path, mask_path) in enumerate(selected):
    img = prepare_gray(img_path)
    mask = prepare_gray(mask_path)

    img_arr = np.array(img).astype(np.uint8)
    mask_arr = np.array(mask).astype(np.uint8)
    mask_bin = mask_arr > 0

    # Masked input: tümör bölgesini siyaha çekiyoruz.
    # Bu sadece preview için. Eğitimde farklı maskeleme stratejileri deneyebiliriz.
    masked_arr = img_arr.copy()
    masked_arr[mask_bin] = 0

    masked = Image.fromarray(masked_arr).convert("RGB")

    mask_vis_arr = np.zeros((cell, cell), dtype=np.uint8)
    mask_vis_arr[mask_bin] = 255
    mask_vis = Image.fromarray(mask_vis_arr).convert("RGB")

    original_rgb = img.convert("RGB")

    overlay = original_rgb.copy()
    overlay_arr = np.array(overlay).astype(np.uint8)
    overlay_arr[mask_bin, 0] = 255
    overlay_arr[mask_bin, 1] = (overlay_arr[mask_bin, 1] * 0.35).astype(np.uint8)
    overlay_arr[mask_bin, 2] = (overlay_arr[mask_bin, 2] * 0.35).astype(np.uint8)
    overlay = Image.fromarray(overlay_arr)

    y0 = r * (cell + label_h) + label_h

    label = f"{r+1:02d} | {img_path.name}"
    draw.text((8, y0 - 22), label, fill="black")

    sheet.paste(original_rgb, (0, y0))
    sheet.paste(mask_vis, (cell, y0))
    sheet.paste(masked, (2 * cell, y0))
    sheet.paste(overlay, (3 * cell, y0))

sheet_path = out_dir / "masked_inpainting_preview_contact_sheet.png"
sheet.save(sheet_path)

print("Preview saved:")
print(sheet_path)
