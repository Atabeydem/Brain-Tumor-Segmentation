from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw, ImageOps

eval_root = Path(r".\outputs\brisc_healthy_unet_ae_256_base64_clean_finetune_freezeBN_finalpolish_e200_lr2e7\eval_no_tumor_test")
fig_root = Path(r".\outputs\FINAL_RECONSTRUCTION_MODEL\figures")

selected_csv = fig_root / "selected_examples.csv"
orig_dir = eval_root / "original"
recon_dir = eval_root / "reconstructed"

out_path = fig_root / "selected_reconstruction_examples_clean_original_reconstructed.png"

df = pd.read_csv(selected_csv)

def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

orig_files = list_images(orig_dir)
recon_files = list_images(recon_dir)

cell_w = 300
cell_h = 350
label_h = 70
cols = 2
rows = len(df)

sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

draw.text((10, 8), "Original", fill="black")
draw.text((cell_w + 10, 8), "Reconstructed", fill="black")

def prepare_img(path):
    img = Image.open(path).convert("L")
    img = ImageOps.contain(img, (cell_w, cell_w))
    canvas = Image.new("RGB", (cell_w, cell_w), "white")
    x = (cell_w - img.width) // 2
    y = (cell_w - img.height) // 2
    canvas.paste(img.convert("RGB"), (x, y))
    return canvas

for r, row in df.iterrows():
    idx = int(row["index"])
    group = str(row["group"])
    psnr = float(row["psnr_db"])
    ssim = float(row["ssim"])

    orig_img = prepare_img(orig_files[idx])
    recon_img = prepare_img(recon_files[idx])

    y_label = r * cell_h + 28
    y_img = r * cell_h + label_h

    label = f"{group} | idx={idx} | PSNR={psnr:.2f} dB | SSIM={ssim:.6f}"
    draw.text((10, y_label), label, fill="black")

    sheet.paste(orig_img, (0, y_img))
    sheet.paste(recon_img, (cell_w, y_img))

sheet.save(out_path)

print("Clean contact sheet saved:")
print(out_path)
