from pathlib import Path
import shutil
import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageChops

root = Path(r".\outputs\brisc_healthy_unet_ae_256_base64_clean_finetune_freezeBN_finalpolish_e200_lr2e7\eval_no_tumor_test")
csv_path = root / "reconstruction_metrics.csv"
orig_dir = root / "original"
recon_dir = root / "reconstructed"

out_dir = root / "selected_representative_examples"
out_dir.mkdir(exist_ok=True)

df = pd.read_csv(csv_path)

# Düşük, orta, yüksek PSNR örnekleri
low = df.nsmallest(2, "psnr_db").copy()
low["group"] = "low_psnr"

high = df.nlargest(2, "psnr_db").copy()
high["group"] = "high_psnr"

used = set(low.index).union(set(high.index))
remaining = df.drop(index=list(used))
median_psnr = df["psnr_db"].median()
mid = remaining.iloc[(remaining["psnr_db"] - median_psnr).abs().argsort()[:2]].copy()
mid["group"] = "mid_psnr"

selected = pd.concat([low, mid, high], ignore_index=True)
selected = selected[["group", "index", "source_path", "mse", "mae", "psnr_db", "ssim"]]
selected.to_csv(out_dir / "selected_examples.csv", index=False)

def list_images(folder):
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

orig_files = list_images(orig_dir)
recon_files = list_images(recon_dir)

if len(orig_files) == 0 or len(recon_files) == 0:
    raise RuntimeError("original veya reconstructed klasöründe görüntü bulunamadı.")

cell_w = 256
cell_h = 300
label_h = 44
cols = 3
rows = len(selected)

sheet = Image.new("RGB", (cols * cell_w, rows * cell_h), "white")
draw = ImageDraw.Draw(sheet)

headers = ["Original", "Reconstructed", "Abs Diff\ncontrast enhanced"]
for c, h in enumerate(headers):
    draw.text((c * cell_w + 8, 4), h, fill="black")

def prepare_img(path):
    img = Image.open(path).convert("L")
    img = ImageOps.contain(img, (cell_w, cell_w))
    canvas = Image.new("RGB", (cell_w, cell_w), "white")
    x = (cell_w - img.width) // 2
    y = (cell_w - img.height) // 2
    canvas.paste(img.convert("RGB"), (x, y))
    return canvas

for r, row in selected.iterrows():
    idx = int(row["index"])
    group = row["group"]
    psnr = float(row["psnr_db"])
    ssim = float(row["ssim"])

    orig_path = orig_files[idx]
    recon_path = recon_files[idx]

    # Dosyaları ayrıca seçili klasöre kopyala
    safe_prefix = f"{r+1:02d}_{group}_idx{idx}_psnr{psnr:.2f}"
    shutil.copy2(orig_path, out_dir / f"{safe_prefix}_original{orig_path.suffix}")
    shutil.copy2(recon_path, out_dir / f"{safe_prefix}_reconstructed{recon_path.suffix}")

    orig_img = prepare_img(orig_path)
    recon_img = prepare_img(recon_path)

    o = Image.open(orig_path).convert("L").resize((cell_w, cell_w))
    rec = Image.open(recon_path).convert("L").resize((cell_w, cell_w))
    diff = ImageChops.difference(o, rec)
    diff = ImageOps.autocontrast(diff)
    diff_img = Image.new("RGB", (cell_w, cell_w), "white")
    diff_img.paste(diff.convert("RGB"), (0, 0))

    y0 = r * cell_h + label_h

    label = f"{group} | idx={idx} | PSNR={psnr:.2f} dB | SSIM={ssim:.6f}"
    draw.text((8, r * cell_h + 22), label, fill="black")

    sheet.paste(orig_img, (0, y0))
    sheet.paste(recon_img, (cell_w, y0))
    sheet.paste(diff_img, (2 * cell_w, y0))

sheet_path = out_dir / "selected_reconstruction_examples_contact_sheet.png"
sheet.save(sheet_path)

print("Selected examples saved to:")
print(out_dir)
print()
print("Contact sheet:")
print(sheet_path)
print()
print(selected[["group", "index", "psnr_db", "ssim", "mse"]])
