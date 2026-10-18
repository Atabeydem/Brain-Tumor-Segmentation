from pathlib import Path
import json, math, random, csv
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

def list_images(folder: Path) -> List[Path]:
    folder = Path(folder)
    return sorted([p for p in folder.rglob('*') if p.suffix.lower() in IMG_EXTS])

def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def load_grayscale_tensor(path: Path, image_size: int, nearest: bool = False) -> torch.Tensor:
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    img = img.resize((image_size, image_size), resample=resample)
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)

class ImageReconstructionDataset(Dataset):
    def __init__(self, files: List[Path], image_size: int):
        self.files = list(files); self.image_size = image_size
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        x = load_grayscale_tensor(self.files[idx], self.image_size, nearest=False)
        return x, x, str(self.files[idx])

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)

class CNNAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__(); b = base_channels
        self.encoder = nn.Sequential(ConvBlock(in_channels,b), nn.MaxPool2d(2), ConvBlock(b,b*2), nn.MaxPool2d(2), ConvBlock(b*2,b*4), nn.MaxPool2d(2), ConvBlock(b*4,b*8))
        self.decoder = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(b*8,b*4), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(b*4,b*2), nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False), ConvBlock(b*2,b), nn.Conv2d(b,in_channels,1), nn.Sigmoid())
    def forward(self, x): return self.decoder(self.encoder(x))

class UNetAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super().__init__(); b = base_channels
        self.e1=ConvBlock(in_channels,b); self.e2=ConvBlock(b,b*2); self.e3=ConvBlock(b*2,b*4); self.bottleneck=ConvBlock(b*4,b*8)
        self.d3=ConvBlock(b*8+b*4,b*4); self.d2=ConvBlock(b*4+b*2,b*2); self.d1=ConvBlock(b*2+b,b)
        self.out=nn.Sequential(nn.Conv2d(b,in_channels,1), nn.Sigmoid())
    def forward(self, x):
        e1=self.e1(x); e2=self.e2(F.max_pool2d(e1,2)); e3=self.e3(F.max_pool2d(e2,2)); z=self.bottleneck(F.max_pool2d(e3,2))
        d3=F.interpolate(z, size=e3.shape[-2:], mode='bilinear', align_corners=False); d3=self.d3(torch.cat([d3,e3], dim=1))
        d2=F.interpolate(d3, size=e2.shape[-2:], mode='bilinear', align_corners=False); d2=self.d2(torch.cat([d2,e2], dim=1))
        d1=F.interpolate(d2, size=e1.shape[-2:], mode='bilinear', align_corners=False); d1=self.d1(torch.cat([d1,e1], dim=1))
        return self.out(d1)

def build_model(model_name: str, base_channels: int = 32) -> nn.Module:
    model_name = model_name.lower()
    if model_name == 'cnn': return CNNAutoencoder(1, base_channels)
    if model_name == 'unet': return UNetAutoencoder(1, base_channels)
    raise ValueError(f'Unknown model: {model_name}. Use cnn or unet.')

def mse_np(a: np.ndarray, b: np.ndarray) -> float: return float(np.mean((a.astype(np.float64)-b.astype(np.float64))**2))
def mae_np(a: np.ndarray, b: np.ndarray) -> float: return float(np.mean(np.abs(a.astype(np.float64)-b.astype(np.float64))))
def psnr_from_mse(mse: float, max_pixel: float = 1.0) -> float:
    if mse <= 0: return float('inf')
    return float(20.0 * math.log10(max_pixel) - 10.0 * math.log10(mse))
def try_ssim(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    try:
        from skimage.metrics import structural_similarity as ssim
        return float(ssim(a, b, data_range=1.0))
    except Exception:
        return None

def save_tensor_image(t: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = t.detach().cpu().squeeze().clamp(0,1).numpy()
    Image.fromarray((arr*255).astype(np.uint8)).save(path)

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    arr = t.detach().cpu().squeeze().clamp(0,1).numpy()
    return Image.fromarray((arr*255).astype(np.uint8)).convert('L')

def save_contact_sheet(rows: List[Tuple[str, List[Image.Image]]], labels: List[str], out_path: Path, cell_size: int = 160):
    if not rows: return
    margin=20; label_h=30; title_w=220; cols=len(labels)
    w=title_w+cols*cell_size+margin*2; h=margin+label_h+len(rows)*(cell_size+label_h)+margin
    canvas=Image.new('RGB',(w,h),'white'); draw=ImageDraw.Draw(canvas)
    try:
        font=ImageFont.truetype('arial.ttf',18); small=ImageFont.truetype('arial.ttf',15)
    except Exception:
        font=ImageFont.load_default(); small=ImageFont.load_default()
    y=margin
    for j, lab in enumerate(labels): draw.text((title_w+j*cell_size+8,y), lab, fill='black', font=font)
    y += label_h
    for row_name, imgs in rows:
        draw.text((margin,y+8), row_name[:28], fill='black', font=small)
        for j,img in enumerate(imgs): canvas.paste(img.convert('RGB').resize((cell_size,cell_size), Image.Resampling.BILINEAR),(title_w+j*cell_size,y))
        y += cell_size+label_h
    out_path.parent.mkdir(parents=True, exist_ok=True); canvas.save(out_path)

def binary_metrics(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-7):
    pred=pred.astype(bool); gt=gt.astype(bool)
    tp=float(np.logical_and(pred,gt).sum()); fp=float(np.logical_and(pred,~gt).sum()); fn=float(np.logical_and(~pred,gt).sum())
    return {'dice': float((2*tp+eps)/(2*tp+fp+fn+eps)), 'iou': float((tp+eps)/(tp+fp+fn+eps)), 'precision': float((tp+eps)/(tp+fp+eps)), 'recall': float((tp+eps)/(tp+fn+eps)), 'tp':tp, 'fp':fp, 'fn':fn}

def write_csv(path: Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows: return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=list(rows[0].keys())); writer.writeheader(); writer.writerows(rows)

def load_checkpoint_model(checkpoint_path: Path, device: torch.device):
    ckpt=torch.load(checkpoint_path, map_location=device)
    config=ckpt.get('config', {})
    model=build_model(config.get('model','unet'), int(config.get('base_channels',32))).to(device)
    model.load_state_dict(ckpt['model_state_dict']); model.eval()
    return model, config, int(config.get('image_size',256))
