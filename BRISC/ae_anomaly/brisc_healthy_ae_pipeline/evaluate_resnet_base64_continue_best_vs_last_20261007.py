#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, csv, json, math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
PATCHES=("TL","TR","BL","BR")

def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def boxes(w,h):
    xm,ym=w//2,h//2
    return {"TL":(0,0,xm,ym),"TR":(xm,0,w,ym),"BL":(0,ym,xm,h),"BR":(xm,ym,w,h)}

def roundup(v,m=8):
    return ((v+m-1)//m)*m

def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"),dtype=np.float32)/255.0

class ResidualBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=nn.Conv2d(c,c,3,padding=1,bias=True)
        self.conv2=nn.Conv2d(c,c,3,padding=1,bias=True)
    def forward(self,x):
        y=F.relu(self.conv1(x),inplace=True)
        y=self.conv2(y)
        return F.relu(y+x,inplace=True)

class NativeResNetAutoencoder(nn.Module):
    def __init__(self,b=32):
        super().__init__()
        self.stem=nn.Conv2d(1,b,3,padding=1,bias=True)
        self.enc1=ResidualBlock(b)
        self.down1=nn.Conv2d(b,b*2,4,2,1,bias=True)
        self.enc2=ResidualBlock(b*2)
        self.down2=nn.Conv2d(b*2,b*4,4,2,1,bias=True)
        self.enc3=ResidualBlock(b*4)
        self.down3=nn.Conv2d(b*4,b*8,4,2,1,bias=True)
        self.bot1=ResidualBlock(b*8)
        self.bot2=ResidualBlock(b*8)
        self.up3=nn.ConvTranspose2d(b*8,b*4,4,2,1,bias=True)
        self.dec3=ResidualBlock(b*4)
        self.up2=nn.ConvTranspose2d(b*4,b*2,4,2,1,bias=True)
        self.dec2=ResidualBlock(b*2)
        self.up1=nn.ConvTranspose2d(b*2,b,4,2,1,bias=True)
        self.dec1=ResidualBlock(b)
        self.out=nn.Conv2d(b,1,1,bias=True)

    def forward(self,x):
        x=F.relu(self.stem(x),inplace=True); x=self.enc1(x)
        x=F.relu(self.down1(x),inplace=True); x=self.enc2(x)
        x=F.relu(self.down2(x),inplace=True); x=self.enc3(x)
        x=F.relu(self.down3(x),inplace=True); x=self.bot1(x); x=self.bot2(x)
        x=F.relu(self.up3(x),inplace=True); x=self.dec3(x)
        x=F.relu(self.up2(x),inplace=True); x=self.dec2(x)
        x=F.relu(self.up1(x),inplace=True); x=self.dec1(x)
        return self.out(x)

def psnr(mse):
    return float("inf") if mse<=0 else float(10.0*math.log10(1.0/mse))

def metrics(a,b):
    mse=float(np.mean((a-b)**2))
    mae=float(np.mean(np.abs(a-b)))
    ssim=float(ssim_fn(a,b,data_range=1.0)) if HAS_SKIMAGE else None
    return {"mse":mse,"mae":mae,"psnr_db":psnr(mse),"ssim":ssim}

def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def summarize(rows):
    out={"n":len(rows)}
    for k in ("mse","mae","psnr_db","ssim"):
        vals=[r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        out[f"mean_{k}"]=float(np.mean(vals)) if vals else None
        out[f"std_{k}"]=float(np.std(vals)) if vals else None
    return out

def load_model(ckpt_path,device):
    ckpt=torch.load(ckpt_path,map_location=device)
    b=int(ckpt.get("base_channels",64))
    model=NativeResNetAutoencoder(b).to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval()
    return model,ckpt,b

@torch.no_grad()
def evaluate_model(model,paths,device):
    image_rows=[]
    patch_rows=[]

    for ii,p in enumerate(paths):
        orig=load_gray(p)
        h,w=orig.shape
        bb=boxes(w,h)
        recon=np.zeros_like(orig,dtype=np.float32)

        for name in PATCHES:
            x0,y0,x1,y1=bb[name]
            patch=orig[y0:y1,x0:x1]
            ph,pw=patch.shape

            t=torch.from_numpy(patch).float()[None,None]
            t=F.pad(
                t,
                (0,roundup(pw)-pw,0,roundup(ph)-ph),
                mode="replicate"
            ).to(device)

            pred=model(t)
            pred=torch.clamp(pred,0.0,1.0)[0,0,:ph,:pw].cpu().numpy()

            recon[y0:y1,x0:x1]=pred

            patch_rows.append({
                "image":p.name,
                "patch":name,
                "patch_width":pw,
                "patch_height":ph,
                **metrics(patch,pred)
            })

        image_rows.append({
            "image":p.name,
            "width":w,
            "height":h,
            **metrics(orig,recon)
        })

        if (ii+1)%25==0:
            print(f"  evaluated {ii+1}/{len(paths)}")

    return patch_rows,image_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_root",required=True)
    ap.add_argument("--best_checkpoint",required=True)
    ap.add_argument("--last_checkpoint",required=True)
    ap.add_argument("--output_dir",required=True)
    args=ap.parse_args()

    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)

    test_dir=Path(args.dataset_root)/"classification_task"/"test"/"no_tumor"
    test_images=list_images(test_dir)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("="*90)
    print("BEST vs LAST CHECKPOINT - RESNET BASE64 CONTINUATION")
    print("="*90)
    print(f"Healthy test images : {len(test_images)}")
    print(f"Device              : {device}")
    print("Spatial resize      : NONE")
    print("Patch strategy      : native 2x2")
    print()

    results={}

    for label,ckpt_path in [
        ("best",args.best_checkpoint),
        ("last",args.last_checkpoint)
    ]:
        print(f"Evaluating {label.upper()} checkpoint:")
        print(f"  {ckpt_path}")

        model,ckpt,b=load_model(ckpt_path,device)
        print(f"  checkpoint epoch  : {ckpt.get('epoch','?')}")
        print(f"  base channels     : {b}")
        print(f"  train MSE saved   : {ckpt.get('train_mse','?')}")

        patch_rows,image_rows=evaluate_model(model,test_images,device)

        write_csv(out/f"{label}_test_patch_metrics.csv",patch_rows)
        write_csv(out/f"{label}_test_image_metrics.csv",image_rows)

        results[label]={
            "checkpoint":str(ckpt_path),
            "checkpoint_epoch":ckpt.get("epoch"),
            "checkpoint_train_mse":ckpt.get("train_mse"),
            "base_channels":b,
            "patch_metrics":summarize(patch_rows),
            "whole_image_metrics":summarize(image_rows),
        }

        print(
            f"  WHOLE PSNR = "
            f"{results[label]['whole_image_metrics']['mean_psnr_db']:.6f} dB"
        )
        print(
            f"  WHOLE MSE  = "
            f"{results[label]['whole_image_metrics']['mean_mse']:.8e}"
        )
        print(
            f"  WHOLE SSIM = "
            f"{results[label]['whole_image_metrics']['mean_ssim']:.9f}"
            if results[label]['whole_image_metrics']['mean_ssim'] is not None
            else "  WHOLE SSIM = unavailable"
        )
        print()

    best_psnr=results["best"]["whole_image_metrics"]["mean_psnr_db"]
    last_psnr=results["last"]["whole_image_metrics"]["mean_psnr_db"]
    delta=best_psnr-last_psnr

    summary={
        "experiment":"best-vs-last ResNet base64 continuation checkpoint evaluation",
        "healthy_test_n":len(test_images),
        "results":results,
        "best_minus_last_psnr_db":delta,
        "winner":"best" if delta>0 else ("last" if delta<0 else "tie"),
        "note":"Both checkpoints evaluated on the exact same independent 140 healthy images with native 2x2 patches and no spatial resize."
    }

    (out/"best_vs_last_summary.json").write_text(
        json.dumps(summary,indent=2),
        encoding="utf-8"
    )

    print("="*90)
    print("FINAL COMPARISON")
    print("="*90)
    print(f"BEST checkpoint PSNR : {best_psnr:.6f} dB")
    print(f"LAST checkpoint PSNR : {last_psnr:.6f} dB")
    print(f"BEST - LAST          : {delta:+.6f} dB")
    print(f"Winner               : {summary['winner'].upper()}")
    print()
    print("Saved outputs:",out)

if __name__=="__main__":
    main()
