#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, csv, json, math, random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

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

class NativePatchDataset(Dataset):
    def __init__(self, paths):
        self.paths=list(paths)
        self.records=[]
        for i,p in enumerate(self.paths):
            with Image.open(p) as im:
                w,h=im.size
            b=boxes(w,h)
            for name in PATCHES:
                self.records.append((i,name,b[name]))
    def __len__(self): return len(self.records)
    def __getitem__(self,idx):
        i,name,(x0,y0,x1,y1)=self.records[idx]
        arr=load_gray(self.paths[i])
        patch=arr[y0:y1,x0:x1]
        return {
            "x":torch.from_numpy(patch).unsqueeze(0),
            "image_name":self.paths[i].name,
            "patch_name":name,
            "native_h":patch.shape[0],
            "native_w":patch.shape[1]
        }

def collate_native(batch):
    mh=roundup(max(x["native_h"] for x in batch),8)
    mw=roundup(max(x["native_w"] for x in batch),8)
    xs,masks=[],[]
    for item in batch:
        x=item["x"]; _,h,w=x.shape
        xp=F.pad(x.unsqueeze(0),(0,mw-w,0,mh-h),mode="replicate").squeeze(0)
        valid=torch.zeros((1,mh,mw),dtype=torch.float32)
        valid[:,:h,:w]=1.0
        xs.append(xp); masks.append(valid)
    return torch.stack(xs),torch.stack(masks)

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
        self.pool=nn.AdaptiveAvgPool2d((2,2))
    def encode(self,x):
        x=F.relu(self.stem(x),inplace=True); x=self.enc1(x)
        x=F.relu(self.down1(x),inplace=True); x=self.enc2(x)
        x=F.relu(self.down2(x),inplace=True); x=self.enc3(x)
        x=F.relu(self.down3(x),inplace=True); x=self.bot1(x); x=self.bot2(x)
        return x
    def forward(self,x,return_embedding=False):
        z=self.encode(x)
        emb=self.pool(z).flatten(1)
        x=F.relu(self.up3(z),inplace=True); x=self.dec3(x)
        x=F.relu(self.up2(x),inplace=True); x=self.dec2(x)
        x=F.relu(self.up1(x),inplace=True); x=self.dec1(x)
        out=self.out(x)
        return (out,emb) if return_embedding else out

def masked_mse(pred,target,valid):
    return (((pred-target)**2)*valid).sum()/valid.sum().clamp_min(1.0)

def psnr(mse):
    return float("inf") if mse<=0 else float(10*math.log10(1.0/mse))

def metrics(a,b):
    mse=float(np.mean((a-b)**2))
    mae=float(np.mean(np.abs(a-b)))
    ssim=float(ssim_fn(a,b,data_range=1.0)) if HAS_SKIMAGE else None
    return {"mse":mse,"mae":mae,"psnr_db":psnr(mse),"ssim":ssim}

def write_csv(path,rows):
    if not rows:return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader();w.writerows(rows)

def summarize(rows):
    d={"n":len(rows)}
    for k in ("mse","mae","psnr_db","ssim"):
        vals=[r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        d[f"mean_{k}"]=float(np.mean(vals)) if vals else None
        d[f"std_{k}"]=float(np.std(vals)) if vals else None
    return d

def train_epoch(model,loader,opt,scaler,device,use_amp):
    model.train(); sqtot=0.; ntot=0.
    for x,valid in loader:
        x=x.to(device,non_blocking=True); valid=valid.to(device,non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=use_amp):
            pred=model(x); loss=masked_mse(pred,x,valid)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        with torch.no_grad():
            sqtot+=((((pred-x)**2)*valid).sum().item())
            ntot+=valid.sum().item()
    return sqtot/max(ntot,1.0)

@torch.no_grad()
def evaluate(model,paths,device,out,prefix,save_examples=10):
    model.eval()
    patch_rows=[]; image_rows=[]; p_emb=[]; p_meta=[]; i_emb=[]; i_names=[]; examples=[]
    for ii,p in enumerate(paths):
        orig=load_gray(p); h,w=orig.shape; bb=boxes(w,h)
        recon=np.zeros_like(orig); embs=[]
        for name in PATCHES:
            x0,y0,x1,y1=bb[name]
            patch=orig[y0:y1,x0:x1]; ph,pw=patch.shape
            t=torch.from_numpy(patch).float()[None,None]
            t=F.pad(t,(0,roundup(pw)-pw,0,roundup(ph)-ph),mode="replicate").to(device)
            pred,emb=model(t,return_embedding=True)
            pred=torch.clamp(pred,0,1)[0,0,:ph,:pw].cpu().numpy()
            recon[y0:y1,x0:x1]=pred
            m=metrics(patch,pred)
            patch_rows.append({"image":p.name,"patch":name,"patch_width":pw,"patch_height":ph,**m})
            e=emb[0].cpu().numpy().astype(np.float32); p_emb.append(e); p_meta.append((p.name,name,pw,ph)); embs.append(e)
        image_rows.append({"image":p.name,"width":w,"height":h,**metrics(orig,recon)})
        i_emb.append(np.mean(np.stack(embs),axis=0)); i_names.append(p.name)
        if len(examples)<save_examples: examples.append((p.name,orig,recon))
        if (ii+1)%25==0: print(f"  {prefix}: {ii+1}/{len(paths)}")

    np.savez_compressed(out/f"{prefix}_patch_embeddings_1024d.npz",
                        embeddings=np.stack(p_emb),
                        image=np.array([x[0] for x in p_meta]),
                        patch=np.array([x[1] for x in p_meta]),
                        patch_width=np.array([x[2] for x in p_meta]),
                        patch_height=np.array([x[3] for x in p_meta]))
    np.savez_compressed(out/f"{prefix}_image_embeddings_1024d.npz",
                        embeddings=np.stack(i_emb),image=np.array(i_names))
    write_csv(out/f"{prefix}_patch_metrics.csv",patch_rows)
    write_csv(out/f"{prefix}_image_metrics.csv",image_rows)

    if examples:
        thumb=260; rh=300
        canvas=Image.new("RGB",(thumb*3+80,rh*len(examples)+40),"white")
        draw=ImageDraw.Draw(canvas)
        draw.text((10,10),"Original",fill="black")
        draw.text((thumb+30,10),"Reconstructed",fill="black")
        draw.text((2*thumb+50,10),"Abs Error (norm)",fill="black")
        y=40
        for name,a,b in examples:
            err=np.abs(a-b); err=err/(err.max()+1e-8)
            draw.text((10,y),name,fill="black"); yy=y+25
            for j,arr in enumerate((a,b,err)):
                im=Image.fromarray((np.clip(arr,0,1)*255).astype(np.uint8),"L")
                scale=min(thumb/im.width,(rh-35)/im.height)
                size=(max(1,int(im.width*scale)),max(1,int(im.height*scale)))
                im=im.resize(size,Image.Resampling.NEAREST).convert("RGB")
                canvas.paste(im,(10+j*(thumb+20),yy))
            y+=rh
        canvas.save(out/f"{prefix}_reconstruction_examples.png")
    return patch_rows,image_rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_root",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--epochs",type=int,default=200)
    ap.add_argument("--lr",type=float,default=5e-7)
    ap.add_argument("--min_lr",type=float,default=1e-7)
    ap.add_argument("--batch_size",type=int,default=4)
    ap.add_argument("--workers",type=int,default=2)
    ap.add_argument("--seed",type=int,default=42)
    ap.add_argument("--save_examples",type=int,default=10)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    root=Path(args.dataset_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    train=list_images(root/"classification_task"/"train"/"no_tumor")
    test=list_images(root/"classification_task"/"test"/"no_tumor")

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt=torch.load(args.checkpoint,map_location=device)
    b=int(ckpt.get("base_channels",32))
    model=NativeResNetAutoencoder(b).to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)

    print("="*90)
    print("HEALTHY NATIVE RESNET - LOW LR CONTINUATION")
    print("="*90)
    print(f"Starting checkpoint epoch : {ckpt.get('epoch','?')}")
    print(f"Healthy train images      : {len(train)}")
    print(f"Healthy test images       : {len(test)}")
    print("Spatial resize            : NONE")
    print("Native 2x2 patches        : YES")
    print("U-Net long skips          : NONE")
    print("Local ResNet residuals    : YES")
    print(f"Continuation epochs       : {args.epochs}")
    print(f"LR -> min LR              : {args.lr:.2e} -> {args.min_lr:.2e}")
    print(f"Device                    : {device}")

    ds=NativePatchDataset(train)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True,num_workers=args.workers,
                      pin_memory=(device.type=="cuda"),persistent_workers=(args.workers>0),
                      collate_fn=collate_native)
    opt=torch.optim.Adam(model.parameters(),lr=args.lr)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs,eta_min=args.min_lr)
    use_amp=device.type=="cuda"; scaler=torch.amp.GradScaler("cuda",enabled=use_amp)

    history=[]; best=float("inf")
    for ep in range(1,args.epochs+1):
        mse=train_epoch(model,loader,opt,scaler,device,use_amp)
        row={"continue_epoch":ep,"lr":opt.param_groups[0]["lr"],"train_mse":mse,"train_psnr_db":psnr(mse)}
        history.append(row); write_csv(out/"history_continue.csv",history)
        state={"model_state_dict":model.state_dict(),"epoch":ep,"train_mse":mse,"base_channels":b,
               "architecture":"NativeResNetAutoencoder","healthy_only":True,"spatial_resize":False,
               "native_2x2_patches":True,"long_encoder_decoder_skip_connections":False,
               "residual_block_connections":True,"continued_from":str(args.checkpoint)}
        torch.save(state,out/"last_autoencoder_continue.pt")
        if mse<best:
            best=mse; torch.save(state,out/"best_autoencoder_continue.pt")
        print(f"Continue {ep:03d}/{args.epochs} | lr={opt.param_groups[0]['lr']:.3e} | "
              f"train_mse={mse:.8e} | train_psnr={psnr(mse):.2f} dB"
              + ("  <-- BEST" if mse==best else ""))
        sched.step()

    print("\nEvaluating healthy TRAIN...")
    trp,tri=evaluate(model,train,device,out,"train_healthy",0)
    print("\nEvaluating independent healthy TEST...")
    tep,tei=evaluate(model,test,device,out,"test_healthy",args.save_examples)

    final={
        "experiment":"healthy native ResNet low-LR continuation",
        "continued_from":str(args.checkpoint),
        "healthy_only_training":True,"spatial_resize":False,"native_2x2_patches":True,
        "long_unet_skips":False,"local_residual_connections":True,
        "continuation_epochs":args.epochs,"initial_lr":args.lr,"min_lr":args.min_lr,
        "train_whole_image_metrics":summarize(tri),
        "test_whole_image_metrics":summarize(tei),
        "test_patch_metrics":summarize(tep),
        "reference_previous_test_psnr_db":61.67342404937434
    }
    (out/"final_summary.json").write_text(json.dumps(final,indent=2),encoding="utf-8")
    cur=final["test_whole_image_metrics"]["mean_psnr_db"]
    print("\n=== FINAL CONTINUED RESNET HEALTHY TEST ===")
    print(json.dumps(final["test_whole_image_metrics"],indent=2))
    print(f"\nPrevious ResNet PSNR : 61.6734 dB")
    print(f"Continued ResNet PSNR: {cur:.4f} dB")
    print(f"Delta                : {cur-61.67342404937434:+.4f} dB")
    print("Saved outputs:",out)

if __name__=="__main__":
    main()
