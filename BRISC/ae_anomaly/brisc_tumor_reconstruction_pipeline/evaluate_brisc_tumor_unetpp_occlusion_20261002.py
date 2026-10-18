#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw

try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SKIMAGE = True
except Exception:
    HAS_SKIMAGE = False

IMAGE_EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}

class ConvBlock(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=True),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=True),nn.ReLU(inplace=True))
    def forward(self,x): return self.block(x)

class UNetPlusPlusAutoencoder(nn.Module):
    def __init__(self,base_channels=32):
        super().__init__(); b=base_channels; self.pool=nn.MaxPool2d(2,2)
        self.conv0_0=ConvBlock(1,b); self.conv1_0=ConvBlock(b,b*2); self.conv2_0=ConvBlock(b*2,b*4); self.conv3_0=ConvBlock(b*4,b*8)
        self.conv0_1=ConvBlock(b+b*2,b); self.conv1_1=ConvBlock(b*2+b*4,b*2); self.conv2_1=ConvBlock(b*4+b*8,b*4)
        self.conv0_2=ConvBlock(b*2+b*2,b); self.conv1_2=ConvBlock(b*4+b*4,b*2); self.conv0_3=ConvBlock(b*3+b*2,b)
        self.out=nn.Conv2d(b,1,1)
    @staticmethod
    def up_to(x,ref): return F.interpolate(x,size=ref.shape[-2:],mode='bilinear',align_corners=False)
    def forward(self,x):
        x0_0=self.conv0_0(x); x1_0=self.conv1_0(self.pool(x0_0)); x2_0=self.conv2_0(self.pool(x1_0)); x3_0=self.conv3_0(self.pool(x2_0))
        x0_1=self.conv0_1(torch.cat([x0_0,self.up_to(x1_0,x0_0)],1)); x1_1=self.conv1_1(torch.cat([x1_0,self.up_to(x2_0,x1_0)],1)); x2_1=self.conv2_1(torch.cat([x2_0,self.up_to(x3_0,x2_0)],1))
        x0_2=self.conv0_2(torch.cat([x0_0,x0_1,self.up_to(x1_1,x0_0)],1)); x1_2=self.conv1_2(torch.cat([x1_0,x1_1,self.up_to(x2_1,x1_0)],1))
        x0_3=self.conv0_3(torch.cat([x0_0,x0_1,x0_2,self.up_to(x1_2,x0_0)],1))
        return self.out(x0_3)

def load_ckpt(path,device):
    ckpt=torch.load(path,map_location=device); size=int(ckpt.get('image_size',256)); base=int(ckpt.get('base_channels',32))
    model=UNetPlusPlusAutoencoder(base).to(device); model.load_state_dict(ckpt['model_state_dict'],strict=True); model.eval(); return model,size,base

def list_images(folder): return sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS)

def load_gray(path,size,nearest=False):
    with Image.open(path) as im:
        im=im.convert('L'); rs=Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR; im=im.resize((size,size),resample=rs)
        arr=np.asarray(im,dtype=np.float32)/255.0
    return torch.from_numpy(arr).unsqueeze(0)

def phash(path,size):
    with Image.open(path) as im:
        im=im.convert('L').resize((size,size),resample=Image.Resampling.BILINEAR); arr=np.asarray(im,dtype=np.uint8)
    h=hashlib.sha256(); h.update(str(arr.shape).encode()); h.update(arr.tobytes()); return h.hexdigest()

def psnr(mse): return float('inf') if mse<=0 else float(10*math.log10(1.0/mse))

def metrics(target,recon,mask):
    non=~mask; err=(target-recon)**2; ae=np.abs(target-recon)
    fm=float(err.mean()); mm=float(err[mask].mean()); nm=float(err[non].mean())
    out={
        'full_mse':fm,'full_mae':float(ae.mean()),'full_psnr_db':psnr(fm),
        'mask_mse':mm,'mask_mae':float(ae[mask].mean()),'mask_psnr_db':psnr(mm),
        'nonmask_mse':nm,'nonmask_mae':float(ae[non].mean()),'nonmask_psnr_db':psnr(nm),
    }
    out['ssim']=float(ssim_fn(target,recon,data_range=1.0)) if HAS_SKIMAGE else None
    return out

def summarize(rows):
    out={'n_images':len(rows)}
    for k in ['full_mse','full_mae','full_psnr_db','mask_mse','mask_mae','mask_psnr_db','nonmask_mse','nonmask_mae','nonmask_psnr_db','ssim']:
        vals=[r[k] for r in rows if r[k] is not None and np.isfinite(r[k])]
        out['mean_'+k]=float(np.mean(vals)) if vals else None; out['std_'+k]=float(np.std(vals)) if vals else None
    return out

def write_csv(path,rows):
    if not rows: return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def to_pil(a): return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8),'L')

def save_sheet(examples,out_path,cell=180):
    if not examples:return
    labels=['Original','Masked Input','Recon','Abs Diff','GT Mask']; m=12; lh=26; nh=22
    W=m+len(labels)*(cell+m); H=m+len(examples)*(cell+lh+nh+m); canvas=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(canvas)
    for c,l in enumerate(labels): d.text((m+c*(cell+m),4),l,fill='black')
    y=lh+m
    for ex in examples:
        d.text((m,y),ex['name'],fill='black'); yy=y+nh
        imgs=[to_pil(ex['orig']),to_pil(ex['masked']),to_pil(ex['recon']),to_pil(ex['diff']),Image.fromarray((ex['mask'].astype(np.uint8)*255),'L')]
        for c,im in enumerate(imgs): canvas.paste(im.convert('RGB').resize((cell,cell)),(m+c*(cell+m),yy))
        y=yy+cell+m
    canvas.save(out_path)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset_root',required=True); ap.add_argument('--checkpoint',required=True); ap.add_argument('--output_dir',required=True); ap.add_argument('--save_examples',type=int,default=12); args=ap.parse_args()
    root=Path(args.dataset_root); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model,size,base=load_ckpt(Path(args.checkpoint),device)
    train=list_images(root/'segmentation_task'/'train'/'images'); test=list_images(root/'segmentation_task'/'test'/'images'); masks=list_images(root/'segmentation_task'/'test'/'masks'); mmap={p.stem:p for p in masks}; pairs=[(p,mmap[p.stem]) for p in test if p.stem in mmap]
    print(f'Device: {device}\nCheckpoint: {args.checkpoint}\nImage size: {size}\nBase channels: {base}\nTrain images: {len(train)} | Test pairs: {len(pairs)}')
    print('\nBuilding post-preprocessing train hashes...'); hs=set()
    for i,p in enumerate(train,1):
        hs.add(phash(p,size));
        if i%500==0: print(f'  train hashed {i}/{len(train)}')
    clean=[]; overlap=[]
    for p,m in pairs: (overlap if phash(p,size) in hs else clean).append((p,m))
    print(f'\nPost-preprocessing overlap test images: {len(overlap)}\nClean test images: {len(clean)}')
    intact_rows=[]; occ_rows=[]; paired=[]; examples=[]
    with torch.no_grad():
        for idx,(ip,mp) in enumerate(clean):
            x=load_gray(ip,size,False).to(device); mt=load_gray(mp,size,True); mask=(mt.squeeze().numpy()>0.5); xb=x.unsqueeze(0); md=(mt.unsqueeze(0).to(device)>0.5).float()
            ri=torch.clamp(model(xb),0,1); xo=xb*(1-md); ro=torch.clamp(model(xo),0,1)
            target=xb.cpu().squeeze().numpy(); intact=ri.cpu().squeeze().numpy(); occin=xo.cpu().squeeze().numpy(); occ=ro.cpu().squeeze().numpy()
            im=metrics(target,intact,mask); om=metrics(target,occ,mask); intact_rows.append({'image':ip.name,**im}); occ_rows.append({'image':ip.name,**om})
            paired.append({'image':ip.name,'mask_area_ratio':float(mask.mean()),'intact_mask_psnr_db':im['mask_psnr_db'],'occluded_mask_psnr_db':om['mask_psnr_db'],'delta_mask_psnr_db':om['mask_psnr_db']-im['mask_psnr_db'],'intact_mask_mse':im['mask_mse'],'occluded_mask_mse':om['mask_mse'],'mse_ratio_occ_over_intact':om['mask_mse']/im['mask_mse'] if im['mask_mse']>0 else None})
            if len(examples)<args.save_examples:
                diff=np.abs(target-occ); examples.append({'name':ip.name,'orig':target,'masked':occin,'recon':occ,'diff':diff/(diff.max()+1e-8),'mask':mask})
            if (idx+1)%100==0: print(f'Processed {idx+1}/{len(clean)}')
    write_csv(out/'intact_metrics_per_case.csv',intact_rows); write_csv(out/'occluded_metrics_per_case.csv',occ_rows); write_csv(out/'paired_occlusion_comparison.csv',paired)
    si=summarize(intact_rows); so=summarize(occ_rows); delta={'mask_psnr_db':so['mean_mask_psnr_db']-si['mean_mask_psnr_db'],'mask_mse':so['mean_mask_mse']-si['mean_mask_mse'],'full_psnr_db':so['mean_full_psnr_db']-si['mean_full_psnr_db'],'nonmask_psnr_db':so['mean_nonmask_psnr_db']-si['mean_nonmask_psnr_db'],'ssim':(so['mean_ssim']-si['mean_ssim']) if so['mean_ssim'] is not None else None}
    summary={'experiment':'GT-mask tumor occlusion sanity test','checkpoint':str(args.checkpoint),'model':'UNetPlusPlusAutoencoder','image_size':size,'base_channels':base,'provided_test_n':len(pairs),'excluded_post_preprocessing_overlap_n':len(overlap),'clean_test_n':len(clean),'intervention':'Inference only: set GT tumor-mask pixels in input to zero. No retraining.','target':'Original unoccluded tumor MRI.','intact':si,'occluded':so,'delta_occluded_minus_intact':delta,'note':'This is an intervention test, not a trained inpainting model. Zero-filled tumor regions are out-of-distribution relative to the original self-reconstruction training.'}
    (out/'occlusion_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); save_sheet(examples,out/'occlusion_examples_contact_sheet.png')
    print('\n=== OCCLUSION TEST SUMMARY ==='); print(json.dumps(summary,indent=2)); print('\nKey comparison:'); print(f"Intact tumor-region PSNR   : {si['mean_mask_psnr_db']:.4f} dB"); print(f"Occluded tumor-region PSNR : {so['mean_mask_psnr_db']:.4f} dB"); print(f"Delta                      : {delta['mask_psnr_db']:.4f} dB"); print(f"Intact tumor-region MSE    : {si['mean_mask_mse']:.8e}"); print(f"Occluded tumor-region MSE  : {so['mean_mask_mse']:.8e}"); print(f'\nSaved outputs: {out}')

if __name__=='__main__': main()
