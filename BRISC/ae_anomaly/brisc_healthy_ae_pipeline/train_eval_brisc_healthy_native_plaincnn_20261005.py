#!/usr/bin/env python
import argparse,csv,json,math,random
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
try:
    from skimage.metrics import structural_similarity as ssim_fn
    HAS_SSIM=True
except Exception:
    HAS_SSIM=False
EXTS={'.png','.jpg','.jpeg','.bmp','.tif','.tiff'}
PATCHES=['TL','TR','BL','BR']

def list_images(d):
    return sorted(p for p in d.rglob('*') if p.is_file() and p.suffix.lower() in EXTS)

def boxes(w,h):
    xm,ym=w//2,h//2
    return {'TL':(0,0,xm,ym),'TR':(xm,0,w,ym),'BL':(0,ym,xm,h),'BR':(xm,ym,w,h)}

def load_gray(p):
    with Image.open(p) as im:
        return np.asarray(im.convert('L'),dtype=np.float32)/255.0

def rup(v,m=8): return ((v+m-1)//m)*m

def psnr(mse): return float('inf') if mse<=0 else 10*math.log10(1.0/mse)

def metrics(a,b):
    mse=float(np.mean((a-b)**2)); mae=float(np.mean(np.abs(a-b)))
    s=float(ssim_fn(a,b,data_range=1.0)) if HAS_SSIM else None
    return {'mse':mse,'mae':mae,'psnr_db':psnr(mse),'ssim':s}

def write_csv(path,rows):
    if not rows:return
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def summary(rows):
    out={'n':len(rows)}
    for k in ['mse','mae','psnr_db','ssim']:
        v=[r[k] for r in rows if r.get(k) is not None and np.isfinite(r[k])]
        out['mean_'+k]=float(np.mean(v)) if v else None
        out['std_'+k]=float(np.std(v)) if v else None
    return out

class PatchDS(Dataset):
    def __init__(self,paths):
        self.paths=list(paths); self.rec=[]
        for i,p in enumerate(self.paths):
            with Image.open(p) as im:w,h=im.size
            bb=boxes(w,h)
            for name in PATCHES:self.rec.append((i,name,bb[name]))
    def __len__(self):return len(self.rec)
    def __getitem__(self,idx):
        i,name,(x0,y0,x1,y1)=self.rec[idx]; p=self.paths[i]; a=load_gray(p)
        patch=a[y0:y1,x0:x1]; x=torch.from_numpy(patch).unsqueeze(0)
        return x, {'image_idx':i,'image_name':p.name,'patch':name,'h':patch.shape[0],'w':patch.shape[1]}

def collate(batch):
    mh=rup(max(m['h'] for _,m in batch)); mw=rup(max(m['w'] for _,m in batch))
    xs=[]; ms=[]; meta=[]
    for x,m in batch:
        _,h,w=x.shape
        xp=F.pad(x.unsqueeze(0),(0,mw-w,0,mh-h),mode='replicate').squeeze(0)
        valid=torch.zeros((1,mh,mw),dtype=torch.float32); valid[:,:h,:w]=1
        xs.append(xp);ms.append(valid);meta.append(m)
    return torch.stack(xs),torch.stack(ms),meta

class Block(nn.Module):
    def __init__(self,a,b):
        super().__init__();self.n=nn.Sequential(nn.Conv2d(a,b,3,padding=1),nn.ReLU(inplace=True),nn.Conv2d(b,b,3,padding=1),nn.ReLU(inplace=True))
    def forward(self,x):return self.n(x)

class PlainAE(nn.Module):
    def __init__(self,b=32):
        super().__init__();
        self.e1=Block(1,b);self.d1=nn.Conv2d(b,b*2,4,2,1)
        self.e2=Block(b*2,b*2);self.d2=nn.Conv2d(b*2,b*4,4,2,1)
        self.e3=Block(b*4,b*4);self.d3=nn.Conv2d(b*4,b*8,4,2,1)
        self.bot=Block(b*8,b*8)
        self.u3=nn.ConvTranspose2d(b*8,b*4,4,2,1);self.c3=Block(b*4,b*4)
        self.u2=nn.ConvTranspose2d(b*4,b*2,4,2,1);self.c2=Block(b*2,b*2)
        self.u1=nn.ConvTranspose2d(b*2,b,4,2,1);self.c1=Block(b,b)
        self.out=nn.Conv2d(b,1,1);self.pool=nn.AdaptiveAvgPool2d((2,2))
    def forward(self,x,emb=False):
        x=self.e1(x);x=F.relu(self.d1(x),inplace=True)
        x=self.e2(x);x=F.relu(self.d2(x),inplace=True)
        x=self.e3(x);x=F.relu(self.d3(x),inplace=True)
        z=self.bot(x); e=self.pool(z).flatten(1)
        x=F.relu(self.u3(z),inplace=True);x=self.c3(x)
        x=F.relu(self.u2(x),inplace=True);x=self.c2(x)
        x=F.relu(self.u1(x),inplace=True);x=self.c1(x)
        y=self.out(x)
        return (y,e) if emb else y

def train_epoch(model,loader,opt,scaler,device,amp):
    model.train();sq=0.0;n=0.0
    for x,v,_ in loader:
        x=x.to(device,non_blocking=True);v=v.to(device,non_blocking=True);opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda',dtype=torch.float16,enabled=amp):
            y=model(x); loss=(((y-x)**2)*v).sum()/v.sum().clamp_min(1)
        scaler.scale(loss).backward();scaler.step(opt);scaler.update()
        with torch.no_grad():sq+=float((((y-x)**2)*v).sum().cpu());n+=float(v.sum().cpu())
    return sq/max(n,1)

@torch.no_grad()
def evaluate(model,paths,device,out,prefix,save_examples):
    model.eval();patch_rows=[];image_rows=[];pemb=[];pimg=[];pname=[];iemb=[];inames=[];examples=[]
    for ii,p in enumerate(paths):
        a=load_gray(p);h,w=a.shape;bb=boxes(w,h);rec=np.zeros_like(a);es=[]
        for name in PATCHES:
            x0,y0,x1,y1=bb[name];q=a[y0:y1,x0:x1];ph,pw=q.shape;H,W=rup(ph),rup(pw)
            x=torch.from_numpy(q).float()[None,None];x=F.pad(x,(0,W-pw,0,H-ph),mode='replicate').to(device)
            y,e=model(x,emb=True);y=torch.clamp(y,0,1)[0,0,:ph,:pw].cpu().numpy();e=e[0].cpu().numpy().astype(np.float32)
            rec[y0:y1,x0:x1]=y;m=metrics(q,y);patch_rows.append({'image':p.name,'patch':name,'patch_width':pw,'patch_height':ph,**m})
            pemb.append(e);pimg.append(p.name);pname.append(name);es.append(e)
        image_rows.append({'image':p.name,'width':w,'height':h,**metrics(a,rec)})
        iemb.append(np.mean(np.stack(es),axis=0));inames.append(p.name)
        if len(examples)<save_examples:examples.append((p.name,a,rec))
        if (ii+1)%25==0:print(f'  {prefix}: {ii+1}/{len(paths)} images')
    np.savez_compressed(out/f'{prefix}_patch_embeddings_1024d.npz',embeddings=np.stack(pemb),image=np.array(pimg),patch=np.array(pname))
    np.savez_compressed(out/f'{prefix}_image_embeddings_1024d.npz',embeddings=np.stack(iemb),image=np.array(inames))
    write_csv(out/f'{prefix}_patch_metrics.csv',patch_rows);write_csv(out/f'{prefix}_image_metrics.csv',image_rows)
    if examples:
        thumb=250;rh=285;can=Image.new('RGB',(thumb*3+80,rh*len(examples)+35),'white');dr=ImageDraw.Draw(can)
        for j,t in enumerate(['Original','Reconstructed','Abs Error (norm)']):dr.text((10+j*(thumb+20),10),t,fill='black')
        y0=35
        for name,a,r in examples:
            er=np.abs(a-r);er=er/(er.max()+1e-8);dr.text((10,y0),name,fill='black')
            for j,z in enumerate([a,r,er]):
                im=Image.fromarray((np.clip(z,0,1)*255).astype(np.uint8),'L');sc=min(thumb/im.width,(rh-30)/im.height);sz=(max(1,int(im.width*sc)),max(1,int(im.height*sc)))
                im=im.resize(sz,Image.Resampling.NEAREST).convert('RGB');can.paste(im,(10+j*(thumb+20),y0+20))
            y0+=rh
        can.save(out/f'{prefix}_reconstruction_examples.png')
    return patch_rows,image_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset_root',required=True);ap.add_argument('--output_dir',required=True)
    ap.add_argument('--base_channels',type=int,default=32);ap.add_argument('--batch_size',type=int,default=4);ap.add_argument('--epochs',type=int,default=200)
    ap.add_argument('--lr',type=float,default=1e-4);ap.add_argument('--min_lr',type=float,default=1e-6);ap.add_argument('--workers',type=int,default=2);ap.add_argument('--seed',type=int,default=42);ap.add_argument('--save_examples',type=int,default=10)
    a=ap.parse_args();random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    root=Path(a.dataset_root);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    tr=list_images(root/'classification_task'/'train'/'no_tumor');te=list_images(root/'classification_task'/'test'/'no_tumor')
    print('='*84);print('HEALTHY NATIVE-PATCH PLAIN CNN AUTOENCODER');print('='*84)
    print(f'Healthy train images : {len(tr)}\nHealthy test images  : {len(te)}\nTrain patches        : {len(tr)*4}\nTest patches         : {len(te)*4}')
    print('Spatial resize       : NONE\nPatch strategy       : native 2x2\nSkip connections     : NONE\nBatchNorm            : NONE\nLatent embedding     : 1024-D')
    ds=PatchDS(tr);dl=DataLoader(ds,batch_size=a.batch_size,shuffle=True,num_workers=a.workers,pin_memory=torch.cuda.is_available(),persistent_workers=a.workers>0,collate_fn=collate)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=dev.type=='cuda';model=PlainAE(a.base_channels).to(dev);params=sum(p.numel() for p in model.parameters())
    opt=torch.optim.Adam(model.parameters(),lr=a.lr);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs,eta_min=a.min_lr);scaler=torch.amp.GradScaler('cuda',enabled=amp)
    hist=[];best=float('inf');print(f'Device               : {dev}\nParameters           : {params:,}')
    for ep in range(1,a.epochs+1):
        mse=train_epoch(model,dl,opt,scaler,dev,amp);p=psnr(mse);lr=opt.param_groups[0]['lr'];hist.append({'epoch':ep,'lr':lr,'train_mse':mse,'train_psnr_db':p});write_csv(out/'history.csv',hist)
        ck={'model_state_dict':model.state_dict(),'epoch':ep,'train_mse':mse,'base_channels':a.base_channels,'architecture':'PlainAE_no_skip_no_bn_native2x2','latent_dim':a.base_channels*8*2*2,'healthy_only':True,'spatial_resize':False}
        torch.save(ck,out/'last_autoencoder.pt')
        if mse<best:best=mse;torch.save(ck,out/'best_train_autoencoder.pt');mark='  <-- BEST TRAIN'
        else:mark=''
        print(f'Epoch {ep:03d}/{a.epochs} | lr={lr:.3e} | train_mse={mse:.8e} | train_psnr={p:.2f} dB{mark}');sch.step()
    print('\nEvaluating healthy TRAIN for embeddings/diagnostics...');trp,tri=evaluate(model,tr,dev,out,'train_healthy',0)
    print('\nEvaluating independent healthy TEST...');tep,tei=evaluate(model,te,dev,out,'test_healthy',a.save_examples)
    final={'experiment':'healthy-only native 2x2 patch plain CNN autoencoder','healthy_only_training':True,'spatial_resize':False,'native_patch_strategy':'fixed 2x2','temporary_padding':'replicate padding, cropped before metrics','skip_connections':False,'batchnorm':False,'latent_embedding_dim':a.base_channels*8*2*2,'train_images':len(tr),'train_patches':len(ds),'test_images':len(te),'test_patches':len(te)*4,'epochs':a.epochs,'base_channels':a.base_channels,'batch_size':a.batch_size,'initial_lr':a.lr,'min_lr':a.min_lr,'model_params':params,'train_patch_metrics':summary(trp),'train_whole_image_metrics':summary(tri),'test_patch_metrics':summary(tep),'test_whole_image_metrics':summary(tei),'primary_result':'test_whole_image_metrics'}
    (out/'final_summary.json').write_text(json.dumps(final,indent=2),encoding='utf-8')
    print('\n=== FINAL HEALTHY TEST - WHOLE NATIVE IMAGE ===');print(json.dumps(summary(tei),indent=2));print('\n=== FINAL HEALTHY TEST - PATCH LEVEL ===');print(json.dumps(summary(tep),indent=2));print('\nSaved outputs:',out)
if __name__=='__main__':main()
