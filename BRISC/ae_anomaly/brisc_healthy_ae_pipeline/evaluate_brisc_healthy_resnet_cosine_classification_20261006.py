
import argparse, csv, json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
    HAS_SK=True
except Exception:
    HAS_SK=False

try:
    import matplotlib.pyplot as plt
    HAS_MPL=True
except Exception:
    HAS_MPL=False

EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}
PATCHES=("TL","TR","BL","BR")

def list_images(folder):
    folder=Path(folder)
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def boxes(w,h):
    xm,ym=w//2,h//2
    return {"TL":(0,0,xm,ym),"TR":(xm,0,w,ym),"BL":(0,ym,xm,h),"BR":(xm,ym,w,h)}

def roundup(v,m=8):
    return ((v+m-1)//m)*m

def load_gray(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("L"),dtype=np.float32)/255.0

class ResBlock(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.conv1=nn.Conv2d(c,c,3,padding=1,bias=True)
        self.conv2=nn.Conv2d(c,c,3,padding=1,bias=True)
    def forward(self,x):
        y=F.relu(self.conv1(x),inplace=True)
        y=self.conv2(y)
        return F.relu(y+x,inplace=True)

class NativeResNetAE(nn.Module):
    def __init__(self,b=32):
        super().__init__()
        self.stem=nn.Conv2d(1,b,3,padding=1,bias=True)
        self.enc1=ResBlock(b)
        self.down1=nn.Conv2d(b,b*2,4,2,1,bias=True)
        self.enc2=ResBlock(b*2)
        self.down2=nn.Conv2d(b*2,b*4,4,2,1,bias=True)
        self.enc3=ResBlock(b*4)
        self.down3=nn.Conv2d(b*4,b*8,4,2,1,bias=True)
        self.bot1=ResBlock(b*8)
        self.bot2=ResBlock(b*8)
        self.up3=nn.ConvTranspose2d(b*8,b*4,4,2,1,bias=True)
        self.dec3=ResBlock(b*4)
        self.up2=nn.ConvTranspose2d(b*4,b*2,4,2,1,bias=True)
        self.dec2=ResBlock(b*2)
        self.up1=nn.ConvTranspose2d(b*2,b,4,2,1,bias=True)
        self.dec1=ResBlock(b)
        self.out=nn.Conv2d(b,1,1,bias=True)
        self.pool=nn.AdaptiveAvgPool2d((2,2))
    def embed(self,x):
        x=F.relu(self.stem(x),inplace=True); x=self.enc1(x)
        x=F.relu(self.down1(x),inplace=True); x=self.enc2(x)
        x=F.relu(self.down2(x),inplace=True); x=self.enc3(x)
        x=F.relu(self.down3(x),inplace=True); x=self.bot1(x); x=self.bot2(x)
        return self.pool(x).flatten(1)

def norm(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def cosmat(a,b):
    return norm(a.astype(np.float64)) @ norm(b.astype(np.float64)).T

def load_npz(path):
    d=np.load(path,allow_pickle=False)
    return d["embeddings"].astype(np.float32), d["image"].astype(str)

@torch.no_grad()
def encode_image(model,path,device):
    arr=load_gray(path); h,w=arr.shape; embs=[]
    for name in PATCHES:
        x0,y0,x1,y1=boxes(w,h)[name]
        p=arr[y0:y1,x0:x1]; ph,pw=p.shape
        t=torch.from_numpy(p).float()[None,None]
        t=F.pad(t,(0,roundup(pw)-pw,0,roundup(ph)-ph),mode="replicate").to(device)
        embs.append(model.embed(t)[0].cpu().numpy().astype(np.float32))
    return np.mean(np.stack(embs),axis=0)

def write_csv(path,rows):
    if not rows: return
    with open(path,"w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_root",required=True)
    ap.add_argument("--checkpoint",required=True)
    ap.add_argument("--healthy_train_embeddings",required=True)
    ap.add_argument("--healthy_test_embeddings",required=True)
    ap.add_argument("--output_dir",required=True)
    ap.add_argument("--healthy_acceptance",type=float,default=0.95)
    ap.add_argument("--base_channels",type=int,default=32)
    a=ap.parse_args()

    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt=torch.load(a.checkpoint,map_location=device)
    b=int(ckpt.get("base_channels",a.base_channels))
    model=NativeResNetAE(b).to(device)
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    model.eval()

    train_emb,train_names=load_npz(a.healthy_train_embeddings)
    test_emb,test_names=load_npz(a.healthy_test_embeddings)

    sim=cosmat(train_emb,train_emb); np.fill_diagonal(sim,-np.inf)
    idx=np.argmax(sim,axis=1); loo=sim[np.arange(len(train_emb)),idx]
    pct=100.0*(1.0-a.healthy_acceptance)
    th=float(np.percentile(loo,pct))
    write_csv(out/"healthy_train_leave_one_out_similarity.csv",[
        {"image":train_names[i],"nearest_healthy_image":train_names[idx[i]],
         "nearest_cosine_similarity":float(loo[i])} for i in range(len(train_emb))
    ])

    hsim=cosmat(test_emb,train_emb); hidx=np.argmax(hsim,axis=1)
    hscores=hsim[np.arange(len(test_emb)),hidx]

    rows=[]
    for i,s in enumerate(hscores):
        pred=int(s<th)
        rows.append({"image":test_names[i],"source_class":"no_tumor","true_label":0,"true_name":"healthy",
                     "nearest_healthy_image":train_names[hidx[i]],"cosine_similarity":float(s),
                     "predicted_label":pred,"predicted_name":"tumor_or_anomaly" if pred else "healthy"})

    test_root=Path(a.dataset_root)/"classification_task"/"test"
    tumor_emb=[]; tumor_meta=[]
    print("Encoding tumor test images for evaluation only...")
    for d in sorted(p for p in test_root.iterdir() if p.is_dir() and p.name.lower()!="no_tumor"):
        ims=list_images(d); print(f"  {d.name}: {len(ims)} images")
        for j,p in enumerate(ims,1):
            tumor_emb.append(encode_image(model,p,device)); tumor_meta.append((p.name,d.name))
            if j%100==0: print(f"    encoded {j}/{len(ims)}")

    tumor_emb=np.stack(tumor_emb)
    tsim=cosmat(tumor_emb,train_emb); tidx=np.argmax(tsim,axis=1)
    tscores=tsim[np.arange(len(tumor_emb)),tidx]
    for i,(name,cls) in enumerate(tumor_meta):
        s=float(tscores[i]); pred=int(s<th)
        rows.append({"image":name,"source_class":cls,"true_label":1,"true_name":"tumor",
                     "nearest_healthy_image":train_names[tidx[i]],"cosine_similarity":s,
                     "predicted_label":pred,"predicted_name":"tumor_or_anomaly" if pred else "healthy"})

    write_csv(out/"cosine_classification_per_image.csv",rows)

    y=np.array([r["true_label"] for r in rows]); yp=np.array([r["predicted_label"] for r in rows])
    ss=np.array([r["cosine_similarity"] for r in rows])
    anomaly=-ss
    hs=ss[y==0]; ts=ss[y==1]

    if HAS_SK:
        tn,fp,fn,tp=[int(x) for x in confusion_matrix(y,yp,labels=[0,1]).ravel()]
        acc=float(accuracy_score(y,yp)); prec=float(precision_score(y,yp,zero_division=0))
        rec=float(recall_score(y,yp,zero_division=0)); f1=float(f1_score(y,yp,zero_division=0))
        spec=float(tn/(tn+fp)) if tn+fp else None
        auc=float(roc_auc_score(y,anomaly))
    else:
        tn=int(np.sum((y==0)&(yp==0))); fp=int(np.sum((y==0)&(yp==1)))
        fn=int(np.sum((y==1)&(yp==0))); tp=int(np.sum((y==1)&(yp==1)))
        acc=float((tn+tp)/len(y)); prec=float(tp/(tp+fp)) if tp+fp else 0.0
        rec=float(tp/(tp+fn)) if tp+fn else 0.0; spec=float(tn/(tn+fp)) if tn+fp else None
        f1=float(2*prec*rec/(prec+rec)) if prec+rec else 0.0; auc=None

    classes={}
    for cls in sorted(set(r["source_class"] for r in rows)):
        rr=[r for r in rows if r["source_class"]==cls]
        vals=np.array([r["cosine_similarity"] for r in rr])
        classes[cls]={
            "n":len(rr),"mean_cosine_similarity":float(vals.mean()),"std_cosine_similarity":float(vals.std()),
            "min_cosine_similarity":float(vals.min()),"max_cosine_similarity":float(vals.max()),
            "classification_accuracy":float(np.mean([r["true_label"]==r["predicted_label"] for r in rr]))
        }

    summary={
        "experiment":"ResNet healthy-reference cosine similarity classification",
        "encoder":"healthy-only native ResNet autoencoder",
        "long_unet_skips":False,"local_residual_connections":True,
        "healthy_train_reference_n":len(train_emb),"healthy_test_n":int(np.sum(y==0)),
        "tumor_test_n":int(np.sum(y==1)),"latent_dim":train_emb.shape[1],
        "similarity_method":"maximum cosine similarity to healthy-train image embedding bank",
        "threshold_method":f"healthy-train leave-one-out nearest-neighbor cosine; lower {pct:.2f}th percentile",
        "healthy_acceptance_target":a.healthy_acceptance,"threshold":th,
        "healthy_train_loo_similarity":{"mean":float(loo.mean()),"std":float(loo.std()),
                                        "min":float(loo.min()),"max":float(loo.max())},
        "test_similarity":{"healthy_mean":float(hs.mean()),"healthy_std":float(hs.std()),
                           "tumor_mean":float(ts.mean()),"tumor_std":float(ts.std())},
        "classification":{"tn_healthy_correct":tn,"fp_healthy_as_tumor":fp,
                          "fn_tumor_as_healthy":fn,"tp_tumor_correct":tp,
                          "accuracy":acc,"precision_tumor":prec,"recall_sensitivity_tumor":rec,
                          "specificity_healthy":spec,"f1_tumor":f1,"roc_auc_anomaly_score":auc},
        "per_source_class":classes,
        "important_note":"ResNet AE was trained only on healthy images; tumor images are evaluation only; threshold uses only healthy-train embeddings."
    }
    (out/"cosine_classification_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")

    if HAS_MPL:
        plt.figure(figsize=(8,5))
        plt.hist(hs,bins=30,alpha=.6,label="Healthy test")
        plt.hist(ts,bins=30,alpha=.6,label="Tumor test")
        plt.axvline(th,linestyle="--",label=f"Threshold={th:.5f}")
        plt.xlabel("Maximum cosine similarity to healthy reference bank")
        plt.ylabel("Image count")
        plt.title("ResNet healthy-reference cosine similarity classification")
        plt.legend(); plt.tight_layout()
        plt.savefig(out/"cosine_similarity_distribution.png",dpi=180); plt.close()

    print("="*90)
    print("FINAL RESNET COSINE CLASSIFICATION")
    print("="*90)
    print(f"Healthy test n              : {int(np.sum(y==0))}")
    print(f"Tumor test n                : {int(np.sum(y==1))}")
    print(f"Threshold                   : {th:.8f}")
    print(f"Healthy mean cosine         : {float(hs.mean()):.8f}")
    print(f"Tumor mean cosine           : {float(ts.mean()):.8f}")
    print(f"TN healthy correct          : {tn}")
    print(f"FP healthy -> tumor         : {fp}")
    print(f"FN tumor -> healthy         : {fn}")
    print(f"TP tumor correct            : {tp}")
    print(f"Accuracy                    : {acc:.6f}")
    print(f"Tumor Precision             : {prec:.6f}")
    print(f"Tumor Recall / Sensitivity  : {rec:.6f}")
    print(f"Healthy Specificity         : {spec:.6f}")
    print(f"Tumor F1                    : {f1:.6f}")
    if auc is not None: print(f"ROC-AUC                     : {auc:.6f}")
    print("Per source class:")
    for k,v in classes.items():
        print(f"  {k:15s} n={v['n']:4d} | mean cosine={v['mean_cosine_similarity']:.8f} | acc={v['classification_accuracy']:.4f}")
    print("Saved outputs:",out)

if __name__=="__main__":
    main()
