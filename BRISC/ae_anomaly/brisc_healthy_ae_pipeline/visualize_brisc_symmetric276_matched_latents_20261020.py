#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Geometry-matched BRISC latent overlays using the FROZEN symmetric-276 AE.

Compare FULL 256x256 patches from EXACTLY 512x512 source images, stratified by
(row_idx,col_idx). No image resizing and no re-training. Healthy embeddings
are read from healthy TRAIN NPZ; tumor-source embeddings are extracted from
SEGMENTATION TRAIN. Optional mask-negative tumor-source control separates
source-domain effects from actual tumor-positive-patch effects.

Produce joint-fit PCA/LLE/t-SNE in 2D and 3D, with interactive 3D HTML.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import LocallyLinearEmbedding, TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.graph_objects as go

import train_eval_brisc_healthy_symmetric276_resnet_20261019 as ae

HEALTHY='Healthy train'
TUMOR='Tumor-positive segtrain'
NEGATIVE='Tumor-source mask-negative segtrain'
PALETTE={HEALTHY:'#2563eb',TUMOR:'#dc2626',NEGATIVE:'#059669'}
POSITIONS=((0,0),(0,1),(1,0),(1,1))


def arguments():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--healthy_npz',type=Path,required=True)
    p.add_argument('--output_dir',type=Path,required=True)
    p.add_argument('--max_per_position',type=int,default=130,help='Per-class cap at each of 4 positions')
    p.add_argument('--include_mask_negative',action='store_true')
    p.add_argument('--batch_size',type=int,default=2)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--neighbors',type=int,default=20)
    p.add_argument('--perplexity',type=float,default=30.0)
    return p.parse_args()


def geometry_ok(rec):
    return (int(rec['original_width'])==512 and int(rec['original_height'])==512
            and int(rec['valid_width'])==256 and int(rec['valid_height'])==256
            and (int(rec['row_idx']),int(rec['col_idx'])) in POSITIONS)


def random_sample(indices,n,rng):
    indices=np.asarray(indices,dtype=int)
    return sorted(rng.choice(indices,size=n,replace=False).tolist()) if len(indices)>n else sorted(indices.tolist())


def load_checkpoint_model(checkpoint,device):
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    base=int(ck.get('base_channels',64))
    if base!=64:raise ValueError('This matched experiment expects the base64 AE')
    model=ae.NativeResNetAutoencoder(base_channels=base)
    model.load_state_dict(ck['model_state_dict'],strict=True)
    model.to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


@torch.inference_mode()
def encode_rows(rows,model,device,batch_size):
    result=[]
    cached_name,cached_img=None,None
    # Preserve selected-row ordering so features and metadata remain aligned.
    for start in range(0,len(rows),batch_size):
        data=[]
        for r in rows[start:start+batch_size]:
            if r['source_image'] != cached_name:
                cached_name=r['source_image']
                cached_img=ae.load_native_gray(Path(cached_name))
            x0,y0=int(r['x0']),int(r['y0'])
            patch=cached_img[y0:int(r['y1_valid']),x0:int(r['x1_valid'])]
            if patch.shape!=(256,256):raise AssertionError(f'Invalid 512/full patch: {r["patch_id"]}')
            data.append(ae.pad_context_276(patch))
        tensor=torch.from_numpy(np.stack(data)[:,None]).float().to(device)
        internal=ae.torch_symmetric_pad2(tensor)
        emb=model.latent_embedding(model.encode_features(internal))
        result.append(emb.cpu().numpy().astype(np.float32))
        if ((start//batch_size)%100==0):print(f'  encoded {min(start+batch_size,len(rows))}/{len(rows)}')
    return np.vstack(result)


def plot_2d(y,meta,name,out):
    fig,ax=plt.subplots(figsize=(9,7))
    for label,color in PALETTE.items():
        mask=(meta['group'].to_numpy()==label)
        if not mask.any():continue
        ax.scatter(y[mask,0],y[mask,1],c=color,s=9,alpha=.48,
                   label=f'{label} (n={mask.sum()})',rasterized=True)
    ax.set(xlabel=f'{name} 1',ylabel=f'{name} 2',
           title=f'{name}: matched full 256px patches, 512px sources (joint fit)')
    ax.legend(loc='best',fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(out/f'{name.lower()}_2d.png',dpi=180);plt.close(fig)


def plot_3d(y,meta,name,out):
    fig=plt.figure(figsize=(9,7));ax=fig.add_subplot(111,projection='3d')
    graph=go.Figure()
    for label,color in PALETTE.items():
        mask=(meta['group'].to_numpy()==label)
        if not mask.any():continue
        ax.scatter(y[mask,0],y[mask,1],y[mask,2],c=color,s=7,alpha=.35,
                   label=f'{label} (n={mask.sum()})',rasterized=True)
        subset=meta.loc[mask]
        hover=[f'{r.image}<br>{r.patch_id}<br>position={r.row_idx},{r.col_idx}<br>tumor_fraction={r.tumor_fraction:.4f}'
               for r in subset.itertuples(index=False)]
        graph.add_trace(go.Scatter3d(x=y[mask,0],y=y[mask,1],z=y[mask,2],
                                     mode='markers',name=label,text=hover,hoverinfo='text',
                                     marker={'size':2.8,'color':color,'opacity':.65}))
    ax.set(xlabel=f'{name} 1',ylabel=f'{name} 2',zlabel=f'{name} 3')
    ax.legend(loc='upper right',fontsize=8,frameon=False)
    fig.tight_layout();fig.savefig(out/f'{name.lower()}_3d.png',dpi=180);plt.close(fig)
    graph.update_layout(title=f'{name} 3D — geometry matched (joint fit)',
                        scene=dict(xaxis_title='Component 1',yaxis_title='Component 2',zaxis_title='Component 3'),
                        margin=dict(l=0,r=0,b=0,t=45),legend=dict(orientation='h',y=1.09))
    graph.write_html(out/f'{name.lower()}_3d_interactive.html',include_plotlyjs=True)


def save_coords(y,meta,label,out):
    df=meta.copy()
    for d in range(y.shape[1]):df[f'component_{d+1}']=y[:,d]
    df.to_csv(out/f'{label.lower()}_{y.shape[1]}d_coordinates.csv',index=False)


def main():
    a=arguments()
    for p in (a.manifest,a.healthy_npz,a.checkpoint):
        if not p.is_file():raise FileNotFoundError(p)
    if a.max_per_position<5 or a.batch_size<1:raise ValueError('max_per_position >=5, batch_size>=1')
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with np.load(a.healthy_npz,allow_pickle=False) as f:
        required=('embeddings','source_image','image','row_idx','col_idx','original_width',
                  'original_height','valid_width','valid_height','patch_id')
        if any(k not in f.files for k in required):raise KeyError('Healthy NPZ missing required geometry metadata')
        healthy_x=f['embeddings'].astype(np.float32)
        healthy_meta=pd.DataFrame({k:f[k] for k in required if k!='embeddings'})
    if healthy_x.shape[1]!=2048:raise ValueError('Healthy features are not 2048-D')
    manifest=ae.read_manifest(a.manifest)
    positive=[r for r in manifest if r['split']=='tumor_train_seg_only_for_analysis' and int(r['patch_label'])==1 and geometry_ok(r)]
    negative=[r for r in manifest if r['split']=='tumor_train_seg_only_for_analysis' and int(r['patch_label'])==0 and geometry_ok(r)]
    if not positive:raise ValueError('No 512x512 tumor-positive SEGTRAIN patches')
    classes=[HEALTHY,TUMOR]+([NEGATIVE] if a.include_mask_negative else [])
    rng=np.random.default_rng(a.seed)
    indices_by_class={k:[] for k in classes}
    summary=[]
    for row,col in POSITIONS:
        good=healthy_meta.index[(healthy_meta['original_width']==512)&(healthy_meta['original_height']==512)&
              (healthy_meta['valid_width']==256)&(healthy_meta['valid_height']==256)&
              (healthy_meta['row_idx']==row)&(healthy_meta['col_idx']==col)].to_numpy(dtype=int)
        pos=np.array([i for i,r in enumerate(positive) if int(r['row_idx'])==row and int(r['col_idx'])==col],dtype=int)
        neg=np.array([i for i,r in enumerate(negative) if int(r['row_idx'])==row and int(r['col_idx'])==col],dtype=int)
        present={HEALTHY:good,TUMOR:pos,NEGATIVE:neg}
        n=min(a.max_per_position,*[len(present[k]) for k in classes])
        if n<5:
            print(f'WARNING: stratum {(row,col)} insufficient matched samples (min={n}), skipping.')
            continue
        for k in classes:indices_by_class[k].extend(random_sample(present[k],n,rng))
        summary.append({'position':f'r{row}c{col}','sample_per_class':n,
                        'healthy_available':len(good),'tumor_positive_available':len(pos),
                        'tumor_source_negative_available':len(neg)})
        print(f'r{row}c{col}: healthy={len(good)}, tumor+={len(pos)}, tumor-maskneg={len(neg)}; matched n={n}/group')
    if not summary:raise RuntimeError('No positions with enough matched samples')
    hi=indices_by_class[HEALTHY]
    xx=[healthy_x[hi]]
    hm=healthy_meta.iloc[hi].copy()
    hm['group']=HEALTHY;hm['tumor_fraction']=0.0
    cols=['group','image','patch_id','row_idx','col_idx','valid_width','valid_height','original_width','original_height','tumor_fraction']
    mm=[hm[cols]]
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=load_checkpoint_model(a.checkpoint,device)
    for label,source in ((TUMOR,positive),(NEGATIVE,negative)):
        if label not in classes:continue
        rows=sorted([source[i] for i in indices_by_class[label]],
                    key=lambda r:(r['source_image'],int(r['row_idx']),int(r['col_idx'])))
        print(f'Encoding {label}: {len(rows)} patches | device={device}')
        feature=encode_rows(rows,model,device,a.batch_size)
        met=pd.DataFrame([{'group':label,'image':Path(r['source_image']).name,'patch_id':r['patch_id'],
                           'row_idx':int(r['row_idx']),'col_idx':int(r['col_idx']),
                           'valid_width':256,'valid_height':256,'original_width':512,'original_height':512,
                           'tumor_fraction':float(r.get('tumor_fraction',0))} for r in rows])
        xx.append(feature);mm.append(met[cols])
    x=np.concatenate(xx,axis=0)
    meta=pd.concat(mm,ignore_index=True)
    if not np.isfinite(x).all():raise ValueError('Non-finite latent features')
    n=len(x)
    print(f'Joint projection: {n} × {x.shape[1]}, groups={list(meta.group.value_counts().items())}')
    meta.to_csv(a.output_dir/'matched_sample_metadata.csv',index=False)
    np.savez_compressed(a.output_dir/'matched_embeddings_2048d.npz',embeddings=x,groups=meta['group'].astype(str).to_numpy())
    y=PCA(n_components=3,svd_solver='randomized',random_state=a.seed).fit_transform(x)
    plot_2d(y[:,:2],meta,'PCA',a.output_dir);plot_3d(y,meta,'PCA',a.output_dir)
    save_coords(y[:,:2],meta,'PCA',a.output_dir);save_coords(y,meta,'PCA',a.output_dir)
    pre_n=min(50,n-2,x.shape[1]);xp=PCA(n_components=pre_n,svd_solver='randomized',random_state=a.seed).fit_transform(x)
    neighbors=min(a.neighbors,n-2)
    perplexity=min(a.perplexity,max(2,(n-1)/3))
    for method in ('LLE','tSNE'):
        for ndim in (2,3):
            print(f'Computing {method} {ndim}D on matched data...')
            if method=='LLE':
                projector=LocallyLinearEmbedding(n_neighbors=neighbors,n_components=ndim,
                                                 method='standard',eigen_solver='arpack',random_state=a.seed)
            else:
                projector=TSNE(n_components=ndim,perplexity=perplexity,learning_rate='auto',
                               init='pca',random_state=a.seed,max_iter=1000)
            coords=projector.fit_transform(xp)
            save_coords(coords,meta,method,a.output_dir)
            (plot_2d if ndim==2 else plot_3d)(coords,meta,method,a.output_dir)
    summary_json={'matched_geometry':'512x512 source, full 256x256 patch, exact row/col stratum',
                  'no_resize':True,'frozen_encoder':True,'tumor_source':'SEGMENTATION TRAIN',
                  'use_masknegative_control':a.include_mask_negative,
                  'classes':classes,'per_position':summary,'samples_total':int(n),'seed':a.seed,
                  'projection_fit':'JOINT; LLE/tSNE PCA-50 preprojection',
                  'caution':'Geometry matching does not remove scanner/domain/site or patient-level confounds'}
    (a.output_dir/'matching_summary.json').write_text(json.dumps(summary_json,indent=2),encoding='utf-8')
    print(json.dumps(summary_json,indent=2))
    print('FINISHED:',a.output_dir.resolve())

if __name__=='__main__':main()
