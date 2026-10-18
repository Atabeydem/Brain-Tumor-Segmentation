import argparse, json, random, csv
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from brisc_ae_common import list_images, set_seed, ImageReconstructionDataset, build_model, psnr_from_mse, save_contact_sheet, tensor_to_pil

def evaluate(model, loader, device):
    model.eval(); criterion=nn.MSELoss(reduction='mean'); losses=[]; psnrs=[]
    with torch.no_grad():
        for x,y,_ in loader:
            x=x.to(device); y=y.to(device); recon=model(x); loss=criterion(recon,y); losses.append(float(loss.item()))
            mse_per=((recon-y)**2).flatten(1).mean(dim=1).detach().cpu().numpy()
            psnrs.extend([psnr_from_mse(float(m)) for m in mse_per])
    return {'mse_loss': float(np.mean(losses)) if losses else float('nan'), 'mean_psnr': float(np.mean(psnrs)) if psnrs else float('nan')}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset_root', default=r'C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025')
    ap.add_argument('--output_dir', default=r'.\outputs\brisc_healthy_unet_ae_256_e50')
    ap.add_argument('--model', default='unet', choices=['cnn','unet']); ap.add_argument('--image_size', type=int, default=256)
    ap.add_argument('--base_channels', type=int, default=32); ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50); ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--val_split', type=float, default=0.15); ap.add_argument('--seed', type=int, default=42); ap.add_argument('--num_workers', type=int, default=0)
    args=ap.parse_args(); set_seed(args.seed)
    root=Path(args.dataset_root); healthy_train_dir=root/'classification_task'/'train'/'no_tumor'
    files=list_images(healthy_train_dir)
    if not files: raise RuntimeError(f'No images found: {healthy_train_dir}')
    random.Random(args.seed).shuffle(files); n_val=max(1,int(len(files)*args.val_split)); val_files=files[:n_val]; train_files=files[n_val:]
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    config=vars(args).copy(); config.update({'healthy_train_dir':str(healthy_train_dir),'n_train':len(train_files),'n_val':len(val_files)})
    (out_dir/'config.json').write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding='utf-8')
    train_loader=DataLoader(ImageReconstructionDataset(train_files,args.image_size), batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_ds=ImageReconstructionDataset(val_files,args.image_size)
    val_loader=DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model=build_model(args.model,args.base_channels).to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=args.lr); criterion=nn.MSELoss(); best_val=float('inf'); history=[]
    print('Device:',device); print('Model:',args.model); print('Healthy train:',len(train_files),'Healthy val:',len(val_files)); print('Output:',out_dir)
    for epoch in range(1,args.epochs+1):
        model.train(); train_losses=[]
        for x,y,_ in train_loader:
            x=x.to(device); y=y.to(device); opt.zero_grad(); recon=model(x); loss=criterion(recon,y); loss.backward(); opt.step(); train_losses.append(float(loss.item()))
        val=evaluate(model,val_loader,device); row={'epoch':epoch,'train_mse_loss':float(np.mean(train_losses)),'val_mse_loss':val['mse_loss'],'val_mean_psnr':val['mean_psnr']}; history.append(row)
        print(f"Epoch {epoch:03d}/{args.epochs} | train_mse={row['train_mse_loss']:.6f} | val_mse={row['val_mse_loss']:.6f} | val_psnr={row['val_mean_psnr']:.2f} dB")
        if row['val_mse_loss'] < best_val:
            best_val=row['val_mse_loss']
            torch.save({'model_state_dict':model.state_dict(),'config':config,'best_epoch':epoch,'best_val_mse_loss':best_val,'best_val_mean_psnr':row['val_mean_psnr']}, out_dir/'best_autoencoder.pt')
        if epoch==1 or epoch%10==0 or epoch==args.epochs:
            model.eval(); sample_rows=[]
            with torch.no_grad():
                for i in range(min(6,len(val_ds))):
                    x,_,path=val_ds[i]; recon=model(x.unsqueeze(0).to(device)).cpu().squeeze(0); sample_rows.append((Path(path).name,[tensor_to_pil(x),tensor_to_pil(recon)]))
            save_contact_sheet(sample_rows,['Original','Reconstructed'], out_dir/f'recon_samples_epoch_{epoch:03d}.png')
    with open(out_dir/'training_history.csv','w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)
    print('\nDone. Best checkpoint:', out_dir/'best_autoencoder.pt')
if __name__=='__main__': main()
