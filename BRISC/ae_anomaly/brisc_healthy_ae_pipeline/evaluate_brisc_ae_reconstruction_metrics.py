import argparse, json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from brisc_ae_common import list_images, ImageReconstructionDataset, load_checkpoint_model, mse_np, mae_np, psnr_from_mse, try_ssim, save_tensor_image, tensor_to_pil, save_contact_sheet, write_csv

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--dataset_root', default=r'C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025')
    ap.add_argument('--eval_folder', default='classification_task/test/no_tumor'); ap.add_argument('--output_dir', required=True)
    ap.add_argument('--max_images', type=int, default=0); ap.add_argument('--batch_size', type=int, default=8); ap.add_argument('--num_workers', type=int, default=0)
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model, config, image_size=load_checkpoint_model(Path(args.checkpoint),device)
    root=Path(args.dataset_root); eval_folder=Path(args.eval_folder); eval_folder=eval_folder if eval_folder.is_absolute() else root/eval_folder
    files=list_images(eval_folder); files=files[:args.max_images] if args.max_images and args.max_images>0 else files
    if not files: raise RuntimeError(f'No images found: {eval_folder}')
    out_dir=Path(args.output_dir); orig_dir=out_dir/'original'; recon_dir=out_dir/'reconstructed'; out_dir.mkdir(parents=True, exist_ok=True)
    loader=DataLoader(ImageReconstructionDataset(files,image_size), batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    rows=[]; sample_rows=[]; idx=0; model.eval()
    with torch.no_grad():
        for x,_,paths in loader:
            x=x.to(device); recon=model(x).detach().cpu(); x_cpu=x.detach().cpu()
            for b in range(x_cpu.shape[0]):
                orig_np=x_cpu[b].squeeze().numpy().clip(0,1); recon_np=recon[b].squeeze().numpy().clip(0,1)
                mse=mse_np(orig_np,recon_np); mae=mae_np(orig_np,recon_np); psnr=psnr_from_mse(mse); ssim_val=try_ssim(orig_np,recon_np)
                stem=f'{idx:04d}_{Path(paths[b]).stem}'; save_tensor_image(x_cpu[b],orig_dir/f'{stem}.png'); save_tensor_image(recon[b],recon_dir/f'{stem}.png')
                rows.append({'index':idx,'source_path':paths[b],'mse':mse,'mae':mae,'psnr_db':psnr,'ssim':'' if ssim_val is None else ssim_val})
                if len(sample_rows)<8: sample_rows.append((Path(paths[b]).name,[tensor_to_pil(x_cpu[b]),tensor_to_pil(recon[b])]))
                idx+=1
    write_csv(out_dir/'reconstruction_metrics.csv', rows)
    psnrs=np.array([r['psnr_db'] for r in rows],dtype=np.float64); mses=np.array([r['mse'] for r in rows],dtype=np.float64); maes=np.array([r['mae'] for r in rows],dtype=np.float64); svals=[r['ssim'] for r in rows if r['ssim']!='']
    summary={'checkpoint':str(args.checkpoint),'eval_folder':str(eval_folder),'n_images':len(rows),'mean_mse':float(np.mean(mses)),'std_mse':float(np.std(mses)),'mean_mae':float(np.mean(maes)),'std_mae':float(np.std(maes)),'mean_psnr_db':float(np.mean(psnrs)),'std_psnr_db':float(np.std(psnrs)),'min_psnr_db':float(np.min(psnrs)),'max_psnr_db':float(np.max(psnrs)),'mean_ssim':None if not svals else float(np.mean(np.array(svals,dtype=np.float64))),'std_ssim':None if not svals else float(np.std(np.array(svals,dtype=np.float64))),'note':'SSIM is None if scikit-image is not installed. Install with: pip install scikit-image'}
    (out_dir/'metrics_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    save_contact_sheet(sample_rows,['Original','Reconstructed'], out_dir/'reconstruction_contact_sheet.png')
    print(json.dumps(summary, indent=2, ensure_ascii=False)); print('\nOriginal:',orig_dir); print('Reconstructed:',recon_dir)
if __name__=='__main__': main()
