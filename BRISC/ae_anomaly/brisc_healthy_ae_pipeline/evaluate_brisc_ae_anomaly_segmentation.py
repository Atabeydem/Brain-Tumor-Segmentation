import argparse, json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from brisc_ae_common import list_images, load_grayscale_tensor, load_checkpoint_model, tensor_to_pil, save_contact_sheet, binary_metrics, write_csv

def match_mask(image_path: Path, mask_files_by_stem: dict):
    if image_path.stem in mask_files_by_stem: return mask_files_by_stem[image_path.stem]
    candidates=[image_path.stem.replace('_image',''), image_path.stem.replace('-image',''), image_path.stem.replace('image','mask')]
    for c in candidates:
        if c in mask_files_by_stem: return mask_files_by_stem[c]
    return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True); ap.add_argument('--dataset_root', default=r'C:\Users\Atabey\Downloads\brisc_Dataset2025\brisc2025')
    ap.add_argument('--split', default='test', choices=['train','test']); ap.add_argument('--output_dir', required=True)
    ap.add_argument('--thresholds', type=float, nargs='+', default=[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.14,0.16,0.18,0.20,0.25,0.30])
    ap.add_argument('--max_cases', type=int, default=0); ap.add_argument('--save_examples', type=int, default=12)
    args=ap.parse_args(); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); model, config, image_size=load_checkpoint_model(Path(args.checkpoint),device)
    root=Path(args.dataset_root); img_dir=root/'segmentation_task'/args.split/'images'; mask_dir=root/'segmentation_task'/args.split/'masks'
    images=list_images(img_dir); masks=list_images(mask_dir); mask_by_stem={p.stem:p for p in masks}
    if args.max_cases and args.max_cases>0: images=images[:args.max_cases]
    out_dir=Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    per_case_rows=[]; example_rows=[]; metrics_by_th={th:[] for th in args.thresholds}; model.eval()
    with torch.no_grad():
        for idx,img_path in enumerate(images):
            mask_path=match_mask(img_path,mask_by_stem)
            if mask_path is None:
                print(f'WARNING: no mask match for {img_path.name}; skipped'); continue
            x=load_grayscale_tensor(img_path,image_size,nearest=False).unsqueeze(0).to(device)
            gt=load_grayscale_tensor(mask_path,image_size,nearest=True).squeeze().numpy()>0.5
            recon=model(x).detach().cpu().squeeze(0); orig=x.detach().cpu().squeeze(0); err=torch.abs(orig-recon).squeeze().numpy()
            if len(example_rows)<args.save_examples:
                err_norm=err/(err.max()+1e-8); err_img=Image.fromarray((err_norm*255).astype(np.uint8)).convert('L'); gt_img=Image.fromarray((gt.astype(np.uint8)*255)).convert('L')
                pred_mid=err>=0.08; pred_img=Image.fromarray((pred_mid.astype(np.uint8)*255)).convert('L')
                example_rows.append((img_path.name,[tensor_to_pil(orig),tensor_to_pil(recon),err_img,pred_img,gt_img]))
            for th in args.thresholds:
                m=binary_metrics(err>=th, gt); m.update({'threshold':th,'image':img_path.name}); metrics_by_th[th].append(m); per_case_rows.append(m)
            if (idx+1)%100==0: print(f'Processed {idx+1}/{len(images)}')
    per_threshold_rows=[]
    for th,ms in metrics_by_th.items():
        if not ms: continue
        per_threshold_rows.append({'threshold':th,'n_cases':len(ms),'mean_dice':float(np.mean([m['dice'] for m in ms])),'mean_iou':float(np.mean([m['iou'] for m in ms])),'mean_precision':float(np.mean([m['precision'] for m in ms])),'mean_recall':float(np.mean([m['recall'] for m in ms]))})
    per_threshold_rows=sorted(per_threshold_rows,key=lambda r:r['mean_dice'],reverse=True); best=per_threshold_rows[0] if per_threshold_rows else {}
    write_csv(out_dir/'anomaly_segmentation_per_threshold.csv',per_threshold_rows); write_csv(out_dir/'anomaly_segmentation_per_case.csv',per_case_rows)
    summary={'checkpoint':str(args.checkpoint),'split':args.split,'image_size':image_size,'best':best,'note':'Prediction is generated from error map: abs(original - reconstructed) >= threshold.'}
    (out_dir/'anomaly_segmentation_summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    save_contact_sheet(example_rows,['Original','Reconstructed','Error map','Pred th=0.08','GT mask'],out_dir/'anomaly_examples_contact_sheet.png',cell_size=140)
    print(json.dumps(summary, indent=2, ensure_ascii=False)); print('\nSaved outputs:',out_dir)
if __name__=='__main__': main()
