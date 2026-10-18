#!/usr/bin/env python
import argparse, csv, json
from collections import Counter
from pathlib import Path
from PIL import Image

EXTS={".png",".jpg",".jpeg",".bmp",".tif",".tiff"}

def imgs(folder):
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)

def boxes(w,h):
    xm,ym=w//2,h//2
    return {"TL":(0,0,xm,ym),"TR":(xm,0,w,ym),"BL":(0,ym,xm,h),"BR":(xm,ym,w,h)}

def audit(folder,label):
    paths=imgs(folder); rows=[]; bad=[]; pc=Counter(); ic=Counter()
    for i,p in enumerate(paths,1):
        try:
            with Image.open(p) as im: w,h=im.size
            ic[(w,h)]+=1
            for name,(x0,y0,x1,y1) in boxes(w,h).items():
                pw,ph=x1-x0,y1-y0; pc[(pw,ph)]+=1
                rows.append(dict(split=label,source_image=p.name,source_width=w,source_height=h,
                                 patch=name,x0=x0,y0=y0,x1=x1,y1=y1,
                                 patch_width=pw,patch_height=ph,source_path=str(p)))
        except Exception as e:
            bad.append(dict(split=label,source_image=p.name,source_path=str(p),error=repr(e)))
        if i%250==0: print(f"{label}: checked {i}/{len(paths)}")
    s=dict(
        split=label,n_images=len(paths),n_patches=len(rows),n_failed=len(bad),
        unique_patch_sizes=len(pc),
        min_patch_width=min((w for w,h in pc),default=None),
        max_patch_width=max((w for w,h in pc),default=None),
        min_patch_height=min((h for w,h in pc),default=None),
        max_patch_height=max((h for w,h in pc),default=None),
        image_size_distribution=[{"width":w,"height":h,"count":c} for (w,h),c in sorted(ic.items(),key=lambda kv:(-kv[1],kv[0]))],
        patch_size_distribution=[{"width":w,"height":h,"count":c} for (w,h),c in sorted(pc.items(),key=lambda kv:(-kv[1],kv[0]))],
    )
    return s,rows,bad

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--dataset_root",required=True)
    ap.add_argument("--output_dir",default="./outputs/HEALTHY_NATIVE_2X2_PATCH_AUDIT")
    a=ap.parse_args()
    root=Path(a.dataset_root); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True)
    tr=root/"classification_task"/"train"/"no_tumor"
    te=root/"classification_task"/"test"/"no_tumor"
    print("="*78)
    print("BRISC HEALTHY NATIVE 2x2 PATCH AUDIT")
    print("NO resize. NO training. Only split coordinates/sizes are computed.")
    print("="*78)
    ts,trrows,trbad=audit(tr,"train_no_tumor")
    es,terows,tebad=audit(te,"test_no_tumor")
    rows=trrows+terows; bad=trbad+tebad
    with open(out/"healthy_native_2x2_patch_manifest.csv","w",newline="",encoding="utf-8") as f:
        fn=["split","source_image","source_width","source_height","patch","x0","y0","x1","y1","patch_width","patch_height","source_path"]
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(rows)
    with open(out/"healthy_native_2x2_patch_failures.csv","w",newline="",encoding="utf-8") as f:
        fn=["split","source_image","source_path","error"]
        w=csv.DictWriter(f,fieldnames=fn); w.writeheader(); w.writerows(bad)
    summary={
        "method":"native-resolution fixed 2x2 split",
        "resize_used_for_data":False,
        "training_performed":False,
        "train":ts,"test":es,
        "important_note":"Native 2x2 splitting preserves source pixels but creates variable-sized patches. Padding/model strategy must be decided after this audit; no resize is used here."
    }
    (out/"healthy_native_2x2_patch_audit_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    for s in (ts,es):
        print(f"\n=== {s['split']} ===")
        print(f"Images             : {s['n_images']}")
        print(f"Patches            : {s['n_patches']}")
        print(f"Failed             : {s['n_failed']}")
        print(f"Unique patch sizes : {s['unique_patch_sizes']}")
        print(f"Width range        : {s['min_patch_width']}..{s['max_patch_width']}")
        print(f"Height range       : {s['min_patch_height']}..{s['max_patch_height']}")
        print("Top patch sizes:")
        for x in s["patch_size_distribution"][:15]:
            print(f"  {x['width']}x{x['height']}: {x['count']}")
    print("\nSaved outputs:",out)
    print("IMPORTANT: No image was resized or modified.")

if __name__=="__main__":
    main()
