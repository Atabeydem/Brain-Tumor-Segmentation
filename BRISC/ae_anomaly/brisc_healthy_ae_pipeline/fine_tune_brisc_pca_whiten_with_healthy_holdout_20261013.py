#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

LEVELS = [
    ("exact", lambda r: (r["original_width"], r["original_height"], r["row_idx"], r["col_idx"], r["valid_width"], r["valid_height"])),
    ("resolution_position", lambda r: (r["original_width"], r["original_height"], r["row_idx"], r["col_idx"])),
    ("position_geometry", lambda r: (r["row_idx"], r["col_idx"], r["valid_width"], r["valid_height"])),
    ("position", lambda r: (r["row_idx"], r["col_idx"])),
    ("global", lambda r: ("global",)),
]

def read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows: return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

def rec(source_image, patch_id, row_idx, col_idx, valid_width, valid_height,
        original_width, original_height, label=0, group="healthy"):
    return {
        "source_image": str(source_image), "patch_id": str(patch_id),
        "row_idx": int(row_idx), "col_idx": int(col_idx),
        "valid_width": int(valid_width), "valid_height": int(valid_height),
        "original_width": int(original_width), "original_height": int(original_height),
        "label": int(label), "group": str(group),
    }

def load_healthy(path, group):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = [rec(d["source_image"][i], d["patch_id"][i], d["row_idx"][i], d["col_idx"][i],
                d["valid_width"][i], d["valid_height"][i], d["original_width"][i],
                d["original_height"][i], 0, group) for i in range(len(x))]
    return x, rows

def load_seg(path):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = []
    for i in range(len(x)):
        y = int(d["patch_label"][i])
        rows.append(rec(
            d["source_image"][i], d["patch_id"][i], d["row_idx"][i], d["col_idx"][i],
            d["valid_width"][i], d["valid_height"][i], d["original_width"][i],
            d["original_height"][i], y,
            "segtrain_positive" if y else "segtrain_mask_negative"
        ))
    return x, rows

def load_tumor_test(path, manifest_path):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    manifest = read_csv(manifest_path)
    idx = {(r["source_image"], r["patch_id"]): r for r in manifest if r["split"] == "tumor_test"}
    rows = []
    for i in range(len(x)):
        s, p = str(d["source_image"][i]), str(d["patch_id"][i])
        m = idx[(s, p)]; y = int(m["patch_label"])
        rows.append(rec(
            s, p, d["row_idx"][i], d["col_idx"][i], d["valid_width"][i], d["valid_height"][i],
            m["original_width"], m["original_height"], y,
            "tumor_test_positive" if y else "tumor_test_mask_negative"
        ))
    return x, rows

def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)

def group_indices(records):
    out = {}
    for name, fn in LEVELS:
        g = defaultdict(list)
        for i, r in enumerate(records): g[fn(r)].append(i)
        out[name] = {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}
    return out

def choose_group(r, models):
    for name, fn in LEVELS:
        key = fn(r)
        if key in models[name]: return name, key, models[name][key]
    raise RuntimeError("No matching group including global fallback.")

def split_images(records, frac, seed):
    meta = {}
    for r in records:
        meta.setdefault(r["source_image"], (r["original_width"], r["original_height"]))
    imgs = np.array(sorted(meta.keys()), dtype=object)
    labels_raw = [f"{meta[i][0]}x{meta[i][1]}" for i in imgs]
    cnt = Counter(labels_raw)
    labels = np.array([x if cnt[x] >= 10 else "other" for x in labels_raw], dtype=object)
    a, b = train_test_split(imgs, test_size=frac, random_state=seed, stratify=labels)
    return set(a.tolist()), set(b.tolist())

def subset(x, rows, image_set):
    ids = np.array([i for i, r in enumerate(rows) if r["source_image"] in image_set], dtype=np.int64)
    return x[ids], [rows[i] for i in ids]

def train_distributions(xn, records, grouped, k, min_ref=20, min_thr=20):
    dist = {n: {} for n, _ in LEVELS}
    for name, _ in LEVELS:
        for key, ids in grouped[name].items():
            if len(ids) < min_ref: continue
            bank = xn[ids]
            names = np.asarray([records[i]["source_image"] for i in ids], dtype=object)
            vals = []
            for start in range(0, len(ids), 256):
                q = bank[start:start+256]
                sims = q @ bank.T
                qn = names[start:start+len(q)]
                sims[qn[:, None] == names[None, :]] = -np.inf
                for row in sims:
                    v = row[np.isfinite(row)]
                    if len(v) == 0: continue
                    kk = min(k, len(v))
                    vals.append(float(np.mean(np.partition(v, -kk)[-kk:])))
            if len(vals) >= min_thr:
                dist[name][key] = {"idx": ids, "train_scores": np.asarray(vals, dtype=np.float32)}
    if ("global",) not in dist["global"]: raise RuntimeError("Global group unavailable.")
    return dist

def precompute_scores(qx, qrows, train_norm, dist, k):
    qn = l2(qx.astype(np.float32)); cache = {}; out = []
    for q, r in zip(qn, qrows):
        level, key, d = choose_group(r, dist)
        ck = (level, key)
        if ck not in cache: cache[ck] = train_norm[d["idx"]]
        sims = cache[ck] @ q
        kk = min(k, len(sims))
        score = float(np.mean(np.partition(sims, -kk)[-kk:]))
        out.append({**r, "match_level": level, "match_key": key, "similarity": score})
    return out

def threshold_models(dist, acceptance):
    q = (1.0 - acceptance) * 100.0
    out = {n: {} for n, _ in LEVELS}
    for name, _ in LEVELS:
        for key, d in dist[name].items():
            out[name][key] = float(np.percentile(d["train_scores"], q))
    return out

def classify(pre_rows, models):
    rows = []
    for r in pre_rows:
        th = models[r["match_level"]][r["match_key"]]
        rows.append({**r, "threshold": th, "pred": int(r["similarity"] < th)})
    return rows

def patch_metrics(rows):
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    p = np.asarray([r["pred"] for r in rows], dtype=np.int64)
    s = np.asarray([-r["similarity"] for r in rows], dtype=np.float64)
    tp = int(((y==1)&(p==1)).sum()); tn = int(((y==0)&(p==0)).sum())
    fp = int(((y==0)&(p==1)).sum()); fn = int(((y==1)&(p==0)).sum())
    prec = tp/max(tp+fp,1); sens = tp/max(tp+fn,1); spec = tn/max(tn+fp,1)
    f1 = 2*prec*sens/max(prec+sens,1e-12)
    return {
        "tp":tp,"tn":tn,"fp":fp,"fn":fn,
        "precision":prec,"sensitivity":sens,"specificity":spec,
        "f1":f1,"balanced_accuracy":0.5*(sens+spec),
        "auc":float(roc_auc_score(y,s)) if len(np.unique(y))>1 else None
    }

def group_specificity(rows, group):
    rr = [r for r in rows if r["group"] == group]
    return float(np.mean([r["pred"] == 0 for r in rr])) if rr else None

def image_metrics(rows, healthy_group):
    by = defaultdict(list)
    for r in rows: by[r["source_image"]].append(r)
    y=[]; p=[]; localized=0; onlyneg=0
    for rr in by.values():
        true = 0 if rr[0]["group"] == healthy_group else 1
        pred = int(any(r["pred"] for r in rr))
        hit = int(any(r["label"]==1 and r["pred"]==1 for r in rr))
        if true and pred:
            if hit: localized += 1
            else: onlyneg += 1
        y.append(true); p.append(pred)
    y=np.asarray(y); p=np.asarray(p)
    tp=int(((y==1)&(p==1)).sum()); tn=int(((y==0)&(p==0)).sum())
    fp=int(((y==0)&(p==1)).sum()); fn=int(((y==1)&(p==0)).sum())
    prec=tp/max(tp+fp,1); sens=tp/max(tp+fn,1); spec=tn/max(tn+fp,1)
    f1=2*prec*sens/max(prec+sens,1e-12)
    return {
        "n_images":len(y),"tp":tp,"tn":tn,"fp":fp,"fn":fn,
        "precision":prec,"sensitivity":sens,"specificity":spec,
        "f1":f1,"balanced_accuracy":0.5*(sens+spec),
        "localized_tumor_images":localized,
        "detected_only_from_mask_negative":onlyneg
    }

def fit_pca(train_x, others, dim):
    dim = min(dim, train_x.shape[1], len(train_x)-1)
    pca = PCA(n_components=dim, whiten=True, random_state=42)
    tr = pca.fit_transform(train_x).astype(np.float32)
    outs = [pca.transform(x).astype(np.float32) for x in others]
    return pca, tr, outs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--healthy_train_embeddings", required=True)
    ap.add_argument("--segtrain_embeddings", required=True)
    ap.add_argument("--healthy_test_embeddings", required=True)
    ap.add_argument("--tumor_test_embeddings", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--healthy_val_fraction", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", nargs="+", type=int, default=[72,80,88,96,104,112,120])
    ap.add_argument("--ks", nargs="+", type=int, default=[3,5,7,10,15])
    ap.add_argument("--acceptances", nargs="+", type=float, default=[0.875,0.90,0.9125,0.925,0.9375,0.95])
    args=ap.parse_args()

    out=Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    hall, hr = load_healthy(args.healthy_train_embeddings, "healthy_train")
    seg, sr = load_seg(args.segtrain_embeddings)
    htest, htr = load_healthy(args.healthy_test_embeddings, "healthy_test")
    ttest, ttr = load_tumor_test(args.tumor_test_embeddings, args.manifest)

    ref_imgs, val_imgs = split_images(hr, args.healthy_val_fraction, args.seed)
    href, href_r = subset(hall, hr, ref_imgs)
    hval, hval_r = subset(hall, hr, val_imgs)

    vx = np.concatenate([hval, seg], axis=0); vr = hval_r + sr

    print("="*100)
    print("BRISC HEALTHY-HOLDOUT FINE TUNING")
    print("="*100)
    print(f"healthy ref images={len(ref_imgs)} | healthy val images={len(val_imgs)}")
    print(f"healthy ref patches={len(href)} | healthy val patches={len(hval)} | segtrain patches={len(seg)}")
    print("Independent test is untouched until selection.\n")

    (out/"healthy_image_split.json").write_text(json.dumps({
        "seed":args.seed,"reference_images":sorted(ref_imgs),"validation_images":sorted(val_imgs)
    }, indent=2), encoding="utf-8")

    search=[]
    for dim in args.dims:
        print(f"PCA dim={dim}, whiten=True")
        _, rz, [vz] = fit_pca(href, [vx], dim)
        rn = l2(rz); grouped = group_indices(href_r)
        for k in args.ks:
            print(f"  k={k}")
            dist = train_distributions(rn, href_r, grouped, k)
            pre = precompute_scores(vz, vr, rn, dist, k)
            for acc in args.acceptances:
                rows = classify(pre, threshold_models(dist, acc))
                pm = patch_metrics(rows)
                im = image_metrics(rows, "healthy_train")
                search.append({
                    "dim":dim,"whiten":True,"k":k,"acceptance":acc,
                    "patch_precision":pm["precision"],"patch_sensitivity":pm["sensitivity"],
                    "patch_specificity":pm["specificity"],"patch_f1":pm["f1"],
                    "patch_balanced_accuracy":pm["balanced_accuracy"],"patch_auc":pm["auc"],
                    "healthy_val_patch_specificity":group_specificity(rows,"healthy_train"),
                    "segtrain_mask_negative_specificity":group_specificity(rows,"segtrain_mask_negative"),
                    "image_precision":im["precision"],"image_sensitivity":im["sensitivity"],
                    "image_specificity":im["specificity"],"image_f1":im["f1"],
                    "image_balanced_accuracy":im["balanced_accuracy"],
                    "localized_tumor_images":im["localized_tumor_images"],
                    "detected_only_from_mask_negative":im["detected_only_from_mask_negative"],
                })

    df=pd.DataFrame(search)
    df.to_csv(out/"fine_search_results.csv", index=False)

    bp = df[df["image_specificity"]>=0.85].copy()
    if len(bp)==0: bp=df.copy()
    balanced = bp.sort_values(["image_f1","image_balanced_accuracy","patch_f1"], ascending=False).iloc[0].to_dict()

    hp = df[df["image_specificity"]>=0.80].copy()
    if len(hp)==0: hp=df.copy()
    high = hp.sort_values(["image_sensitivity","image_f1","patch_sensitivity"], ascending=False).iloc[0].to_dict()

    selected = {
        "balanced_healthy_image_specificity_ge_0.85": balanced,
        "high_sensitivity_healthy_image_specificity_ge_0.80": high,
    }

    print("\nSELECTED ON VALIDATION ONLY")
    print(json.dumps(selected, indent=2, default=float))

    testx=np.concatenate([htest,ttest],axis=0); testr=htr+ttr
    final={}
    cache={}

    for tag,cfg in selected.items():
        dim=int(cfg["dim"]); k=int(cfg["k"]); acc=float(cfg["acceptance"])
        ck=(dim,k,acc)
        if ck not in cache:
            _, allz, [tz] = fit_pca(hall, [testx], dim)
            alln=l2(allz); grouped=group_indices(hr)
            dist=train_distributions(alln, hr, grouped, k)
            pre=precompute_scores(tz, testr, alln, dist, k)
            rows=classify(pre, threshold_models(dist, acc))
            cache[ck]=rows
        else:
            rows=cache[ck]

        pm=patch_metrics(rows); im=image_metrics(rows,"healthy_test")
        final[tag]={
            "config":{"dim":dim,"whiten":True,"k":k,"acceptance":acc},
            "test_patch_metrics":pm,
            "test_image_metrics":im,
        }
        write_csv(out/f"{tag}_test_patch_scores.csv", rows)

    summary={
        "protocol":"Image-level 80/20 healthy holdout + segmentation-train labels for selection; final PCA/reference refit on all healthy train; independent test evaluated only after selection.",
        "selected":selected,
        "final_test":final
    }
    (out/"fine_tuning_summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    print("\n"+"="*100)
    print("FINAL INDEPENDENT TEST")
    print("="*100)
    print(json.dumps(final, indent=2, default=float))
    print("\nSaved:")
    print(out/"fine_search_results.csv")
    print(out/"healthy_image_split.json")
    print(out/"fine_tuning_summary.json")

if __name__ == "__main__":
    main()
