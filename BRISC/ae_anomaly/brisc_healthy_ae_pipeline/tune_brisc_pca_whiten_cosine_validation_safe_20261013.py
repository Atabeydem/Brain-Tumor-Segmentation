#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Validation-safe PCA/whitening/top-k cosine tuning for BRISC.

Key idea
--------
Hyperparameters are selected on SEGMENTATION TRAIN tumor patches, not on the
860-image tumor test set. The final test set is evaluated only after selection.

Reference / thresholds:
- HEALTHY TRAIN embeddings only.
- Matching hierarchy:
  exact -> resolution+position -> position+geometry -> position -> global
- Thresholds are healthy-train image-disjoint top-k cosine percentiles.

Validation:
- segmentation-train tumor-positive vs tumor-source mask-negative patches.
- Stage 1: search PCA dim / whitening / k at acceptance=0.95.
- Stage 2: tune healthy acceptance on top Stage-1 configs.

Final:
- evaluate selected configs on independent healthy test + tumor test.

No retraining, no resize, no patch averaging.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score


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
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def rec(source_image, patch_id, row_idx, col_idx, valid_width, valid_height,
        original_width, original_height, label=0, group="healthy"):
    return {
        "source_image": str(source_image),
        "patch_id": str(patch_id),
        "row_idx": int(row_idx), "col_idx": int(col_idx),
        "valid_width": int(valid_width), "valid_height": int(valid_height),
        "original_width": int(original_width), "original_height": int(original_height),
        "label": int(label), "group": group,
    }


def load_healthy(path, group):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = [
        rec(d["source_image"][i], d["patch_id"][i], d["row_idx"][i], d["col_idx"][i],
            d["valid_width"][i], d["valid_height"][i],
            d["original_width"][i], d["original_height"][i], 0, group)
        for i in range(len(x))
    ]
    return x, rows


def load_labeled_tumor_npz(path, group_prefix):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = []
    for i in range(len(x)):
        y = int(d["patch_label"][i])
        rows.append(
            rec(d["source_image"][i], d["patch_id"][i], d["row_idx"][i], d["col_idx"][i],
                d["valid_width"][i], d["valid_height"][i],
                d["original_width"][i], d["original_height"][i], y,
                f"{group_prefix}_positive" if y else f"{group_prefix}_mask_negative")
        )
    return x, rows


def load_tumor_test(path, manifest_path):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    manifest = read_csv(manifest_path)
    idx = {(r["source_image"], r["patch_id"]): r for r in manifest if r["split"] == "tumor_test"}
    rows = []
    for i in range(len(x)):
        s, p = str(d["source_image"][i]), str(d["patch_id"][i])
        m = idx[(s, p)]
        y = int(m["patch_label"])
        rows.append(
            rec(s, p, d["row_idx"][i], d["col_idx"][i],
                d["valid_width"][i], d["valid_height"][i],
                m["original_width"], m["original_height"], y,
                "tumor_test_positive" if y else "tumor_test_mask_negative")
        )
    return x, rows


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def group_indices(records):
    out = {}
    for name, fn in LEVELS:
        g = defaultdict(list)
        for i, r in enumerate(records):
            g[fn(r)].append(i)
        out[name] = {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}
    return out


def choose_group(r, models):
    for name, fn in LEVELS:
        key = fn(r)
        if key in models[name]:
            return name, key, models[name][key]
    raise RuntimeError("No group, including global fallback.")


def train_topk_distributions(xn, records, grouped, k, min_ref=20):
    """
    For each group, compute healthy-train image-disjoint top-k score distribution.
    Threshold percentile can later be changed WITHOUT recomputing similarities.
    """
    dist = {name: {} for name, _ in LEVELS}

    for name, _ in LEVELS:
        for key, idx in grouped[name].items():
            if len(idx) < min_ref:
                continue

            bank = xn[idx]
            names = np.asarray([records[i]["source_image"] for i in idx], dtype=object)
            scores = []

            for start in range(0, len(idx), 256):
                q = bank[start:start+256]
                sims = q @ bank.T
                qnames = names[start:start+len(q)]
                sims[qnames[:, None] == names[None, :]] = -np.inf

                for row in sims:
                    v = row[np.isfinite(row)]
                    if len(v) == 0:
                        continue
                    kk = min(k, len(v))
                    scores.append(float(np.mean(np.partition(v, -kk)[-kk:])))

            if len(scores) >= 20:
                dist[name][key] = {
                    "idx": idx,
                    "train_scores": np.asarray(scores, dtype=np.float32),
                }

    if ("global",) not in dist["global"]:
        raise RuntimeError("Global fallback unavailable.")
    return dist


def threshold_models(distributions, acceptance):
    q = (1.0 - acceptance) * 100.0
    models = {name: {} for name, _ in LEVELS}
    for name, _ in LEVELS:
        for key, d in distributions[name].items():
            models[name][key] = {
                "idx": d["idx"],
                "threshold": float(np.percentile(d["train_scores"], q)),
            }
    return models


def query_topk_scores(qx, qrecords, train_norm, distributions, k):
    """
    Query similarity is independent of acceptance threshold, so compute once.
    """
    qn = l2(qx.astype(np.float32))
    cache = {}
    out = []

    for q, r in zip(qn, qrecords):
        level, key, d = choose_group(r, distributions)
        ck = (level, key)
        if ck not in cache:
            cache[ck] = train_norm[d["idx"]]
        sims = cache[ck] @ q
        kk = min(k, len(sims))
        score = float(np.mean(np.partition(sims, -kk)[-kk:]))
        out.append((level, key, score))
    return out


def classify_from_scores(qrecords, pre_scores, models):
    rows = []
    for r, (level, key, score) in zip(qrecords, pre_scores):
        m = models[level][key]
        pred = int(score < m["threshold"])
        rows.append({
            **r,
            "match_level": level,
            "similarity": score,
            "threshold": m["threshold"],
            "pred": pred,
        })
    return rows


def classification_metrics(rows):
    y = np.asarray([r["label"] for r in rows], dtype=np.int64)
    p = np.asarray([r["pred"] for r in rows], dtype=np.int64)
    s = np.asarray([-r["similarity"] for r in rows], dtype=np.float64)

    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())

    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)
    bal = 0.5 * (sensitivity + specificity)
    auc = float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else None

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": bal,
        "auc": auc,
    }


def image_metrics(rows, healthy_group_name):
    by_img = defaultdict(list)
    for r in rows:
        by_img[r["source_image"]].append(r)

    y, p = [], []
    localized = 0
    only_negative = 0

    for _, rr in by_img.items():
        true = 0 if rr[0]["group"] == healthy_group_name else 1
        pred = int(any(r["pred"] == 1 for r in rr))
        hit = int(any(r["label"] == 1 and r["pred"] == 1 for r in rr))

        if true and pred:
            if hit:
                localized += 1
            else:
                only_negative += 1

        y.append(true)
        p.append(pred)

    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)

    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "localized_tumor_images": localized,
        "detected_only_from_mask_negative": only_negative,
    }


def transform_pca(train_x, val_x, test_x, dim, whiten):
    dim = min(dim, len(train_x)-1, train_x.shape[1])
    pca = PCA(n_components=dim, whiten=whiten, random_state=42)
    a = pca.fit_transform(train_x)
    b = pca.transform(val_x)
    c = pca.transform(test_x)
    return a.astype(np.float32), b.astype(np.float32), c.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy_train_embeddings", required=True)
    ap.add_argument("--segtrain_embeddings", required=True)
    ap.add_argument("--healthy_test_embeddings", required=True)
    ap.add_argument("--tumor_test_embeddings", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--dims", nargs="+", type=int, default=[64,96,128,160,192,256])
    ap.add_argument("--ks", nargs="+", type=int, default=[3,5,10,20])
    ap.add_argument("--stage1_acceptance", type=float, default=0.95)
    ap.add_argument("--acceptances", nargs="+", type=float, default=[0.85,0.90,0.925,0.95,0.975])
    ap.add_argument("--top_stage1", type=int, default=3)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    htr, htr_r = load_healthy(args.healthy_train_embeddings, "healthy_train")
    val, val_r = load_labeled_tumor_npz(args.segtrain_embeddings, "segtrain")
    hte, hte_r = load_healthy(args.healthy_test_embeddings, "healthy_test")
    tte, tte_r = load_tumor_test(args.tumor_test_embeddings, args.manifest)

    test = np.concatenate([hte, tte], axis=0)
    test_r = hte_r + tte_r

    print("="*100)
    print("VALIDATION-SAFE PCA/WHITEN/TOP-K TUNING")
    print("="*100)
    print(f"healthy train refs : {len(htr)}")
    print(f"seg-train val      : {len(val)}")
    print(f"healthy test       : {len(hte)}")
    print(f"tumor test         : {len(tte)}")
    print("Tumor test is NOT used for hyperparameter selection.\n")

    stage1 = []
    cache = {}

    # Stage 1: dim / whiten / k @ acceptance=.95
    total = len(args.dims) * 2 * len(args.ks)
    c = 0

    for dim in args.dims:
        for whiten in [False, True]:
            key = (dim, whiten)
            print(f"\nPCA dim={dim}, whiten={whiten}")
            tr_z, val_z, test_z = transform_pca(htr, val, test, dim, whiten)
            cache[key] = (tr_z, val_z, test_z)

            for k in args.ks:
                c += 1
                print(f"  [{c}/{total}] k={k}")

                trn = l2(tr_z)
                gr = group_indices(htr_r)
                dist = train_topk_distributions(trn, htr_r, gr, k)
                val_pre = query_topk_scores(val_z, val_r, trn, dist, k)
                models = threshold_models(dist, args.stage1_acceptance)
                rows = classify_from_scores(val_r, val_pre, models)
                m = classification_metrics(rows)

                stage1.append({
                    "dim": dim, "whiten": whiten, "k": k,
                    "acceptance": args.stage1_acceptance,
                    **m
                })

    s1 = pd.DataFrame(stage1).sort_values(
        ["f1", "balanced_accuracy", "auc"],
        ascending=False
    ).reset_index(drop=True)
    s1.to_csv(out / "stage1_search.csv", index=False)

    top = s1.head(args.top_stage1)[["dim","whiten","k"]].to_dict("records")

    print("\nTop Stage-1 configs:")
    print(s1.head(args.top_stage1).to_string(index=False))

    # Stage 2: acceptance tuning on top configs.
    stage2 = []

    for cfg in top:
        dim, whiten, k = int(cfg["dim"]), bool(cfg["whiten"]), int(cfg["k"])
        tr_z, val_z, test_z = cache[(dim, whiten)]
        trn = l2(tr_z)
        gr = group_indices(htr_r)
        dist = train_topk_distributions(trn, htr_r, gr, k)
        val_pre = query_topk_scores(val_z, val_r, trn, dist, k)

        for acc in args.acceptances:
            models = threshold_models(dist, acc)
            rows = classify_from_scores(val_r, val_pre, models)
            m = classification_metrics(rows)
            stage2.append({
                "dim": dim, "whiten": whiten, "k": k,
                "acceptance": acc, **m
            })

    s2 = pd.DataFrame(stage2).sort_values(
        ["f1", "balanced_accuracy", "auc"],
        ascending=False
    ).reset_index(drop=True)
    s2.to_csv(out / "stage2_search.csv", index=False)

    # Selection 1: best validation F1.
    best_f1 = s2.iloc[0].to_dict()

    # Selection 2: highest sensitivity while keeping validation specificity >= 0.80.
    feasible = s2[s2["specificity"] >= 0.80]
    if len(feasible):
        best_sens = feasible.sort_values(
            ["sensitivity", "f1", "balanced_accuracy"],
            ascending=False
        ).iloc[0].to_dict()
    else:
        best_sens = best_f1

    selected = {
        "best_validation_f1": best_f1,
        "high_sensitivity_with_val_specificity_ge_0.80": best_sens,
    }

    print("\nSelected before test evaluation:")
    print(json.dumps(selected, indent=2, default=float))

    final = {}

    for tag, cfg in selected.items():
        dim = int(cfg["dim"])
        whiten = bool(cfg["whiten"])
        k = int(cfg["k"])
        acc = float(cfg["acceptance"])

        tr_z, val_z, test_z = cache[(dim, whiten)]
        trn = l2(tr_z)
        gr = group_indices(htr_r)
        dist = train_topk_distributions(trn, htr_r, gr, k)
        test_pre = query_topk_scores(test_z, test_r, trn, dist, k)
        models = threshold_models(dist, acc)
        test_rows = classify_from_scores(test_r, test_pre, models)

        patch_m = classification_metrics(test_rows)
        img_m = image_metrics(test_rows, "healthy_test")

        final[tag] = {
            "config": {
                "dim": dim, "whiten": whiten, "k": k, "acceptance": acc
            },
            "validation_metrics": {
                k2: float(v) if isinstance(v, (np.floating,)) else v
                for k2, v in cfg.items()
                if k2 in ["sensitivity","specificity","precision","f1","balanced_accuracy","auc"]
            },
            "test_patch_metrics": patch_m,
            "test_image_metrics": img_m,
        }

        write_csv(out / f"{tag}_test_patch_scores.csv", test_rows)

    summary = {
        "selection_protocol": (
            "Hyperparameters selected on segmentation-train tumor patch labels. "
            "Tumor test was untouched until final selected-config evaluation."
        ),
        "selected": selected,
        "final_test": final,
    }

    (out / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8"
    )

    print("\n" + "="*100)
    print("FINAL INDEPENDENT TEST")
    print("="*100)
    print(json.dumps(final, indent=2, default=float))
    print("\nSaved:")
    print(out / "stage1_search.csv")
    print(out / "stage2_search.csv")
    print(out / "tuning_summary.json")


if __name__ == "__main__":
    main()
