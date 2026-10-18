#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Inference-only benchmark for BRISC fixed-256 latent anomaly scoring.

Methods:
- raw cosine top-k: k=1,5,10,20
- z-score + cosine top-10
- PCA(128, whiten=True) + cosine top-10
- PCA(64) + LedoitWolf Mahalanobis

All preprocessing/thresholds use HEALTHY TRAIN only.
Matching hierarchy:
exact -> resolution+position -> position+geometry -> position -> global

No retraining, no resize, no patch averaging.
"""

from __future__ import annotations
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score


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
        original_width, original_height, true_label=0, source_group="healthy_test"):
    return {
        "source_image": str(source_image),
        "patch_id": str(patch_id),
        "row_idx": int(row_idx), "col_idx": int(col_idx),
        "valid_width": int(valid_width), "valid_height": int(valid_height),
        "original_width": int(original_width), "original_height": int(original_height),
        "true_label": int(true_label),
        "source_group": source_group,
    }


def load_healthy_npz(path, group_name):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = [
        rec(d["source_image"][i], d["patch_id"][i], d["row_idx"][i], d["col_idx"][i],
            d["valid_width"][i], d["valid_height"][i],
            d["original_width"][i], d["original_height"][i],
            0, group_name)
        for i in range(len(x))
    ]
    return x, rows


def load_tumor_npz(npz_path, manifest_path):
    d = np.load(npz_path, allow_pickle=True)
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
                "tumor_positive" if y else "tumor_source_mask_negative")
        )
    return x, rows


def l2(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, 1e-12, None)


def groups(records):
    out = {}
    for name, fn in LEVELS:
        g = defaultdict(list)
        for i, r in enumerate(records):
            g[fn(r)].append(i)
        out[name] = {k: np.asarray(v, dtype=np.int64) for k, v in g.items()}
    return out


def choose(r, models):
    for name, fn in LEVELS:
        key = fn(r)
        if key in models[name]:
            return name, key, models[name][key]
    raise RuntimeError("No matching group, including global.")


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    tp = int(((y == 1) & (p == 1)).sum())
    tn = int(((y == 0) & (p == 0)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec_ = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    f1 = 2 * prec * rec_ / max(prec + rec_, 1e-12)
    return {
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "accuracy": (tp + tn) / max(len(y), 1),
        "precision": prec, "sensitivity": rec_, "specificity": spec,
        "balanced_accuracy": 0.5 * (rec_ + spec), "f1": f1,
    }


def aucs(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return None, None
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


# ---------- cosine top-k ----------

def train_topk_scores(xn, records, indices, k):
    idx = np.asarray(indices)
    bank = xn[idx]
    names = np.asarray([records[i]["source_image"] for i in idx], dtype=object)
    out = []

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
            out.append(float(np.mean(np.partition(v, -kk)[-kk:])))
    return np.asarray(out)


def build_cos_models(train_x, train_records, k, accept, min_ref=20, min_thr=20):
    xn = l2(train_x.astype(np.float32))
    gr = groups(train_records)
    q = (1.0 - accept) * 100.0
    models = {name: {} for name, _ in LEVELS}

    for name, _ in LEVELS:
        for key, idx in gr[name].items():
            if len(idx) < min_ref:
                continue
            s = train_topk_scores(xn, train_records, idx, k)
            if len(s) < min_thr:
                continue
            models[name][key] = {
                "idx": idx,
                "threshold": float(np.percentile(s, q)),
                "n": int(len(idx)),
            }

    if ("global",) not in models["global"]:
        raise RuntimeError("Global cosine fallback missing.")
    return xn, models


def score_cos(query_x, query_records, train_norm, models, k):
    qn = l2(query_x.astype(np.float32))
    cache, rows = {}, []

    for q, r in zip(qn, query_records):
        level, key, m = choose(r, models)
        ck = (level, key)
        if ck not in cache:
            cache[ck] = train_norm[m["idx"]]
        sims = cache[ck] @ q
        kk = min(k, len(sims))
        score = float(np.mean(np.partition(sims, -kk)[-kk:]))
        pred = int(score < m["threshold"])
        rows.append({
            **r, "match_level": level, "raw_score": score,
            "anomaly_score": -score, "threshold": m["threshold"], "pred_label": pred
        })
    return rows


# ---------- Mahalanobis ----------

def build_maha_models(z, records, accept, min_ref=20, min_thr=20):
    gr = groups(records)
    models = {name: {} for name, _ in LEVELS}

    for name, _ in LEVELS:
        for key, idx in gr[name].items():
            if len(idx) < min_ref:
                continue
            zz = z[idx]
            lw = LedoitWolf().fit(zz)
            mu, precision = lw.location_, lw.precision_
            d = zz - mu
            dist = np.sqrt(np.clip(np.einsum("bi,ij,bj->b", d, precision, d), 0, None))
            if len(dist) < min_thr:
                continue
            models[name][key] = {
                "mu": mu, "precision": precision,
                "threshold": float(np.percentile(dist, accept * 100)),
                "n": int(len(idx)),
            }

    if ("global",) not in models["global"]:
        raise RuntimeError("Global Mahalanobis fallback missing.")
    return models


def score_maha(zq, records, models):
    rows = []
    for z, r in zip(zq, records):
        level, key, m = choose(r, models)
        d = z - m["mu"]
        score = math.sqrt(max(float(d @ m["precision"] @ d), 0.0))
        pred = int(score > m["threshold"])
        rows.append({
            **r, "match_level": level, "raw_score": score,
            "anomaly_score": score, "threshold": m["threshold"], "pred_label": pred
        })
    return rows


def summarize(method, rows, outdir):
    y = [r["true_label"] for r in rows]
    p = [r["pred_label"] for r in rows]
    s = [r["anomaly_score"] for r in rows]

    pm = metrics(y, p)
    pm["roc_auc"], pm["average_precision"] = aucs(y, s)

    sub = {}
    for g in ["healthy_test", "tumor_source_mask_negative", "tumor_positive"]:
        rr = [r for r in rows if r["source_group"] == g]
        sub[g] = {
            "n": len(rr),
            "mean_raw_score": float(np.mean([r["raw_score"] for r in rr])),
            "predicted_tumor_rate": float(np.mean([r["pred_label"] for r in rr])),
        }

    by_img = defaultdict(list)
    for r in rows:
        by_img[r["source_image"]].append(r)

    img_rows = []
    localized = only_neg = 0

    for image, rr in by_img.items():
        true = 0 if rr[0]["source_group"] == "healthy_test" else 1
        pred = int(any(r["pred_label"] for r in rr))
        hit = int(any(r["true_label"] == 1 and r["pred_label"] == 1 for r in rr))
        if true and pred:
            if hit:
                localized += 1
            else:
                only_neg += 1
        img_rows.append({
            "source_image": image, "true_label": true, "pred_label": pred,
            "anomaly_score": max(r["anomaly_score"] for r in rr),
            "true_tumor_patch_hit": hit,
        })

    im = metrics([r["true_label"] for r in img_rows], [r["pred_label"] for r in img_rows])
    im["roc_auc"], im["average_precision"] = aucs(
        [r["true_label"] for r in img_rows],
        [r["anomaly_score"] for r in img_rows],
    )

    loc = {
        "true_tumor_patch_localized_images": localized,
        "true_tumor_patch_localization_rate": localized / 860.0,
        "detected_only_from_mask_negative_patch": only_neg,
    }

    md = Path(outdir) / method
    md.mkdir(parents=True, exist_ok=True)
    write_csv(md / "patch_scores.csv", rows)
    write_csv(md / "image_scores.csv", img_rows)
    (md / "summary.json").write_text(
        json.dumps({"patch": pm, "subgroups": sub, "image": im, "localization": loc}, indent=2),
        encoding="utf-8"
    )
    return pm, im, loc, sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train_embeddings", required=True)
    ap.add_argument("--healthy_test_embeddings", required=True)
    ap.add_argument("--tumor_embeddings", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--healthy_acceptance", type=float, default=0.95)
    ap.add_argument("--topk", nargs="+", type=int, default=[1,5,10,20])
    ap.add_argument("--pca_whiten_dim", type=int, default=128)
    ap.add_argument("--mahalanobis_pca_dim", type=int, default=64)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tr, trr = load_healthy_npz(args.train_embeddings, "healthy_train")
    he, her = load_healthy_npz(args.healthy_test_embeddings, "healthy_test")
    tu, tur = load_tumor_npz(args.tumor_embeddings, args.manifest)
    qx = np.concatenate([he, tu], axis=0)
    qr = her + tur

    print("="*92)
    print("BRISC LATENT ANOMALY-SCORING BENCHMARK")
    print("="*92)
    print(f"train healthy={len(tr)} | healthy test={len(he)} | tumor-source={len(tu)} | dim={tr.shape[1]}")
    print("No retraining. Healthy-train-only transforms and thresholds.\n")

    results = []

    def run(method, rows):
        pm, im, loc, sub = summarize(method, rows, out)
        results.append({
            "method": method,
            "patch_sensitivity": pm["sensitivity"],
            "patch_specificity": pm["specificity"],
            "patch_precision": pm["precision"],
            "patch_f1": pm["f1"],
            "patch_auc": pm["roc_auc"],
            "image_sensitivity": im["sensitivity"],
            "image_specificity": im["specificity"],
            "image_precision": im["precision"],
            "image_f1": im["f1"],
            "image_auc": im["roc_auc"],
            "localization_rate": loc["true_tumor_patch_localization_rate"],
        })
        print(f"{method:30s} | patch sens {pm['sensitivity']*100:6.2f}% F1 {pm['f1']:.4f} AUC {pm['roc_auc']:.4f}"
              f" | image sens {im['sensitivity']*100:6.2f}% spec {im['specificity']*100:6.2f}% F1 {im['f1']:.4f}")

    # Raw top-k
    for k in args.topk:
        print(f"\n[raw cosine top-{k}]")
        tn, models = build_cos_models(tr, trr, k, args.healthy_acceptance)
        run(f"raw_cosine_top{k}", score_cos(qx, qr, tn, models, k))

    # Z-score + cosine
    print("\n[z-score cosine top-10]")
    mu = tr.mean(axis=0, keepdims=True)
    sd = tr.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    ztr, zq = (tr-mu)/sd, (qx-mu)/sd
    tn, models = build_cos_models(ztr, trr, 10, args.healthy_acceptance)
    run("zscore_cosine_top10", score_cos(zq, qr, tn, models, 10))

    # PCA whiten + cosine
    dim = min(args.pca_whiten_dim, len(tr)-1, tr.shape[1])
    print(f"\n[PCA({dim}) whiten + cosine top-10]")
    pw = PCA(n_components=dim, whiten=True, random_state=42)
    ptr, pq = pw.fit_transform(tr), pw.transform(qx)
    tn, models = build_cos_models(ptr, trr, 10, args.healthy_acceptance)
    run(f"pca{dim}_whiten_cosine_top10", score_cos(pq, qr, tn, models, 10))

    # PCA + Mahalanobis
    mdim = min(args.mahalanobis_pca_dim, len(tr)-1, tr.shape[1])
    print(f"\n[PCA({mdim}) + LedoitWolf Mahalanobis]")
    pm = PCA(n_components=mdim, whiten=False, random_state=42)
    mtr, mq = pm.fit_transform(tr), pm.transform(qx)
    mm = build_maha_models(mtr, trr, args.healthy_acceptance)
    run(f"pca{mdim}_ledoitwolf_mahalanobis", score_maha(mq, qr, mm))

    df = pd.DataFrame(results).sort_values(["patch_f1", "patch_auc"], ascending=False)
    df.to_csv(out / "benchmark_results.csv", index=False)

    # Plot
    labels = df["method"].tolist()
    x = np.arange(len(df))
    w = 0.25
    fig, ax = plt.subplots(figsize=(14,7))
    ax.bar(x-w, df["patch_sensitivity"]*100, w, label="Patch sensitivity")
    ax.bar(x, df["image_sensitivity"]*100, w, label="Image sensitivity")
    ax.bar(x+w, df["image_specificity"]*100, w, label="Image specificity")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0,105)
    ax.set_ylabel("Percent")
    ax.set_title("BRISC latent anomaly-score benchmark")
    ax.grid(True, axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "benchmark_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "retrained": False,
        "healthy_train_reference_patches": len(tr),
        "healthy_test_patches": len(he),
        "tumor_source_test_patches": len(tu),
        "embedding_dim": tr.shape[1],
        "results": results,
        "best_patch_f1": df.iloc[0]["method"],
        "best_patch_auc": df.loc[df["patch_auc"].idxmax(), "method"],
        "best_image_f1": df.loc[df["image_f1"].idxmax(), "method"],
    }
    (out / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "="*92)
    print("FINAL BENCHMARK")
    print("="*92)
    show = df.copy()
    for c in ["patch_sensitivity","patch_specificity","image_sensitivity","image_specificity","localization_rate"]:
        show[c] = show[c]*100
    print(show.to_string(index=False))
    print("\nSaved:")
    print(out / "benchmark_results.csv")
    print(out / "benchmark_summary.json")
    print(out / "benchmark_comparison.png")


if __name__ == "__main__":
    main()
