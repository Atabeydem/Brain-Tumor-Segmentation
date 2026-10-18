#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Frozen-encoder supervised linear-probe benchmark for BRISC fixed-256 embeddings.

This tests whether the current autoencoder latent already contains tumor information
that cosine anomaly scoring fails to read.

Protocol
--------
- Encoder is never trained here.
- Healthy train uses the saved image-level reference/validation split.
- Segmentation-train is split 80/20 by source image.
- Hyperparameters and thresholds are selected on validation only.
- Dataset test split is evaluated only after model/threshold selection.

Models
------
- Logistic Regression
- Linear SVM

Features
--------
- raw 2048-D latent + StandardScaler
- PCA-whitened latent with configurable dimensions (default: 96,112,128)
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


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


def load_healthy(path, group):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = [
        {
            "source_image": str(d["source_image"][i]),
            "patch_id": str(d["patch_id"][i]),
            "label": 0,
            "group": group,
        }
        for i in range(len(x))
    ]
    return x, rows


def load_seg(path):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)
    rows = []
    for i in range(len(x)):
        y = int(d["patch_label"][i])
        rows.append(
            {
                "source_image": str(d["source_image"][i]),
                "patch_id": str(d["patch_id"][i]),
                "label": y,
                "group": "seg_positive" if y else "seg_mask_negative",
            }
        )
    return x, rows


def load_tumor_test(path, manifest_path):
    d = np.load(path, allow_pickle=True)
    x = d["embeddings"].astype(np.float32)

    manifest = read_csv(manifest_path)
    idx = {
        (r["source_image"], r["patch_id"]): r
        for r in manifest
        if r["split"] == "tumor_test"
    }

    rows = []
    for i in range(len(x)):
        s = str(d["source_image"][i])
        p = str(d["patch_id"][i])
        m = idx[(s, p)]
        y = int(m["patch_label"])
        rows.append(
            {
                "source_image": s,
                "patch_id": p,
                "label": y,
                "group": "tumor_test_positive" if y else "tumor_test_mask_negative",
            }
        )
    return x, rows


def healthy_split_indices(rows, split_json):
    data = json.loads(Path(split_json).read_text(encoding="utf-8"))
    ref = set(data["reference_images"])
    val = set(data["validation_images"])

    ref_idx = np.asarray(
        [i for i, r in enumerate(rows) if r["source_image"] in ref],
        dtype=np.int64,
    )
    val_idx = np.asarray(
        [i for i, r in enumerate(rows) if r["source_image"] in val],
        dtype=np.int64,
    )

    if len(ref_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError("Healthy split JSON paths did not match healthy embedding paths.")
    return ref_idx, val_idx


def seg_image_split(rows, val_fraction=0.20, seed=42):
    by_image = defaultdict(list)
    for i, r in enumerate(rows):
        by_image[r["source_image"]].append(i)

    images = np.asarray(sorted(by_image.keys()), dtype=object)
    strata = []
    for img in images:
        n_pos = sum(rows[i]["label"] == 1 for i in by_image[img])
        strata.append(str(min(int(n_pos), 4)))

    counts = Counter(strata)
    strata = np.asarray([s if counts[s] >= 10 else "other" for s in strata], dtype=object)

    train_images, val_images = train_test_split(
        images,
        test_size=val_fraction,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )

    train_set, val_set = set(train_images.tolist()), set(val_images.tolist())
    tr_idx = np.asarray(
        [i for i, r in enumerate(rows) if r["source_image"] in train_set],
        dtype=np.int64,
    )
    va_idx = np.asarray(
        [i for i, r in enumerate(rows) if r["source_image"] in val_set],
        dtype=np.int64,
    )
    return tr_idx, va_idx, train_set, val_set


def patch_metrics(y, pred, score):
    y = np.asarray(y, dtype=np.int64)
    pred = np.asarray(pred, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)

    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())

    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
    }


def healthy_image_specificity(rows, pred, healthy_group):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        if r["group"] == healthy_group:
            by[r["source_image"]].append(i)

    if not by:
        return np.nan

    correct = 0
    for ids in by.values():
        correct += int(not any(pred[i] == 1 for i in ids))
    return correct / len(by)


def image_metrics(rows, pred, healthy_group):
    by = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["source_image"]].append(i)

    y_img, p_img = [], []
    localized, only_negative = 0, 0

    for ids in by.values():
        rr = [rows[i] for i in ids]
        true = 0 if rr[0]["group"] == healthy_group else 1
        ipred = int(any(pred[i] == 1 for i in ids))
        true_patch_hit = int(any(rows[i]["label"] == 1 and pred[i] == 1 for i in ids))

        if true == 1 and ipred == 1:
            if true_patch_hit:
                localized += 1
            else:
                only_negative += 1

        y_img.append(true)
        p_img.append(ipred)

    y_img = np.asarray(y_img)
    p_img = np.asarray(p_img)
    tp = int(((y_img == 1) & (p_img == 1)).sum())
    tn = int(((y_img == 0) & (p_img == 0)).sum())
    fp = int(((y_img == 0) & (p_img == 1)).sum())
    fn = int(((y_img == 1) & (p_img == 0)).sum())
    precision = tp / max(tp + fp, 1)
    sensitivity = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    f1 = 2 * precision * sensitivity / max(precision + sensitivity, 1e-12)

    return {
        "n_images": int(len(y_img)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": 0.5 * (sensitivity + specificity),
        "localized_tumor_images": int(localized),
        "detected_only_from_mask_negative": int(only_negative),
    }


def threshold_candidates(scores, n=401):
    return np.unique(np.quantile(np.asarray(scores), np.linspace(0.01, 0.99, n)))


def tune_thresholds(rows, y, scores):
    y = np.asarray(y, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    best_bal = None
    best_high = None

    for th in threshold_candidates(scores):
        pred = (scores >= th).astype(np.int64)
        pm = patch_metrics(y, pred, scores)
        hspec = healthy_image_specificity(rows, pred, "healthy_val")

        if hspec >= 0.85:
            key = (pm["f1"], pm["balanced_accuracy"], pm["auc"])
            if best_bal is None or key > best_bal[0]:
                best_bal = (key, float(th), pm, float(hspec))

        if hspec >= 0.80 and pm["specificity"] >= 0.80:
            key = (pm["sensitivity"], pm["f1"], pm["balanced_accuracy"])
            if best_high is None or key > best_high[0]:
                best_high = (key, float(th), pm, float(hspec))

    # Fallback if constraints are impossible.
    if best_bal is None:
        for th in threshold_candidates(scores):
            pred = (scores >= th).astype(np.int64)
            pm = patch_metrics(y, pred, scores)
            hspec = healthy_image_specificity(rows, pred, "healthy_val")
            key = (pm["f1"], pm["balanced_accuracy"], pm["auc"])
            if best_bal is None or key > best_bal[0]:
                best_bal = (key, float(th), pm, float(hspec))

    if best_high is None:
        best_high = best_bal

    return best_bal, best_high


def transform_features(kind, dim, train_x, val_x, test_x):
    if kind == "raw_zscore":
        scaler = StandardScaler()
        tr = scaler.fit_transform(train_x).astype(np.float32)
        va = scaler.transform(val_x).astype(np.float32)
        te = scaler.transform(test_x).astype(np.float32)
        return scaler, tr, va, te

    if kind == "pca_whiten":
        dim = min(int(dim), train_x.shape[1], len(train_x) - 1)
        pca = PCA(
            n_components=dim,
            whiten=True,
            svd_solver="randomized",
            random_state=42,
        )
        tr = pca.fit_transform(train_x).astype(np.float32)
        va = pca.transform(val_x).astype(np.float32)
        te = pca.transform(test_x).astype(np.float32)
        return pca, tr, va, te

    raise ValueError(kind)


def make_model(name, c):
    if name == "logreg":
        return LogisticRegression(
            C=float(c),
            class_weight="balanced",
            solver="liblinear",
            max_iter=5000,
            random_state=42,
        )
    if name == "linear_svm":
        return LinearSVC(
            C=float(c),
            class_weight="balanced",
            dual="auto",
            max_iter=20000,
            random_state=42,
        )
    raise ValueError(name)


def decision_scores(model, x):
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(x), dtype=np.float64)
    return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--healthy_train_embeddings", required=True)
    ap.add_argument("--healthy_split_json", required=True)
    ap.add_argument("--segtrain_embeddings", required=True)
    ap.add_argument("--healthy_test_embeddings", required=True)
    ap.add_argument("--tumor_test_embeddings", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seg_val_fraction", type=float, default=0.20)
    ap.add_argument("--pca_dims", nargs="+", type=int, default=[96, 112, 128])
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    hx, hr = load_healthy(args.healthy_train_embeddings, "healthy_train_all")
    sx, sr = load_seg(args.segtrain_embeddings)
    htx, htr = load_healthy(args.healthy_test_embeddings, "healthy_test")
    ttx, ttr = load_tumor_test(args.tumor_test_embeddings, args.manifest)

    h_ref_idx, h_val_idx = healthy_split_indices(hr, args.healthy_split_json)
    s_tr_idx, s_val_idx, s_tr_images, s_val_images = seg_image_split(
        sr,
        val_fraction=args.seg_val_fraction,
        seed=args.seed,
    )

    h_ref_rows = [{**hr[i], "group": "healthy_ref"} for i in h_ref_idx]
    h_val_rows = [{**hr[i], "group": "healthy_val"} for i in h_val_idx]
    s_tr_rows = [sr[i] for i in s_tr_idx]
    s_val_rows = [sr[i] for i in s_val_idx]

    train_x = np.concatenate([hx[h_ref_idx], sx[s_tr_idx]], axis=0)
    train_r = h_ref_rows + s_tr_rows
    val_x = np.concatenate([hx[h_val_idx], sx[s_val_idx]], axis=0)
    val_r = h_val_rows + s_val_rows
    test_x = np.concatenate([htx, ttx], axis=0)
    test_r = htr + ttr

    y_train = np.asarray([r["label"] for r in train_r], dtype=np.int64)
    y_val = np.asarray([r["label"] for r in val_r], dtype=np.int64)
    y_test = np.asarray([r["label"] for r in test_r], dtype=np.int64)

    print("=" * 104)
    print("BRISC FROZEN-ENCODER LINEAR PROBE")
    print("=" * 104)
    print(f"Train patches      : {len(train_x)} | pos={int(y_train.sum())} neg={int((y_train==0).sum())}")
    print(f"Validation patches : {len(val_x)} | pos={int(y_val.sum())} neg={int((y_val==0).sum())}")
    print(f"Test patches       : {len(test_x)} | pos={int(y_test.sum())} neg={int((y_test==0).sum())}")
    print(f"Seg train images   : {len(s_tr_images)}")
    print(f"Seg val images     : {len(s_val_images)}")
    print("Encoder training   : NONE")
    print()

    feature_variants = [("raw_zscore", None)] + [("pca_whiten", d) for d in args.pca_dims]
    model_grid = [
        ("logreg", [0.01, 0.1, 1.0, 10.0]),
        ("linear_svm", [0.001, 0.01, 0.1, 1.0]),
    ]

    search_rows = []
    candidates = []

    for feature_kind, dim in feature_variants:
        feature_name = feature_kind if dim is None else f"{feature_kind}_{dim}"
        print(f"[features] {feature_name}")
        transformer, tr_z, va_z, te_z = transform_features(
            feature_kind,
            dim,
            train_x,
            val_x,
            test_x,
        )

        for model_name, c_values in model_grid:
            for c in c_values:
                print(f"  {model_name} C={c}")
                model = make_model(model_name, c)
                model.fit(tr_z, y_train)
                va_score = decision_scores(model, va_z)
                balanced, high = tune_thresholds(val_r, y_val, va_score)

                for mode, selected in [("balanced", balanced), ("high_sensitivity", high)]:
                    _, threshold, pm, healthy_img_spec = selected
                    row = {
                        "feature_kind": feature_kind,
                        "pca_dim": dim,
                        "model": model_name,
                        "C": c,
                        "mode": mode,
                        "threshold": threshold,
                        "val_patch_precision": pm["precision"],
                        "val_patch_sensitivity": pm["sensitivity"],
                        "val_patch_specificity": pm["specificity"],
                        "val_patch_f1": pm["f1"],
                        "val_patch_balanced_accuracy": pm["balanced_accuracy"],
                        "val_patch_auc": pm["auc"],
                        "val_healthy_image_specificity": healthy_img_spec,
                    }
                    search_rows.append(row)
                    candidates.append(
                        {
                            **row,
                            "model_object": model,
                            "test_features": te_z,
                        }
                    )

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(out / "linear_probe_search.csv", index=False)

    bal_df = search_df.loc[search_df["mode"].eq("balanced")].copy()
    bal_sel = bal_df.sort_values(
        ["val_patch_f1", "val_patch_balanced_accuracy", "val_patch_auc"],
        ascending=False,
    ).iloc[0]

    high_df = search_df.loc[search_df["mode"].eq("high_sensitivity")].copy()
    high_sel = high_df.sort_values(
        ["val_patch_sensitivity", "val_patch_f1", "val_patch_balanced_accuracy"],
        ascending=False,
    ).iloc[0]

    def find_candidate(row):
        for c in candidates:
            same_dim = (
                (pd.isna(row["pca_dim"]) and c["pca_dim"] is None)
                or (
                    not pd.isna(row["pca_dim"])
                    and c["pca_dim"] is not None
                    and int(row["pca_dim"]) == int(c["pca_dim"])
                )
            )
            if (
                c["feature_kind"] == row["feature_kind"]
                and same_dim
                and c["model"] == row["model"]
                and abs(float(c["C"]) - float(row["C"])) < 1e-12
                and c["mode"] == row["mode"]
            ):
                return c
        raise RuntimeError("Selected candidate not found.")

    selected = {
        "balanced": find_candidate(bal_sel),
        "high_sensitivity": find_candidate(high_sel),
    }

    final = {}
    for tag, c in selected.items():
        score = decision_scores(c["model_object"], c["test_features"])
        pred = (score >= c["threshold"]).astype(np.int64)
        pm = patch_metrics(y_test, pred, score)
        im = image_metrics(test_r, pred, "healthy_test")

        test_rows = []
        for r, s, p in zip(test_r, score, pred):
            test_rows.append(
                {
                    **r,
                    "decision_score": float(s),
                    "threshold": float(c["threshold"]),
                    "pred": int(p),
                }
            )
        write_csv(out / f"{tag}_linear_probe_test_patch_scores.csv", test_rows)

        final[tag] = {
            "config": {
                "feature_kind": c["feature_kind"],
                "pca_dim": c["pca_dim"],
                "model": c["model"],
                "C": c["C"],
                "threshold": c["threshold"],
            },
            "validation_patch_metrics": {
                "precision": c["val_patch_precision"],
                "sensitivity": c["val_patch_sensitivity"],
                "specificity": c["val_patch_specificity"],
                "f1": c["val_patch_f1"],
                "balanced_accuracy": c["val_patch_balanced_accuracy"],
                "auc": c["val_patch_auc"],
            },
            "validation_healthy_image_specificity": c["val_healthy_image_specificity"],
            "test_patch_metrics": pm,
            "test_image_metrics": im,
        }

    summary = {
        "protocol": (
            "Frozen encoder. Healthy train uses saved image-level 80/20 split. "
            "Segmentation-train is split image-level 80/20. Hyperparameters and "
            "thresholds are selected on validation only. Dataset test is evaluated "
            "only after selection."
        ),
        "train_patch_counts": {
            "n": int(len(y_train)),
            "positive": int(y_train.sum()),
            "negative": int((y_train == 0).sum()),
        },
        "validation_patch_counts": {
            "n": int(len(y_val)),
            "positive": int(y_val.sum()),
            "negative": int((y_val == 0).sum()),
        },
        "final_test": final,
    }

    (out / "linear_probe_summary.json").write_text(
        json.dumps(summary, indent=2, default=float),
        encoding="utf-8",
    )

    print()
    print("=" * 104)
    print("SELECTED + FINAL TEST")
    print("=" * 104)
    print(json.dumps(final, indent=2, default=float))
    print()
    print("Saved:")
    print(out / "linear_probe_search.csv")
    print(out / "linear_probe_summary.json")
    print(out / "balanced_linear_probe_test_patch_scores.csv")
    print(out / "high_sensitivity_linear_probe_test_patch_scores.csv")


if __name__ == "__main__":
    main()
