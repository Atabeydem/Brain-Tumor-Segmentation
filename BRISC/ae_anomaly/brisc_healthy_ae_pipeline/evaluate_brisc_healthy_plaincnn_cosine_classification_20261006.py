#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score,
        f1_score, confusion_matrix, roc_auc_score
    )
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
PATCH_NAMES = ["TL", "TR", "BL", "BR"]


def list_images(folder: Path):
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def split_boxes_2x2(width, height):
    xm, ym = width // 2, height // 2
    return {
        "TL": (0, 0, xm, ym),
        "TR": (xm, 0, width, ym),
        "BL": (0, ym, xm, height),
        "BR": (xm, ym, width, height),
    }


def round_up(v, multiple=8):
    return ((v + multiple - 1) // multiple) * multiple


def load_native_gray(path: Path):
    with Image.open(path) as im:
        im = im.convert("L")
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class NativePlainCNNAutoencoder(nn.Module):
    """
    Skip-OFF model used in the healthy native-resolution experiment.
    """
    def __init__(self, base_channels=32):
        super().__init__()
        b = base_channels

        self.enc1 = ConvBlock(1, b)
        self.down1 = nn.Conv2d(b, b * 2, 4, stride=2, padding=1)

        self.enc2 = ConvBlock(b * 2, b * 2)
        self.down2 = nn.Conv2d(b * 2, b * 4, 4, stride=2, padding=1)

        self.enc3 = ConvBlock(b * 4, b * 4)
        self.down3 = nn.Conv2d(b * 4, b * 8, 4, stride=2, padding=1)

        self.bottleneck = ConvBlock(b * 8, b * 8)

        self.up3 = nn.ConvTranspose2d(b * 8, b * 4, 4, stride=2, padding=1)
        self.dec3 = ConvBlock(b * 4, b * 4)

        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, 4, stride=2, padding=1)
        self.dec2 = ConvBlock(b * 2, b * 2)

        self.up1 = nn.ConvTranspose2d(b * 2, b, 4, stride=2, padding=1)
        self.dec1 = ConvBlock(b, b)

        self.out = nn.Conv2d(b, 1, 1)
        self.latent_pool = nn.AdaptiveAvgPool2d((2, 2))

    def encode_features(self, x):
        x = self.enc1(x)
        x = F.relu(self.down1(x), inplace=True)

        x = self.enc2(x)
        x = F.relu(self.down2(x), inplace=True)

        x = self.enc3(x)
        x = F.relu(self.down3(x), inplace=True)

        z = self.bottleneck(x)
        return z

    def latent_embedding(self, z):
        return self.latent_pool(z).flatten(1)

    def forward_embedding(self, x):
        z = self.encode_features(x)
        return self.latent_embedding(z)


def l2_normalize(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def cosine_similarity_matrix(a, b):
    a_n = l2_normalize(a.astype(np.float64))
    b_n = l2_normalize(b.astype(np.float64))
    return a_n @ b_n.T


def load_embedding_npz(path):
    data = np.load(path, allow_pickle=False)
    emb = data["embeddings"].astype(np.float32)
    names = data["image"].astype(str)
    return emb, names


@torch.no_grad()
def encode_image(model, image_path, device):
    """
    Native image -> fixed 2x2 patches -> 4 x 1024-D -> mean image embedding.
    No spatial resize.
    """
    arr = load_native_gray(image_path)
    h, w = arr.shape
    boxes = split_boxes_2x2(w, h)

    patch_embeddings = []

    for patch_name in PATCH_NAMES:
        x0, y0, x1, y1 = boxes[patch_name]
        patch = arr[y0:y1, x0:x1]
        ph, pw = patch.shape

        pad_h = round_up(ph, 8)
        pad_w = round_up(pw, 8)

        x = torch.from_numpy(patch).float().unsqueeze(0).unsqueeze(0)
        x = F.pad(
            x,
            (0, pad_w - pw, 0, pad_h - ph),
            mode="replicate"
        ).to(device)

        emb = model.forward_embedding(x)
        patch_embeddings.append(emb[0].cpu().numpy().astype(np.float32))

    return np.mean(np.stack(patch_embeddings, axis=0), axis=0)


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--healthy_train_embeddings", required=True)
    ap.add_argument("--healthy_test_embeddings", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument(
        "--healthy_acceptance",
        type=float,
        default=0.95,
        help="Fraction of healthy-train LOO similarities intended to remain above threshold."
    )
    ap.add_argument("--base_channels", type=int, default=32)

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    class_test_root = dataset_root / "classification_task" / "test"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # Load skip-OFF model
    # ------------------------------------------------------------
    ckpt = torch.load(args.checkpoint, map_location=device)

    base_channels = int(ckpt.get("base_channels", args.base_channels))
    model = NativePlainCNNAutoencoder(base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print("=" * 88)
    print("HEALTHY-REFERENCE COSINE SIMILARITY CLASSIFICATION")
    print("=" * 88)
    print("Training / reference class : healthy only")
    print("Autoencoder              : skip-OFF Plain CNN")
    print("Spatial resize           : NONE")
    print("Patch strategy           : native 2x2")
    print("Image embedding          : mean of four 1024-D patch embeddings")
    print("Similarity               : cosine similarity")
    print("Reference decision       : nearest healthy-train embedding")
    print(f"Healthy acceptance target: {args.healthy_acceptance:.2%}")
    print(f"Device                   : {device}")
    print()

    # ------------------------------------------------------------
    # Healthy reference bank
    # ------------------------------------------------------------
    train_emb, train_names = load_embedding_npz(args.healthy_train_embeddings)
    healthy_test_emb, healthy_test_names = load_embedding_npz(args.healthy_test_embeddings)

    print(f"Healthy train embeddings : {train_emb.shape}")
    print(f"Healthy test embeddings  : {healthy_test_emb.shape}")

    if train_emb.ndim != 2 or train_emb.shape[1] != 1024:
        raise ValueError(f"Expected healthy train embeddings Nx1024, got {train_emb.shape}")

    # LOO nearest-neighbor cosine similarity on healthy training bank.
    sim_train = cosine_similarity_matrix(train_emb, train_emb)
    np.fill_diagonal(sim_train, -np.inf)

    loo_best_idx = np.argmax(sim_train, axis=1)
    loo_best_sim = sim_train[np.arange(len(train_emb)), loo_best_idx]

    lower_percentile = 100.0 * (1.0 - args.healthy_acceptance)
    threshold = float(np.percentile(loo_best_sim, lower_percentile))

    loo_rows = []
    for i in range(len(train_emb)):
        loo_rows.append({
            "image": train_names[i],
            "nearest_healthy_image": train_names[loo_best_idx[i]],
            "nearest_cosine_similarity": float(loo_best_sim[i]),
        })

    write_csv(out / "healthy_train_leave_one_out_similarity.csv", loo_rows)

    print()
    print("=== HEALTHY-ONLY THRESHOLD ===")
    print(f"LOO nearest cosine mean : {float(np.mean(loo_best_sim)):.8f}")
    print(f"LOO nearest cosine std  : {float(np.std(loo_best_sim)):.8f}")
    print(f"LOO nearest cosine min  : {float(np.min(loo_best_sim)):.8f}")
    print(f"Threshold percentile    : {lower_percentile:.2f}th")
    print(f"Classification threshold: {threshold:.8f}")
    print()

    # ------------------------------------------------------------
    # Healthy independent test scores
    # ------------------------------------------------------------
    healthy_test_sim = cosine_similarity_matrix(healthy_test_emb, train_emb)
    healthy_best_idx = np.argmax(healthy_test_sim, axis=1)
    healthy_best_sim = healthy_test_sim[
        np.arange(len(healthy_test_emb)),
        healthy_best_idx
    ]

    records = []

    for i in range(len(healthy_test_emb)):
        score = float(healthy_best_sim[i])
        records.append({
            "image": healthy_test_names[i],
            "source_class": "no_tumor",
            "true_label": 0,  # healthy
            "true_name": "healthy",
            "nearest_healthy_image": train_names[healthy_best_idx[i]],
            "cosine_similarity": score,
            "predicted_label": int(score < threshold),  # 1 = tumor/anomaly
            "predicted_name": "tumor_or_anomaly" if score < threshold else "healthy",
        })

    # ------------------------------------------------------------
    # Tumor test classes - evaluation only, no training
    # ------------------------------------------------------------
    test_class_dirs = sorted(
        p for p in class_test_root.iterdir()
        if p.is_dir() and p.name.lower() != "no_tumor"
    )

    tumor_embeddings = []
    tumor_meta = []

    print("Encoding tumor test images for evaluation only...")

    for class_dir in test_class_dirs:
        class_images = list_images(class_dir)
        print(f"  {class_dir.name}: {len(class_images)} images")

        for j, p in enumerate(class_images, 1):
            emb = encode_image(model, p, device)
            tumor_embeddings.append(emb)
            tumor_meta.append((p.name, class_dir.name))

            if j % 100 == 0:
                print(f"    encoded {j}/{len(class_images)}")

    tumor_embeddings = np.stack(tumor_embeddings, axis=0)

    tumor_sim = cosine_similarity_matrix(tumor_embeddings, train_emb)
    tumor_best_idx = np.argmax(tumor_sim, axis=1)
    tumor_best_sim = tumor_sim[
        np.arange(len(tumor_embeddings)),
        tumor_best_idx
    ]

    for i, (image_name, class_name) in enumerate(tumor_meta):
        score = float(tumor_best_sim[i])
        records.append({
            "image": image_name,
            "source_class": class_name,
            "true_label": 1,  # tumor
            "true_name": "tumor",
            "nearest_healthy_image": train_names[tumor_best_idx[i]],
            "cosine_similarity": score,
            "predicted_label": int(score < threshold),
            "predicted_name": "tumor_or_anomaly" if score < threshold else "healthy",
        })

    write_csv(out / "cosine_classification_per_image.csv", records)

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------
    y_true = np.array([r["true_label"] for r in records], dtype=int)
    y_pred = np.array([r["predicted_label"] for r in records], dtype=int)
    similarity_scores = np.array(
        [r["cosine_similarity"] for r in records],
        dtype=float
    )

    # Lower cosine => more anomalous, so anomaly score = -cosine.
    anomaly_scores = -similarity_scores

    healthy_scores = similarity_scores[y_true == 0]
    tumor_scores = similarity_scores[y_true == 1]

    if HAS_SKLEARN:
        acc = float(accuracy_score(y_true, y_pred))
        precision = float(precision_score(y_true, y_pred, zero_division=0))
        tumor_recall = float(recall_score(y_true, y_pred, zero_division=0))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = [int(x) for x in cm.ravel()]
        specificity = float(tn / (tn + fp)) if (tn + fp) else None
        roc_auc = float(roc_auc_score(y_true, anomaly_scores))
    else:
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))
        tp = int(np.sum((y_true == 1) & (y_pred == 1)))

        acc = float((tn + tp) / len(y_true))
        precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
        tumor_recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
        specificity = float(tn / (tn + fp)) if (tn + fp) else None
        f1 = (
            float(2 * precision * tumor_recall / (precision + tumor_recall))
            if (precision + tumor_recall) else 0.0
        )
        roc_auc = None

    per_class = {}
    for class_name in sorted(set(r["source_class"] for r in records)):
        rr = [r for r in records if r["source_class"] == class_name]
        vals = np.array([r["cosine_similarity"] for r in rr], dtype=float)
        correct = np.array(
            [r["true_label"] == r["predicted_label"] for r in rr],
            dtype=bool
        )

        per_class[class_name] = {
            "n": len(rr),
            "mean_cosine_similarity": float(np.mean(vals)),
            "std_cosine_similarity": float(np.std(vals)),
            "min_cosine_similarity": float(np.min(vals)),
            "max_cosine_similarity": float(np.max(vals)),
            "classification_accuracy": float(np.mean(correct)),
        }

    summary = {
        "experiment": "healthy-reference cosine similarity classification",
        "encoder": "healthy-only native Plain CNN autoencoder, skip OFF",
        "healthy_train_reference_n": int(len(train_emb)),
        "healthy_test_n": int(np.sum(y_true == 0)),
        "tumor_test_n": int(np.sum(y_true == 1)),
        "latent_dim": int(train_emb.shape[1]),
        "similarity_method": "maximum cosine similarity to healthy-train image embedding bank",
        "threshold_method": (
            "threshold derived ONLY from healthy-train leave-one-out nearest-neighbor "
            f"cosine similarities; lower {lower_percentile:.2f}th percentile"
        ),
        "healthy_acceptance_target": args.healthy_acceptance,
        "threshold": threshold,
        "healthy_train_loo_similarity": {
            "mean": float(np.mean(loo_best_sim)),
            "std": float(np.std(loo_best_sim)),
            "min": float(np.min(loo_best_sim)),
            "max": float(np.max(loo_best_sim)),
        },
        "test_similarity": {
            "healthy_mean": float(np.mean(healthy_scores)),
            "healthy_std": float(np.std(healthy_scores)),
            "tumor_mean": float(np.mean(tumor_scores)),
            "tumor_std": float(np.std(tumor_scores)),
        },
        "classification": {
            "tn_healthy_correct": tn,
            "fp_healthy_as_tumor": fp,
            "fn_tumor_as_healthy": fn,
            "tp_tumor_correct": tp,
            "accuracy": acc,
            "precision_tumor": precision,
            "recall_sensitivity_tumor": tumor_recall,
            "specificity_healthy": specificity,
            "f1_tumor": f1,
            "roc_auc_anomaly_score": roc_auc,
        },
        "per_source_class": per_class,
        "important_note": (
            "The autoencoder/encoder was trained only on healthy images. Tumor test images "
            "are used only after training for evaluation. The classification threshold is "
            "not tuned on tumor images."
        ),
    }

    (out / "cosine_classification_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8"
    )

    # ------------------------------------------------------------
    # Simple visualization
    # ------------------------------------------------------------
    if HAS_MPL:
        plt.figure(figsize=(8, 5))
        plt.hist(healthy_scores, bins=30, alpha=0.6, label="Healthy test")
        plt.hist(tumor_scores, bins=30, alpha=0.6, label="Tumor test")
        plt.axvline(threshold, linestyle="--", label=f"Threshold={threshold:.5f}")
        plt.xlabel("Maximum cosine similarity to healthy reference bank")
        plt.ylabel("Image count")
        plt.title("Healthy-reference cosine similarity classification")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "cosine_similarity_distribution.png", dpi=180)
        plt.close()

    # ------------------------------------------------------------
    # Terminal summary
    # ------------------------------------------------------------
    print()
    print("=" * 88)
    print("FINAL COSINE CLASSIFICATION")
    print("=" * 88)
    print(f"Healthy test n              : {int(np.sum(y_true == 0))}")
    print(f"Tumor test n                : {int(np.sum(y_true == 1))}")
    print(f"Threshold                   : {threshold:.8f}")
    print(f"Healthy mean cosine         : {float(np.mean(healthy_scores)):.8f}")
    print(f"Tumor mean cosine           : {float(np.mean(tumor_scores)):.8f}")
    print()
    print(f"TN healthy correct          : {tn}")
    print(f"FP healthy -> tumor         : {fp}")
    print(f"FN tumor -> healthy         : {fn}")
    print(f"TP tumor correct            : {tp}")
    print()
    print(f"Accuracy                    : {acc:.6f}")
    print(f"Tumor Precision             : {precision:.6f}")
    print(f"Tumor Recall / Sensitivity  : {tumor_recall:.6f}")
    print(f"Healthy Specificity         : {specificity:.6f}")
    print(f"Tumor F1                    : {f1:.6f}")
    if roc_auc is not None:
        print(f"ROC-AUC                     : {roc_auc:.6f}")

    print()
    print("Per source class:")
    for name, vals in per_class.items():
        print(
            f"  {name:15s} n={vals['n']:4d} | "
            f"mean cosine={vals['mean_cosine_similarity']:.8f} | "
            f"acc={vals['classification_accuracy']:.4f}"
        )

    print()
    print("Saved outputs:", out)


if __name__ == "__main__":
    main()
