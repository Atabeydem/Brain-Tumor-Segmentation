#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BRISC fixed-256 position/resolution-matched cosine evaluation.

Why
---
The healthy latent-space controls showed that the 2048-D representation strongly
encodes patch position, source resolution, and padding/valid geometry.

This evaluation therefore compares each query patch only with the most closely
matched healthy reference bank available.

Matching hierarchy
------------------
1) exact:
   original resolution + row/col position + valid patch geometry
2) resolution_position:
   original resolution + row/col position
3) position_geometry:
   row/col position + valid patch geometry
4) position:
   row/col position
5) global:
   all healthy train patches

For each matching group, the anomaly threshold is estimated from HEALTHY TRAIN
ONLY using image-disjoint nearest-neighbor cosine scores:
- query patch cannot match any patch from the same source image
- default threshold = 5th percentile (95% healthy acceptance)

No retraining. No patch embedding averaging.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------

def read_csv_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def l2_normalize(x, eps=1e-12):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(n, eps, None)


# ---------------------------------------------------------------------
# Metadata / matching keys
# ---------------------------------------------------------------------

def make_record(source_image, patch_id, row_idx, col_idx,
                valid_width, valid_height, original_width, original_height):
    return {
        "source_image": str(source_image),
        "patch_id": str(patch_id),
        "row_idx": int(row_idx),
        "col_idx": int(col_idx),
        "valid_width": int(valid_width),
        "valid_height": int(valid_height),
        "original_width": int(original_width),
        "original_height": int(original_height),
    }


def key_exact(r):
    return (
        r["original_width"], r["original_height"],
        r["row_idx"], r["col_idx"],
        r["valid_width"], r["valid_height"],
    )


def key_resolution_position(r):
    return (
        r["original_width"], r["original_height"],
        r["row_idx"], r["col_idx"],
    )


def key_position_geometry(r):
    return (
        r["row_idx"], r["col_idx"],
        r["valid_width"], r["valid_height"],
    )


def key_position(r):
    return (r["row_idx"], r["col_idx"])


LEVELS = [
    ("exact", key_exact),
    ("resolution_position", key_resolution_position),
    ("position_geometry", key_position_geometry),
    ("position", key_position),
    ("global", lambda r: ("global",)),
]


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def binary_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {
        "n": int(len(y_true)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": float((tp + tn) / max(len(y_true), 1)),
        "precision_tumor": float(tp / max(tp + fp, 1)),
        "recall_sensitivity_tumor": float(tp / max(tp + fn, 1)),
        "specificity_healthy": float(tn / max(tn + fp, 1)),
        "f1_tumor": float(
            2 * tp / max(2 * tp + fp + fn, 1)
        ),
    }


# ---------------------------------------------------------------------
# Train banks + thresholds
# ---------------------------------------------------------------------

def build_group_indices(records):
    by_level = {}
    for level_name, key_fn in LEVELS:
        groups = defaultdict(list)
        for i, r in enumerate(records):
            groups[key_fn(r)].append(i)
        by_level[level_name] = {
            k: np.asarray(v, dtype=np.int64)
            for k, v in groups.items()
        }
    return by_level


def image_disjoint_nn_scores(x_norm, records, indices, chunk=256):
    """
    Nearest-neighbor cosine within `indices`, excluding all patches
    from the same source image.

    Returns only finite scores.
    """
    idx = np.asarray(indices, dtype=np.int64)
    if len(idx) < 2:
        return np.asarray([], dtype=np.float32)

    bank = x_norm[idx]
    image_names = np.asarray(
        [records[i]["source_image"] for i in idx],
        dtype=object,
    )

    scores = []

    for start in range(0, len(idx), chunk):
        q = bank[start:start + chunk]
        sims = q @ bank.T

        q_names = image_names[start:start + len(q)]
        same_img = q_names[:, None] == image_names[None, :]
        sims[same_img] = -np.inf

        s = np.max(sims, axis=1)
        s = s[np.isfinite(s)]
        scores.append(s.astype(np.float32))

    if not scores:
        return np.asarray([], dtype=np.float32)

    return np.concatenate(scores)


def prepare_group_models(train_emb, train_records, healthy_acceptance,
                         min_reference_patches, min_threshold_samples):
    """
    Precompute:
    - reference indices
    - group-specific healthy-only thresholds

    A group is usable only if it has enough reference patches AND enough
    image-disjoint threshold samples.
    """
    x_norm = l2_normalize(train_emb.astype(np.float32))
    grouped = build_group_indices(train_records)

    lower_q = (1.0 - healthy_acceptance) * 100.0
    models = {}

    print("Building matched healthy reference groups / thresholds...")

    for level_name, _ in LEVELS:
        models[level_name] = {}
        keys = list(grouped[level_name].keys())

        for j, key in enumerate(keys, 1):
            idx = grouped[level_name][key]

            if len(idx) < min_reference_patches:
                continue

            loo = image_disjoint_nn_scores(
                x_norm, train_records, idx
            )

            if len(loo) < min_threshold_samples:
                continue

            models[level_name][key] = {
                "indices": idx,
                "threshold": float(np.percentile(loo, lower_q)),
                "threshold_n": int(len(loo)),
                "reference_n": int(len(idx)),
                "train_nn_mean": float(np.mean(loo)),
                "train_nn_std": float(np.std(loo)),
            }

        print(
            f"  {level_name:22s}: "
            f"{len(models[level_name])} usable group(s)"
        )

    if ("global",) not in models["global"]:
        raise RuntimeError(
            "Global fallback group could not be constructed. "
            "Reduce --min_reference_patches / --min_threshold_samples."
        )

    return x_norm, models


def choose_group(record, models):
    for level_name, key_fn in LEVELS:
        key = key_fn(record)
        if key in models[level_name]:
            return level_name, key, models[level_name][key]
    raise RuntimeError("No matching group found, including global fallback.")


def score_queries(query_emb, query_records, train_norm, models):
    q_norm = l2_normalize(query_emb.astype(np.float32))
    out = []

    # Cache matched banks because many query patches share the same group.
    bank_cache = {}

    for i, (q, r) in enumerate(zip(q_norm, query_records)):
        level_name, key, model = choose_group(r, models)
        cache_key = (level_name, key)

        if cache_key not in bank_cache:
            ref_idx = model["indices"]
            bank_cache[cache_key] = train_norm[ref_idx]

        bank = bank_cache[cache_key]
        score = float(np.max(bank @ q))
        pred = int(score < model["threshold"])

        out.append({
            **r,
            "match_level": level_name,
            "match_key": repr(key),
            "reference_n": int(model["reference_n"]),
            "threshold_n": int(model["threshold_n"]),
            "threshold": float(model["threshold"]),
            "cosine_similarity": score,
            "pred_label": pred,
        })

        if (i + 1) % 500 == 0 or (i + 1) == len(query_records):
            print(f"  scored {i+1}/{len(query_records)}")

    return out


# ---------------------------------------------------------------------
# NPZ metadata loaders
# ---------------------------------------------------------------------

def load_train_or_healthy_npz(path: Path):
    d = np.load(path, allow_pickle=True)

    emb = d["embeddings"].astype(np.float32)

    records = []
    for i in range(len(emb)):
        records.append(
            make_record(
                d["source_image"][i],
                d["patch_id"][i],
                d["row_idx"][i],
                d["col_idx"][i],
                d["valid_width"][i],
                d["valid_height"][i],
                d["original_width"][i],
                d["original_height"][i],
            )
        )

    return emb, records


def load_tumor_npz_with_manifest(npz_path: Path, manifest_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    emb = d["embeddings"].astype(np.float32)

    manifest = read_csv_rows(manifest_path)
    tumor_manifest = {
        (r["source_image"], r["patch_id"]): r
        for r in manifest
        if r["split"] == "tumor_test"
    }

    records = []

    for i in range(len(emb)):
        source_image = str(d["source_image"][i])
        patch_id = str(d["patch_id"][i])
        key = (source_image, patch_id)

        if key not in tumor_manifest:
            raise RuntimeError(
                f"Tumor embedding could not be joined to manifest: {key}"
            )

        m = tumor_manifest[key]

        r = make_record(
            source_image,
            patch_id,
            d["row_idx"][i],
            d["col_idx"][i],
            d["valid_width"][i],
            d["valid_height"][i],
            m["original_width"],
            m["original_height"],
        )

        r["true_label"] = int(m["patch_label"])
        r["tumor_pixels"] = int(m["tumor_pixels"])
        r["tumor_fraction"] = float(m["tumor_fraction"])
        r["source_group"] = (
            "tumor_positive"
            if r["true_label"] == 1
            else "tumor_source_mask_negative"
        )

        records.append(r)

    return emb, records


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def save_histogram(rows, out_png: Path):
    groups = [
        "healthy_test",
        "tumor_source_mask_negative",
        "tumor_positive",
    ]

    fig, ax = plt.subplots(figsize=(10, 7))

    for g in groups:
        vals = [
            r["cosine_similarity"]
            for r in rows
            if r["source_group"] == g
        ]
        ax.hist(
            vals,
            bins=60,
            alpha=0.45,
            density=True,
            label=f"{g} (n={len(vals)})",
        )

    ax.set_title("Matched cosine similarity distributions")
    ax.set_xlabel("Cosine similarity to matched healthy reference bank")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)
    ap.add_argument("--train_embeddings", required=True)
    ap.add_argument("--healthy_test_embeddings", required=True)
    ap.add_argument("--tumor_embeddings", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--healthy_acceptance", type=float, default=0.95)
    ap.add_argument("--min_reference_patches", type=int, default=20)
    ap.add_argument("--min_threshold_samples", type=int, default=20)

    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_emb, train_records = load_train_or_healthy_npz(
        Path(args.train_embeddings)
    )

    healthy_emb, healthy_records = load_train_or_healthy_npz(
        Path(args.healthy_test_embeddings)
    )

    tumor_emb, tumor_records = load_tumor_npz_with_manifest(
        Path(args.tumor_embeddings),
        Path(args.manifest),
    )

    if not (
        train_emb.shape[1]
        == healthy_emb.shape[1]
        == tumor_emb.shape[1]
    ):
        raise RuntimeError(
            f"Embedding dimensions disagree: "
            f"train={train_emb.shape}, healthy={healthy_emb.shape}, tumor={tumor_emb.shape}"
        )

    for r in healthy_records:
        r["true_label"] = 0
        r["tumor_pixels"] = 0
        r["tumor_fraction"] = 0.0
        r["source_group"] = "healthy_test"

    print("=" * 98)
    print("BRISC POSITION / RESOLUTION / GEOMETRY-MATCHED COSINE EVALUATION")
    print("=" * 98)
    print(f"Healthy train references   : {len(train_emb)}")
    print(f"Independent healthy test   : {len(healthy_emb)}")
    print(f"Tumor-source test patches  : {len(tumor_emb)}")
    print(f"Embedding dim              : {train_emb.shape[1]}")
    print(f"Healthy acceptance target  : {args.healthy_acceptance:.3f}")
    print(f"Min reference patches      : {args.min_reference_patches}")
    print(f"Min threshold samples      : {args.min_threshold_samples}")
    print("Embedding averaging        : NONE")
    print("Threshold data             : HEALTHY TRAIN ONLY")
    print()

    train_norm, models = prepare_group_models(
        train_emb,
        train_records,
        args.healthy_acceptance,
        args.min_reference_patches,
        args.min_threshold_samples,
    )

    print()
    print("Scoring independent healthy-test patches...")
    healthy_scored = score_queries(
        healthy_emb, healthy_records, train_norm, models
    )

    print()
    print("Scoring tumor-source patches...")
    tumor_scored = score_queries(
        tumor_emb, tumor_records, train_norm, models
    )

    rows = healthy_scored + tumor_scored

    write_csv(
        out / "matched_patch_cosine_scores.csv",
        rows,
    )

    patch_metrics = binary_metrics(
        [r["true_label"] for r in rows],
        [r["pred_label"] for r in rows],
    )

    subgroup_summary = {}
    for g in [
        "healthy_test",
        "tumor_source_mask_negative",
        "tumor_positive",
    ]:
        rr = [r for r in rows if r["source_group"] == g]
        subgroup_summary[g] = {
            "n": int(len(rr)),
            "mean_cosine_similarity": float(
                np.mean([r["cosine_similarity"] for r in rr])
            ),
            "std_cosine_similarity": float(
                np.std([r["cosine_similarity"] for r in rr])
            ),
            "predicted_tumor_count": int(
                sum(r["pred_label"] for r in rr)
            ),
            "predicted_tumor_rate": float(
                np.mean([r["pred_label"] for r in rr])
            ),
        }

    match_level_counts = dict(
        Counter(r["match_level"] for r in rows)
    )

    # ------------------------------------------------------------
    # Image-level: any anomalous patch => tumor
    # ------------------------------------------------------------
    by_image = defaultdict(list)
    image_true = {}

    for r in rows:
        by_image[r["source_image"]].append(r)
        image_true[r["source_image"]] = (
            0 if r["source_group"] == "healthy_test" else 1
        )

    image_rows = []

    tumor_image_detected_any = 0
    tumor_image_localized_true_patch = 0
    tumor_image_detected_only_mask_negative = 0
    tumor_image_missed = 0

    for source_image, rr in sorted(by_image.items()):
        true = int(image_true[source_image])
        pred_any = int(any(r["pred_label"] == 1 for r in rr))

        true_tumor_patch_hit = int(
            any(
                r["true_label"] == 1 and r["pred_label"] == 1
                for r in rr
            )
        )

        mask_negative_hit = int(
            any(
                r["source_group"] == "tumor_source_mask_negative"
                and r["pred_label"] == 1
                for r in rr
            )
        )

        if true == 1:
            if pred_any:
                tumor_image_detected_any += 1
                if true_tumor_patch_hit:
                    tumor_image_localized_true_patch += 1
                else:
                    tumor_image_detected_only_mask_negative += 1
            else:
                tumor_image_missed += 1

        image_rows.append({
            "source_image": source_image,
            "true_label": true,
            "pred_label": pred_any,
            "n_patches": int(len(rr)),
            "n_predicted_anomalous_patches": int(
                sum(r["pred_label"] for r in rr)
            ),
            "n_true_tumor_patches": int(
                sum(r["true_label"] for r in rr)
            ),
            "true_tumor_patch_hit": true_tumor_patch_hit,
            "mask_negative_patch_hit": mask_negative_hit,
            "min_matched_cosine": float(
                min(r["cosine_similarity"] for r in rr)
            ),
        })

    write_csv(
        out / "matched_image_cosine_scores.csv",
        image_rows,
    )

    image_metrics = binary_metrics(
        [r["true_label"] for r in image_rows],
        [r["pred_label"] for r in image_rows],
    )

    tumor_image_count = int(
        sum(r["true_label"] == 1 for r in image_rows)
    )

    localization = {
        "n_tumor_images": tumor_image_count,
        "detected_by_any_anomalous_patch": tumor_image_detected_any,
        "detected_by_any_anomalous_patch_rate": float(
            tumor_image_detected_any / max(tumor_image_count, 1)
        ),
        "at_least_one_true_tumor_patch_detected": tumor_image_localized_true_patch,
        "true_tumor_patch_localization_rate": float(
            tumor_image_localized_true_patch / max(tumor_image_count, 1)
        ),
        "detected_only_because_mask_negative_patch_was_anomalous":
            tumor_image_detected_only_mask_negative,
        "missed_tumor_images": tumor_image_missed,
    }

    # ------------------------------------------------------------
    # Threshold table
    # ------------------------------------------------------------
    threshold_rows = []
    for level_name, level_models in models.items():
        for key, m in level_models.items():
            threshold_rows.append({
                "match_level": level_name,
                "match_key": repr(key),
                "reference_n": m["reference_n"],
                "threshold_n": m["threshold_n"],
                "threshold": m["threshold"],
                "train_nn_mean": m["train_nn_mean"],
                "train_nn_std": m["train_nn_std"],
            })

    write_csv(
        out / "matched_threshold_groups.csv",
        threshold_rows,
    )

    save_histogram(
        rows,
        out / "matched_cosine_distributions.png",
    )

    # Prior corrected-but-unmatched baseline from the previous experiment.
    previous = {
        "patch_sensitivity": 0.1347980997624703,
        "image_sensitivity_any_patch": 0.38372093023255816,
        "image_specificity": 0.9785714285714285,
    }

    summary = {
        "experiment": "position/resolution/geometry-matched cosine similarity",
        "embedding_dim": int(train_emb.shape[1]),
        "healthy_train_reference_patches": int(len(train_emb)),
        "healthy_test_patches": int(len(healthy_emb)),
        "tumor_source_test_patches": int(len(tumor_emb)),
        "healthy_acceptance_target": args.healthy_acceptance,
        "matching_hierarchy": [
            "exact: original_resolution + row/col + valid_geometry",
            "resolution_position",
            "position_geometry",
            "position",
            "global",
        ],
        "match_level_usage": match_level_counts,
        "patch_level": patch_metrics,
        "patch_subgroups": subgroup_summary,
        "image_level_any_anomalous_patch": image_metrics,
        "tumor_image_localization": localization,
        "previous_unmatched_patch_baseline": previous,
        "delta_vs_previous": {
            "patch_sensitivity": float(
                patch_metrics["recall_sensitivity_tumor"]
                - previous["patch_sensitivity"]
            ),
            "image_sensitivity_any_patch": float(
                image_metrics["recall_sensitivity_tumor"]
                - previous["image_sensitivity_any_patch"]
            ),
            "image_specificity": float(
                image_metrics["specificity_healthy"]
                - previous["image_specificity"]
            ),
        },
        "important_note": (
            "All thresholds are derived from healthy train patches only. "
            "Tumor masks are used only as evaluation labels. "
            "No patch embeddings are averaged."
        ),
    }

    (out / "matched_cosine_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 98)
    print("MATCH LEVEL USAGE")
    print("=" * 98)
    print(json.dumps(match_level_counts, indent=2))

    print()
    print("=" * 98)
    print("PATCH LEVEL")
    print("=" * 98)
    print(json.dumps(patch_metrics, indent=2))

    print()
    print("=" * 98)
    print("PATCH SUBGROUPS")
    print("=" * 98)
    print(json.dumps(subgroup_summary, indent=2))

    print()
    print("=" * 98)
    print("IMAGE LEVEL — ANY ANOMALOUS PATCH => TUMOR")
    print("=" * 98)
    print(json.dumps(image_metrics, indent=2))

    print()
    print("=" * 98)
    print("TUMOR IMAGE LOCALIZATION CHECK")
    print("=" * 98)
    print(json.dumps(localization, indent=2))

    print()
    print("=" * 98)
    print("CHANGE VS PREVIOUS UNMATCHED PATCH COSINE")
    print("=" * 98)
    print(
        f"Patch sensitivity : "
        f"{previous['patch_sensitivity']*100:.2f}% -> "
        f"{patch_metrics['recall_sensitivity_tumor']*100:.2f}% "
        f"({summary['delta_vs_previous']['patch_sensitivity']*100:+.2f} pp)"
    )
    print(
        f"Image sensitivity : "
        f"{previous['image_sensitivity_any_patch']*100:.2f}% -> "
        f"{image_metrics['recall_sensitivity_tumor']*100:.2f}% "
        f"({summary['delta_vs_previous']['image_sensitivity_any_patch']*100:+.2f} pp)"
    )
    print(
        f"Image specificity : "
        f"{previous['image_specificity']*100:.2f}% -> "
        f"{image_metrics['specificity_healthy']*100:.2f}% "
        f"({summary['delta_vs_previous']['image_specificity']*100:+.2f} pp)"
    )

    print()
    print("Saved outputs:", out)
    print(" ", out / "matched_cosine_summary.json")
    print(" ", out / "matched_patch_cosine_scores.csv")
    print(" ", out / "matched_image_cosine_scores.csv")
    print(" ", out / "matched_threshold_groups.csv")
    print(" ", out / "matched_cosine_distributions.png")


if __name__ == "__main__":
    main()
