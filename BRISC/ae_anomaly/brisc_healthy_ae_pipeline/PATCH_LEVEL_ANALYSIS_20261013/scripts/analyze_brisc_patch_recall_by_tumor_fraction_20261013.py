#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Analyze tumor-positive patch recall as a function of annotated tumor area.

No training and no threshold changes. The script joins the current patch-score
CSV to the fixed-256 audit manifest and reports how recall changes as the tumor
occupies a larger fraction of a 256x256 patch.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BINS = np.array([0.0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 1.000001], dtype=float)
LABELS = ["0-0.1%", "0.1-0.5%", "0.5-1%", "1-2%", "2-5%", "5-10%", "10-25%", ">25%"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_scores", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores = pd.read_csv(args.patch_scores)
    manifest = pd.read_csv(args.manifest)

    required_scores = {"source_image", "patch_id", "label", "pred"}
    required_manifest = {"source_image", "patch_id", "split", "tumor_fraction", "tumor_pixels"}
    if not required_scores.issubset(scores.columns):
        raise RuntimeError(f"Patch-score CSV missing columns: {sorted(required_scores - set(scores.columns))}")
    if not required_manifest.issubset(manifest.columns):
        raise RuntimeError(f"Manifest missing columns: {sorted(required_manifest - set(manifest.columns))}")

    tumor_manifest = manifest.loc[
        manifest["split"].eq("tumor_test"),
        ["source_image", "patch_id", "tumor_fraction", "tumor_pixels"],
    ].copy()

    merged = scores.merge(
        tumor_manifest,
        on=["source_image", "patch_id"],
        how="left",
        validate="one_to_one",
    )

    pos = merged.loc[merged["label"].eq(1)].copy()
    if pos["tumor_fraction"].isna().any():
        raise RuntimeError(
            f"{int(pos['tumor_fraction'].isna().sum())} positive patches could not be joined to the manifest."
        )

    pos["tumor_fraction_pct"] = pos["tumor_fraction"] * 100.0
    pos["fraction_bin"] = pd.cut(
        pos["tumor_fraction"],
        bins=BINS,
        labels=LABELS,
        include_lowest=True,
        right=False,
    )

    if {"similarity", "threshold"}.issubset(pos.columns):
        # Lower cosine similarity than threshold means more anomalous.
        pos["anomaly_margin"] = pos["threshold"] - pos["similarity"]
    else:
        pos["anomaly_margin"] = np.nan

    rows = []
    for label in LABELS:
        g = pos.loc[pos["fraction_bin"].eq(label)]
        n = len(g)
        detected = int(g["pred"].eq(1).sum()) if n else 0
        rows.append(
            {
                "tumor_fraction_bin": label,
                "n_positive_patches": n,
                "detected": detected,
                "missed": n - detected,
                "recall": detected / n if n else np.nan,
                "mean_tumor_fraction_pct": float(g["tumor_fraction_pct"].mean()) if n else np.nan,
                "median_tumor_fraction_pct": float(g["tumor_fraction_pct"].median()) if n else np.nan,
                "mean_anomaly_margin": (
                    float(g["anomaly_margin"].mean())
                    if n and g["anomaly_margin"].notna().any()
                    else np.nan
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "tumor_fraction_recall.csv", index=False)
    pos.to_csv(out / "tumor_fraction_positive_patches.csv", index=False)

    print("=" * 92)
    print("TUMOR FRACTION VS PATCH RECALL")
    print("=" * 92)
    show = summary.copy()
    show["recall_pct"] = show["recall"] * 100.0
    print(
        show[
            [
                "tumor_fraction_bin",
                "n_positive_patches",
                "detected",
                "missed",
                "recall_pct",
                "median_tumor_fraction_pct",
            ]
        ].to_string(index=False)
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(summary))
    ax.bar(x, summary["recall"].fillna(0) * 100.0)
    ax.set_xticks(x)
    ax.set_xticklabels(summary["tumor_fraction_bin"], rotation=30, ha="right")
    ax.set_ylabel("Patch recall (%)")
    ax.set_xlabel("Annotated tumor area within 256x256 patch")
    ax.set_title("Patch recall vs tumor area fraction")
    ax.set_ylim(0, 100)
    ax.grid(True, axis="y", alpha=0.2)
    for i, r in summary.iterrows():
        if pd.notna(r["recall"]):
            ax.text(
                i,
                r["recall"] * 100.0 + 2,
                f"n={int(r['n_positive_patches'])}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    fig.savefig(out / "tumor_fraction_recall.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    overall = float(pos["pred"].eq(1).mean())
    print()
    print(f"Overall tumor-positive patch recall: {overall * 100:.2f}%")
    print("Saved:")
    print(out / "tumor_fraction_recall.csv")
    print(out / "tumor_fraction_positive_patches.csv")
    print(out / "tumor_fraction_recall.png")


if __name__ == "__main__":
    main()
