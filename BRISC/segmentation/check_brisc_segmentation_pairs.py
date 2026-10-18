r"""
Check BRISC segmentation image-mask pairs.

Run from:
    C:\Users\Atabey\Brain-Tumor-Segmentation\BRISC\segmentation

Example:
    python .\check_brisc_segmentation_pairs.py
"""

from pathlib import Path
import re
import pandas as pd

DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "brisc_Dataset2025" / "brisc2025"


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True, "1": True, "yes": True, "y": True,
            "false": False, "0": False, "no": False, "n": False,
        })
        .fillna(False)
        .astype(bool)
    )


def norm_path(value) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip().replace("\\", "/").lower()


def simplified_stem(value) -> str:
    text = norm_path(value)
    if not text:
        return ""
    stem = Path(text).stem.lower()
    stem = re.sub(r"(_mask|_masks|_seg|_segmentation|-mask|-seg|-segmentation)$", "", stem)
    return stem


def row_pair_key(row) -> str:
    parts = []
    for col in ["split", "index", "tumor_code", "plane_code", "sequence"]:
        if col in row and not pd.isna(row[col]):
            parts.append(str(row[col]).strip().lower())
    return "|".join(parts)


def build_mask_lookup(masks: pd.DataFrame) -> dict:
    lookup = {}

    def add(key, row):
        if key and key not in lookup:
            lookup[key] = row

    for _, row in masks.iterrows():
        rel = norm_path(row.get("relative_path", ""))
        filename = norm_path(row.get("filename", ""))
        linked = norm_path(row.get("linked_image", ""))

        for value in [linked, Path(linked).name if linked else ""]:
            add(value, row)
            add(simplified_stem(value), row)

        for value in [rel, filename, Path(rel).name if rel else ""]:
            add(value, row)
            add(simplified_stem(value), row)

        add(row_pair_key(row), row)

    return lookup


def find_mask_for_image(image_row, mask_lookup):
    rel = norm_path(image_row.get("relative_path", ""))
    filename = norm_path(image_row.get("filename", ""))
    candidates = [
        rel,
        filename,
        Path(rel).name if rel else "",
        simplified_stem(rel),
        simplified_stem(filename),
        row_pair_key(image_row),
    ]
    for key in candidates:
        if key and key in mask_lookup:
            return mask_lookup[key]
    return None


def main():
    dataset_path = DEFAULT_DATASET_PATH
    manifest_path = dataset_path / "manifest.csv"

    print("Dataset path:", dataset_path)
    print("Dataset exists:", dataset_path.exists())
    print("Manifest exists:", manifest_path.exists())

    if not manifest_path.exists():
        raise SystemExit("manifest.csv bulunamadı. Dataset path yanlış olabilir.")

    manifest = pd.read_csv(manifest_path)
    manifest["is_mask"] = to_bool_series(manifest["is_mask"])

    df = manifest[manifest["task"] == "segmentation"].copy()
    images = df[~df["is_mask"]].reset_index(drop=True)
    masks = df[df["is_mask"]].reset_index(drop=True)

    print("Segmentation image rows:", len(images))
    print("Segmentation mask rows:", len(masks))
    print("Splits:", df["split"].value_counts().to_dict())

    mask_lookup = build_mask_lookup(masks)
    pairs = []
    missing = []

    for _, image_row in images.iterrows():
        mask_row = find_mask_for_image(image_row, mask_lookup)
        if mask_row is None:
            missing.append(image_row)
        else:
            pairs.append((image_row, mask_row))

    print("Matched pairs:", len(pairs))
    print("Unmatched images:", len(missing))

    if pairs:
        image_row, mask_row = pairs[0]
        image_path = dataset_path / norm_path(image_row["relative_path"])
        mask_path = dataset_path / norm_path(mask_row["relative_path"])
        print("Sample image:", image_path)
        print("Sample image exists:", image_path.exists())
        print("Sample mask:", mask_path)
        print("Sample mask exists:", mask_path.exists())

    if missing:
        print("\nFirst unmatched examples:")
        for row in missing[:5]:
            print(" -", row.get("relative_path", row.get("filename", "?")))
        raise SystemExit("Bazı image-mask eşleşmeleri bulunamadı. Manifest formatını ayrıca kontrol etmek gerekiyor.")


if __name__ == "__main__":
    main()
