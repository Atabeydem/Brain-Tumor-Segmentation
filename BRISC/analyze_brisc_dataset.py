import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "brisc_Dataset2025" / "brisc2025"


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a BRISC-style 2D MRI dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=(
            "Path to the BRISC dataset root folder (the folder that contains manifest.csv). "
            f"Default: {DEFAULT_DATASET_PATH}"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional custom output folder. Default: BRISC/outputs/reports",
    )
    return parser.parse_args()


def format_resolution(width, height):
    return f"{width} x {height}"


def get_dtype_and_bit_depth(image_mode):
    if image_mode in ["L", "P", "RGB", "RGBA"]:
        return "uint8", 8
    if image_mode == "I;16":
        return "uint16", 16
    if image_mode == "I":
        return "int32", 32
    if image_mode == "F":
        return "float32", 32
    return "unknown", None


def extract_metadata(row, dataset_path):
    relative_path = str(row["relative_path"]).replace("\\", "/")
    file_path = dataset_path / relative_path

    image = Image.open(file_path)
    width, height = image.size
    mode = image.mode
    dtype, bit_depth = get_dtype_and_bit_depth(mode)

    return {
        "file_path": str(file_path),
        "file_name": row["filename"],
        "task": row["task"],
        "split": row["split"],
        "index": int(row["index"]),
        "tumor_code": row["tumor_code"],
        "tumor_label": row["tumor_label"],
        "plane_code": row["plane_code"],
        "plane_label": row["plane_label"],
        "sequence": row["sequence"],
        "is_mask": bool(row["is_mask"]),
        "linked_image": str(row["linked_image"]).replace("\\", "/") if pd.notna(row["linked_image"]) else "",
        "file_extension": Path(row["filename"]).suffix.lower(),
        "width": width,
        "height": height,
        "resolution_str": format_resolution(width, height),
        "mode": mode,
        "dtype": dtype,
        "bit_depth": bit_depth,
        "is_2d_image": True,
        "num_slices": 1,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
    }


def generate_summary(df):
    classification_df = df[(df["task"] == "classification") & (~df["is_mask"])]
    segmentation_img_df = df[(df["task"] == "segmentation") & (~df["is_mask"])]
    segmentation_mask_df = df[(df["task"] == "segmentation") & (df["is_mask"])]

    summary = {
        "total_files": int(len(df)),
        "total_images": int((~df["is_mask"]).sum()),
        "total_masks": int(df["is_mask"].sum()),
        "classification_images": int(len(classification_df)),
        "segmentation_images": int(len(segmentation_img_df)),
        "segmentation_masks": int(len(segmentation_mask_df)),
        "task_counts": df["task"].value_counts().to_dict(),
        "split_counts": df["split"].value_counts().to_dict(),
        "classification_split_counts": classification_df["split"].value_counts().to_dict(),
        "segmentation_image_split_counts": segmentation_img_df["split"].value_counts().to_dict(),
        "segmentation_mask_split_counts": segmentation_mask_df["split"].value_counts().to_dict(),
        "classification_class_counts": classification_df["tumor_label"].value_counts().to_dict(),
        "segmentation_class_counts": segmentation_img_df["tumor_label"].value_counts().to_dict(),
        "classification_plane_counts": classification_df["plane_label"].value_counts().to_dict(),
        "segmentation_plane_counts": segmentation_img_df["plane_label"].value_counts().to_dict(),
        "all_files_are_2d_images": bool(df["is_2d_image"].all()),
        "most_common_resolution": df["resolution_str"].mode().iloc[0],
        "unique_resolutions_count": int(df["resolution_str"].nunique()),
        "unique_dtypes": sorted(df["dtype"].dropna().unique().tolist()),
        "unique_bit_depths": sorted(df["bit_depth"].dropna().unique().tolist()),
        "most_common_dtype": df["dtype"].mode().iloc[0],
        "most_common_bit_depth": int(df["bit_depth"].mode().iloc[0]) if not df["bit_depth"].dropna().empty else None,
        "min_width": int(df["width"].min()),
        "max_width": int(df["width"].max()),
        "min_height": int(df["height"].min()),
        "max_height": int(df["height"].max()),
        "total_size_mb": round(df["file_size_mb"].sum(), 3),
    }

    return summary


def write_text_report(summary, output_path):
    lines = [
        "DATASET ANALYSIS REPORT",
        "",
        f"Total files: {summary['total_files']}",
        f"Total images: {summary['total_images']}",
        f"Total masks: {summary['total_masks']}",
        f"Classification images: {summary['classification_images']}",
        f"Segmentation images: {summary['segmentation_images']}",
        f"Segmentation masks: {summary['segmentation_masks']}",
        "",
        f"Task counts: {summary['task_counts']}",
        f"Split counts: {summary['split_counts']}",
        f"Classification split counts: {summary['classification_split_counts']}",
        f"Segmentation image split counts: {summary['segmentation_image_split_counts']}",
        f"Segmentation mask split counts: {summary['segmentation_mask_split_counts']}",
        "",
        f"Classification class counts: {summary['classification_class_counts']}",
        f"Segmentation class counts: {summary['segmentation_class_counts']}",
        "",
        f"Classification plane counts: {summary['classification_plane_counts']}",
        f"Segmentation plane counts: {summary['segmentation_plane_counts']}",
        "",
        f"All files are 2D images: {summary['all_files_are_2d_images']}",
        f"Most common resolution: {summary['most_common_resolution']}",
        f"Unique resolutions count: {summary['unique_resolutions_count']}",
        f"Unique dtypes: {summary['unique_dtypes']}",
        f"Unique bit depths: {summary['unique_bit_depths']}",
        f"Most common dtype: {summary['most_common_dtype']}",
        f"Most common bit depth: {summary['most_common_bit_depth']}",
        f"Width range: {summary['min_width']} - {summary['max_width']}",
        f"Height range: {summary['min_height']} - {summary['max_height']}",
        f"Total dataset size (MB): {summary['total_size_mb']}",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    if args.output_dir is None:
        output_dir = Path(__file__).resolve().parent / "outputs" / "reports"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    manifest_path = dataset_path / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in: {dataset_path}")

    manifest_df = pd.read_csv(manifest_path)

    rows = []
    errors = []

    for _, row in manifest_df.iterrows():
        try:
            rows.append(extract_metadata(row, dataset_path))
        except Exception as e:
            errors.append(
                {
                    "filename": str(row.get("filename", "")),
                    "relative_path": str(row.get("relative_path", "")),
                    "error": str(e),
                }
            )

    if not rows:
        raise RuntimeError("No files could be analyzed.")

    df = pd.DataFrame(rows)
    summary = generate_summary(df)

    df.to_csv(output_dir / "file_metadata.csv", index=False)
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_text_report(summary, output_dir / "dataset_report.txt")
    (output_dir / "errors.json").write_text(
        json.dumps(errors, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=== Analysis Completed ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Total files analyzed: {len(df)}")
    print(f"Classification images: {summary['classification_images']}")
    print(f"Segmentation images: {summary['segmentation_images']}")
    print(f"Segmentation masks: {summary['segmentation_masks']}")
    print(f"Most common resolution: {summary['most_common_resolution']}")
    print(f"Most common dtype: {summary['most_common_dtype']}")
    print(f"Most common bit depth: {summary['most_common_bit_depth']}")
    print(f"Output folder: {output_dir.resolve()}")

    if errors:
        print(f"Warning: {len(errors)} file(s) produced errors. Check errors.json")


if __name__ == "__main__":
    main()
