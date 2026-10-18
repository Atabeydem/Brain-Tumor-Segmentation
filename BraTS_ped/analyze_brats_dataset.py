import argparse
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd


EXPECTED_MODALITIES = {"t1n", "t1c", "t2w", "t2f", "seg"}


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze a BraTS-style NIfTI dataset.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Root folder containing Training and Validation folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Optional custom output folder. If not given, outputs will be saved under BraTS/outputs/reports.",
    )
    parser.add_argument(
        "--inspect_seg_labels",
        action="store_true",
        help="Load segmentation masks and extract unique label values.",
    )
    return parser.parse_args()


def strip_nii_gz(filename: str) -> str:
    if filename.endswith(".nii.gz"):
        return filename[:-7]
    return filename


def get_split_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    if "training" in parts:
        return "Training"
    if "validation" in parts:
        return "Validation"
    return "Unknown"


def parse_case_and_modality(file_path: Path):
    base_name = strip_nii_gz(file_path.name)
    parts = base_name.split("-")
    if len(parts) < 2:
        return base_name, "unknown"

    modality = parts[-1].lower()
    case_id = "-".join(parts[:-1])
    return case_id, modality


def format_shape(shape):
    return " x ".join(str(x) for x in shape)


def format_spacing(spacing):
    return " x ".join(f"{x:.3f}" for x in spacing)


def extract_metadata(file_path: Path, inspect_seg_labels: bool = False):
    img = nib.load(str(file_path))
    header = img.header

    shape = tuple(int(x) for x in img.shape)
    spacing = tuple(float(x) for x in header.get_zooms()[: len(shape)])
    dtype = np.dtype(header.get_data_dtype())
    bit_depth = int(dtype.itemsize * 8)

    case_id, modality = parse_case_and_modality(file_path)

    is_3d_volume = len(shape) == 3 and shape[2] > 1
    axial_slices = shape[2] if len(shape) >= 3 else 1
    coronal_slices = shape[1] if len(shape) >= 2 else 1
    sagittal_slices = shape[0] if len(shape) >= 1 else 1

    seg_labels = None
    if inspect_seg_labels and modality == "seg":
        data = np.asanyarray(img.dataobj)
        seg_labels = [float(x) for x in np.unique(data).tolist()]

    return {
        "file_path": str(file_path),
        "file_name": file_path.name,
        "split": get_split_from_path(file_path),
        "case_id": case_id,
        "modality": modality,
        "shape": str(shape),
        "shape_str": format_shape(shape),
        "spacing": str(spacing),
        "spacing_str": format_spacing(spacing),
        "x_dim": shape[0] if len(shape) > 0 else None,
        "y_dim": shape[1] if len(shape) > 1 else None,
        "z_dim": shape[2] if len(shape) > 2 else None,
        "axial_slices": axial_slices,
        "coronal_slices": coronal_slices,
        "sagittal_slices": sagittal_slices,
        "dtype": str(dtype),
        "bit_depth": bit_depth,
        "is_3d_volume": is_3d_volume,
        "file_size_mb": round(file_path.stat().st_size / (1024 * 1024), 3),
        "seg_labels": str(seg_labels) if seg_labels is not None else "",
    }


def build_case_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (split, case_id), group in df.groupby(["split", "case_id"]):
        modalities = sorted(group["modality"].tolist())
        missing_modalities = sorted(EXPECTED_MODALITIES - set(modalities))
        unique_shapes = sorted(group["shape_str"].unique().tolist())
        unique_spacings = sorted(group["spacing_str"].unique().tolist())
        unique_dtypes = sorted(group["dtype"].unique().tolist())
        unique_bits = sorted(group["bit_depth"].unique().tolist())
        axial_values = sorted(group["axial_slices"].unique().tolist())

        seg_rows = group[group["modality"] == "seg"]
        seg_labels = ""
        if not seg_rows.empty:
            non_empty = seg_rows["seg_labels"][seg_rows["seg_labels"] != ""]
            if len(non_empty) > 0:
                seg_labels = non_empty.iloc[0]

        rows.append(
            {
                "split": split,
                "case_id": case_id,
                "num_files": len(group),
                "modalities_found": ", ".join(modalities),
                "missing_modalities": ", ".join(missing_modalities),
                "all_files_are_3d": bool(group["is_3d_volume"].all()),
                "unique_shapes": " | ".join(unique_shapes),
                "unique_spacings": " | ".join(unique_spacings),
                "unique_dtypes": " | ".join(unique_dtypes),
                "unique_bit_depths": " | ".join(str(x) for x in unique_bits),
                "axial_slices_values": " | ".join(str(x) for x in axial_values),
                "seg_labels": seg_labels,
            }
        )

    return pd.DataFrame(rows).sort_values(["split", "case_id"]).reset_index(drop=True)


def generate_summary(df: pd.DataFrame, case_df: pd.DataFrame):
    summary = {
        "total_files": int(len(df)),
        "total_cases": int(case_df["case_id"].nunique()),
        "files_per_split": df["split"].value_counts().to_dict(),
        "cases_per_split": case_df["split"].value_counts().to_dict(),
        "modalities_count": df["modality"].value_counts().to_dict(),
        "all_files_are_3d_volumes": bool(df["is_3d_volume"].all()),
        "unique_shapes": sorted(df["shape_str"].unique().tolist()),
        "unique_spacings": sorted(df["spacing_str"].unique().tolist()),
        "unique_dtypes": sorted(df["dtype"].unique().tolist()),
        "unique_bit_depths": sorted(df["bit_depth"].unique().tolist()),
        "most_common_shape": df["shape_str"].mode().iloc[0],
        "most_common_spacing": df["spacing_str"].mode().iloc[0],
        "most_common_dtype": df["dtype"].mode().iloc[0],
        "most_common_bit_depth": int(df["bit_depth"].mode().iloc[0]),
        "most_common_axial_slices": int(df["axial_slices"].mode().iloc[0]),
    }
    return summary


def write_text_report(summary: dict, output_path: Path):
    lines = [
        "DATASET ANALYSIS REPORT",
        "",
        f"Total files: {summary['total_files']}",
        f"Total cases: {summary['total_cases']}",
        f"Files per split: {summary['files_per_split']}",
        f"Cases per split: {summary['cases_per_split']}",
        f"Modalities count: {summary['modalities_count']}",
        "",
        f"All files are 3D volumes: {summary['all_files_are_3d_volumes']}",
        f"Most common shape: {summary['most_common_shape']}",
        f"Most common spacing (mm): {summary['most_common_spacing']}",
        f"Most common axial slice count: {summary['most_common_axial_slices']}",
        f"Most common dtype: {summary['most_common_dtype']}",
        f"Most common bit depth: {summary['most_common_bit_depth']}",
        "",
        f"Unique shapes: {summary['unique_shapes']}",
        f"Unique spacings: {summary['unique_spacings']}",
        f"Unique dtypes: {summary['unique_dtypes']}",
        f"Unique bit depths: {summary['unique_bit_depths']}",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    script_dir = Path(__file__).resolve().parent

    if args.output_dir is None:
        
        output_dir = script_dir / "outputs" / "reports"
    else:
        output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    nii_files = sorted(dataset_path.rglob("*.nii.gz"))

    if not nii_files:

        raise FileNotFoundError("No .nii.gz files found under the dataset path.")

    rows = []
    errors = []

    for file_path in nii_files:

        try:
            rows.append(extract_metadata(file_path, inspect_seg_labels=args.inspect_seg_labels))

        except Exception as e:
            errors.append(
                {"file_path": str(file_path), "error": str(e)}
                
                )

    if not rows:
        raise RuntimeError("No files could be analyzed."   )

    df = pd.DataFrame(rows)
    case_df = build_case_summary(df)
    summary = generate_summary(df, case_df)

    df.to_csv(output_dir / "volume_metadata.csv", index=False)
    case_df.to_csv(output_dir / "case_summary.csv", index=False)
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
    print(f"Total unique cases: {case_df['case_id'].nunique()}")
    print(f"All files are 3D volumes: {summary['all_files_are_3d_volumes']}")
    print(f"Most common shape: {summary['most_common_shape']}")
    print(f"Most common spacing: {summary['most_common_spacing']} mm")
    print(f"Most common axial slice count: {summary['most_common_axial_slices']}")
    print(f"Most common dtype: {summary['most_common_dtype']}")
    print(f"Most common bit depth: {summary['most_common_bit_depth']}")
    print(f"Output folder: {output_dir.resolve()}")

    if errors:
        print(f"Warning: {len(errors)} file(s) produced errors. Check errors.json")


if __name__ == "__main__":
    main()