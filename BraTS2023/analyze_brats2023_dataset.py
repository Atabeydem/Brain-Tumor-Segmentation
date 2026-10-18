import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

EXPECTED_MODALITIES = {"t1n", "t1c", "t2w", "t2f", "seg"}


def format_spacing(zooms):
    return tuple(round(float(z), 5) for z in zooms)


def detect_modality(file_path: Path) -> str:
    name = file_path.name.lower()
    for suffix in ["seg", "t1c", "t1n", "t2f", "t2w"]:
        if name.endswith(f"-{suffix}.nii.gz"):
            return suffix
    return "unknown"


def inspect_segmentation_labels(seg_paths, max_cases=None):
    label_counter = Counter()
    inspected_cases = 0

    for seg_path in seg_paths:
        if max_cases is not None and inspected_cases >= max_cases:
            break

        img = nib.load(str(seg_path))
        data = np.asarray(img.dataobj)
        unique_vals = np.unique(data)
        for v in unique_vals:
            if float(v).is_integer():
                label_counter[int(v)] += 1
            else:
                label_counter[float(v)] += 1
        inspected_cases += 1

    return dict(sorted(label_counter.items(), key=lambda x: x[0])), inspected_cases


def build_report(dataset_path: Path, inspect_seg_labels: bool, max_seg_cases: int | None):
    nii_files = sorted(dataset_path.rglob("*.nii.gz"))
    if not nii_files:
        raise FileNotFoundError(f"No .nii.gz files found under: {dataset_path}")

    file_rows = []
    case_to_modalities = defaultdict(set)
    case_to_files = defaultdict(list)

    shape_counter = Counter()
    spacing_counter = Counter()
    dtype_counter = Counter()
    bit_depth_counter = Counter()
    ndim_counter = Counter()
    slice_counter = Counter()
    extension_counter = Counter()
    modality_counter = Counter()
    orientation_counter = Counter()
    seg_paths = []

    for file_path in nii_files:
        case_id = file_path.parent.name
        modality = detect_modality(file_path)
        extension = ".nii.gz" if file_path.name.lower().endswith(".nii.gz") else file_path.suffix

        img = nib.load(str(file_path))
        shape = tuple(int(x) for x in img.shape)
        ndim = len(shape)
        zooms = img.header.get_zooms()[:ndim]
        spacing = format_spacing(zooms)
        dtype = str(img.get_data_dtype())
        bit_depth = int(np.dtype(img.get_data_dtype()).itemsize * 8)
        orientation = "".join(nib.aff2axcodes(img.affine)) if img.affine is not None else "unknown"
        slice_count = shape[2] if ndim >= 3 else None
        file_size_mb = round(file_path.stat().st_size / (1024 * 1024), 4)

        row = {
            "case_id": case_id,
            "file_name": file_path.name,
            "file_path": str(file_path),
            "extension": extension,
            "modality": modality,
            "shape": " x ".join(map(str, shape)),
            "ndim": ndim,
            "spacing_mm": " x ".join(map(str, spacing)),
            "dtype": dtype,
            "bit_depth": bit_depth,
            "orientation": orientation,
            "slice_count_axis2": slice_count,
            "file_size_mb": file_size_mb,
        }
        file_rows.append(row)

        case_to_modalities[case_id].add(modality)
        case_to_files[case_id].append(file_path.name)
        shape_counter[shape] += 1
        spacing_counter[spacing] += 1
        dtype_counter[dtype] += 1
        bit_depth_counter[bit_depth] += 1
        ndim_counter[ndim] += 1
        extension_counter[extension] += 1
        modality_counter[modality] += 1
        orientation_counter[orientation] += 1
        if slice_count is not None:
            slice_counter[slice_count] += 1
        if modality == "seg":
            seg_paths.append(file_path)

    case_rows = []
    complete_cases = 0
    incomplete_cases = 0
    missing_patterns = Counter()

    for case_id, modalities in sorted(case_to_modalities.items()):
        missing = sorted(EXPECTED_MODALITIES - modalities)
        has_complete_modalities = EXPECTED_MODALITIES.issubset(modalities)
        if has_complete_modalities:
            complete_cases += 1
        else:
            incomplete_cases += 1
            missing_patterns[", ".join(missing) if missing else "none"] += 1

        case_rows.append(
            {
                "case_id": case_id,
                "modalities_found": ", ".join(sorted(modalities)),
                "file_count": len(case_to_files[case_id]),
                "is_complete_5file_case": has_complete_modalities,
                "missing_modalities": ", ".join(missing) if missing else "",
            }
        )

    seg_label_summary = None
    inspected_seg_case_count = 0
    if inspect_seg_labels and seg_paths:
        seg_label_summary, inspected_seg_case_count = inspect_segmentation_labels(seg_paths, max_cases=max_seg_cases)

    summary = {
        "dataset_root": str(dataset_path),
        "total_files_analyzed": len(file_rows),
        "total_unique_cases": len(case_rows),
        "extension_counts": dict(extension_counter),
        "modality_counts": dict(modality_counter),
        "all_files_are_3d_volumes": set(ndim_counter.keys()) == {3},
        "ndim_distribution": dict(ndim_counter),
        "most_common_shape": " x ".join(map(str, shape_counter.most_common(1)[0][0])),
        "top_5_shapes": [
            {"shape": " x ".join(map(str, shape)), "count": count}
            for shape, count in shape_counter.most_common(5)
        ],
        "most_common_spacing_mm": " x ".join(map(str, spacing_counter.most_common(1)[0][0])),
        "top_5_spacings_mm": [
            {"spacing_mm": " x ".join(map(str, spacing)), "count": count}
            for spacing, count in spacing_counter.most_common(5)
        ],
        "most_common_slice_count_axis2": slice_counter.most_common(1)[0][0] if slice_counter else None,
        "slice_count_distribution_top_5": [
            {"slice_count": slice_count, "count": count}
            for slice_count, count in slice_counter.most_common(5)
        ],
        "most_common_dtype": dtype_counter.most_common(1)[0][0],
        "dtype_distribution": dict(dtype_counter),
        "most_common_bit_depth": bit_depth_counter.most_common(1)[0][0],
        "bit_depth_distribution": dict(bit_depth_counter),
        "orientation_distribution": dict(orientation_counter),
        "complete_5file_cases": complete_cases,
        "incomplete_cases": incomplete_cases,
        "missing_modality_patterns": dict(missing_patterns),
        "segmentation_label_summary": seg_label_summary,
        "segmentation_cases_inspected_for_labels": inspected_seg_case_count,
    }

    return summary, pd.DataFrame(file_rows), pd.DataFrame(case_rows)


def write_outputs(summary, files_df, cases_df, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json_path = output_dir / "summary.json"
    summary_txt_path = output_dir / "summary.txt"
    files_csv_path = output_dir / "per_file_analysis.csv"
    cases_csv_path = output_dir / "per_case_analysis.csv"

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    with open(summary_txt_path, "w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    files_df.to_csv(files_csv_path, index=False, encoding="utf-8-sig")
    cases_df.to_csv(cases_csv_path, index=False, encoding="utf-8-sig")

    return summary_json_path, summary_txt_path, files_csv_path, cases_csv_path


def main():
    parser = argparse.ArgumentParser(description="Analyze ASNR-MICCAI-BraTS2023-GLI dataset structure and metadata.")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset root folder.")
    parser.add_argument(
        "--output_dir",
        default="brats2023_gli_analysis_outputs",
        help="Directory where CSV/JSON/TXT outputs will be saved.",
    )
    parser.add_argument(
        "--inspect_seg_labels",
        action="store_true",
        help="Inspect segmentation label values by reading segmentation volumes.",
    )
    parser.add_argument(
        "--max_seg_cases",
        type=int,
        default=None,
        help="Optional limit for how many segmentation cases to inspect for unique label values.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)

    summary, files_df, cases_df = build_report(
        dataset_path=dataset_path,
        inspect_seg_labels=args.inspect_seg_labels,
        max_seg_cases=args.max_seg_cases,
    )
    paths = write_outputs(summary, files_df, cases_df, output_dir)

    print("\n=== DATASET SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\n=== OUTPUT FILES ===")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
