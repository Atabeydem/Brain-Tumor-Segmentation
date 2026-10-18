from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import pydicom


PATIENT_DIR_PATTERN = re.compile(r"UPENN-GBM-\d{5}$", re.IGNORECASE)


@dataclass
class SeriesRecord:
    patient_id: str
    patient_dir: str
    study_dir_name: str
    study_instance_uid: str
    series_dir_name: str
    series_instance_uid: str
    series_path: str
    n_dicom_files: int
    all_files_in_series_dir: int
    modality: str
    series_description: str
    protocol_name: str
    sequence_name: str
    image_type: str
    plane: str
    sequence_category: str
    is_processed_captk: bool
    rows: Optional[int]
    columns: Optional[int]
    matrix_size: str
    slice_count: int
    pixel_spacing_x: Optional[float]
    pixel_spacing_y: Optional[float]
    pixel_spacing_xy: str
    slice_thickness: Optional[float]
    spacing_between_slices: Optional[float]
    bits_allocated: Optional[int]
    bits_stored: Optional[int]
    high_bit: Optional[int]
    pixel_representation: Optional[int]
    dtype_guess: str
    photometric_interpretation: str
    samples_per_pixel: Optional[int]
    manufacturer: str
    manufacturer_model_name: str
    magnetic_field_strength: Optional[float]
    study_description: str
    accession_number: str
    body_part_examined: str
    transfer_syntax_uid: str
    representative_dicom: str


# ------------------------------
# Generic helpers
# ------------------------------


def safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()



def safe_float(value: object) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except Exception:
        return None



def format_float_tuple(values: list[Optional[float]], precision: int = 4) -> str:
    clean = []
    for value in values:
        if value is None:
            clean.append("NA")
        else:
            clean.append(f"{value:.{precision}f}")
    return " x ".join(clean)



def mode_or_unknown(values: Iterable[object], unknown: str = "Unknown") -> str:
    filtered = [str(v) for v in values if pd.notna(v) and str(v).strip()]
    if not filtered:
        return unknown
    return Counter(filtered).most_common(1)[0][0]



def numeric_mode(values: Iterable[object]) -> Optional[float]:
    filtered = []
    for value in values:
        try:
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                filtered.append(float(value))
        except Exception:
            pass
    if not filtered:
        return None
    return Counter(filtered).most_common(1)[0][0]



def list_patient_dirs(dataset_path: Path) -> list[Path]:
    """
    Return patient directories. If dataset_path itself is already a patient folder,
    return it directly.
    """
    dataset_path = dataset_path.resolve()

    if PATIENT_DIR_PATTERN.search(dataset_path.name):
        return [dataset_path]

    patient_dirs = sorted(
        p for p in dataset_path.iterdir()
        if p.is_dir() and PATIENT_DIR_PATTERN.search(p.name)
    )

    if patient_dirs:
        return patient_dirs

    # Fallback: if the user points directly to a folder that contains study UIDs,
    # analyze that as a single patient-like folder.
    if any(p.is_dir() for p in dataset_path.iterdir()):
        return [dataset_path]

    return []



def find_series_dirs(patient_dir: Path) -> list[Path]:
    """
    A series directory is any directory that directly contains .dcm files.
    """
    series_dirs: list[Path] = []
    for root, _, files in os.walk(patient_dir):
        if any(file.lower().endswith(".dcm") for file in files):
            series_dirs.append(Path(root))
    return sorted(series_dirs)



def dicom_files_in_dir(series_dir: Path) -> list[Path]:
    return sorted([p for p in series_dir.iterdir() if p.is_file() and p.suffix.lower() == ".dcm"])


# ------------------------------
# DICOM interpretation helpers
# ------------------------------


def infer_plane_from_iop(image_orientation_patient: object) -> str:
    """
    Infer acquisition plane from ImageOrientationPatient.
    The normal vector is computed as cross(row_dir, col_dir).
    Dominant axis of the normal determines the plane.
    """
    try:
        if image_orientation_patient is None:
            return "unknown"
        values = [float(x) for x in image_orientation_patient]
        if len(values) != 6:
            return "unknown"
        row = values[:3]
        col = values[3:]
        normal = [
            row[1] * col[2] - row[2] * col[1],
            row[2] * col[0] - row[0] * col[2],
            row[0] * col[1] - row[1] * col[0],
        ]
        axis = max(range(3), key=lambda idx: abs(normal[idx]))
        return {0: "sagittal", 1: "coronal", 2: "axial"}.get(axis, "unknown")
    except Exception:
        return "unknown"



def normalize_text_for_matching(*parts: object) -> str:
    text = " ".join(safe_str(part).lower() for part in parts if part is not None)
    return re.sub(r"\s+", " ", text).strip()



def categorize_series(series_description: str, protocol_name: str, sequence_name: str, image_type: str) -> str:
    text = normalize_text_for_matching(series_description, protocol_name, sequence_name, image_type)

    if any(token in text for token in ["seg", "label", "mask", "annotation"]):
        return "segmentation"

    if any(token in text for token in ["perfusion", "perf", "dsc", "cbv", "cbf", "psr", "ph", "rcbv"]):
        return "perfusion"

    if any(token in text for token in ["dti", "diff", "adc", "fa", "rd", "ad", "trace"]):
        return "diffusion"

    if "flair" in text or "t2_flair" in text or "tir" in text:
        return "t2_flair"

    # Put T1-Gd/T1C before plain T1.
    if any(token in text for token in ["t1gd", "t1-gd", "t1 gd", "t1c", "post contrast", "post-contrast", "postcontrast", "gad", "+c"]):
        return "t1_gd"

    if "t2" in text:
        return "t2"

    if "t1" in text:
        return "t1"

    return "other"



def dtype_guess_from_dicom(bits_allocated: object, pixel_representation: object) -> str:
    try:
        bits = int(bits_allocated)
    except Exception:
        return "unknown"

    try:
        signed = int(pixel_representation) == 1
    except Exception:
        signed = False

    prefix = "int" if signed else "uint"
    return f"{prefix}{bits}"



def bool_is_processed(series_description: str, protocol_name: str, image_type: str) -> bool:
    text = normalize_text_for_matching(series_description, protocol_name, image_type)
    return ("processed_captk" in text) or text.startswith("derived") or ("derived" in text)


# ------------------------------
# Core extraction
# ------------------------------


def build_series_record(series_dir: Path, patient_dir: Path) -> Optional[SeriesRecord]:
    dcm_files = dicom_files_in_dir(series_dir)
    if not dcm_files:
        return None

    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
    except Exception as exc:
        print(f"[WARN] Could not read DICOM header: {dcm_files[0]} -> {exc}")
        return None

    series_description = safe_str(getattr(ds, "SeriesDescription", ""))
    protocol_name = safe_str(getattr(ds, "ProtocolName", ""))
    sequence_name = safe_str(getattr(ds, "SequenceName", ""))
    image_type_values = getattr(ds, "ImageType", None)
    if image_type_values is None:
        image_type = ""
    else:
        try:
            image_type = "\\".join(str(x) for x in image_type_values)
        except Exception:
            image_type = safe_str(image_type_values)

    patient_id = safe_str(getattr(ds, "PatientID", "")) or patient_dir.name
    study_instance_uid = safe_str(getattr(ds, "StudyInstanceUID", "")) or series_dir.parent.name
    series_instance_uid = safe_str(getattr(ds, "SeriesInstanceUID", "")) or series_dir.name

    pixel_spacing = getattr(ds, "PixelSpacing", None)
    pixel_spacing_x = None
    pixel_spacing_y = None
    if pixel_spacing is not None:
        try:
            pixel_spacing_x = safe_float(pixel_spacing[0])
            pixel_spacing_y = safe_float(pixel_spacing[1])
        except Exception:
            pass

    rows = getattr(ds, "Rows", None)
    columns = getattr(ds, "Columns", None)
    matrix_size = f"{rows} x {columns}" if rows and columns else "NA"

    plane = infer_plane_from_iop(getattr(ds, "ImageOrientationPatient", None))
    sequence_category = categorize_series(series_description, protocol_name, sequence_name, image_type)
    is_processed_captk = bool_is_processed(series_description, protocol_name, image_type)

    transfer_syntax_uid = ""
    try:
        transfer_syntax_uid = safe_str(ds.file_meta.TransferSyntaxUID)
    except Exception:
        pass

    return SeriesRecord(
        patient_id=patient_id,
        patient_dir=patient_dir.name,
        study_dir_name=series_dir.parent.name,
        study_instance_uid=study_instance_uid,
        series_dir_name=series_dir.name,
        series_instance_uid=series_instance_uid,
        series_path=str(series_dir),
        n_dicom_files=len(dcm_files),
        all_files_in_series_dir=sum(1 for p in series_dir.iterdir() if p.is_file()),
        modality=safe_str(getattr(ds, "Modality", "")),
        series_description=series_description,
        protocol_name=protocol_name,
        sequence_name=sequence_name,
        image_type=image_type,
        plane=plane,
        sequence_category=sequence_category,
        is_processed_captk=is_processed_captk,
        rows=rows,
        columns=columns,
        matrix_size=matrix_size,
        slice_count=len(dcm_files),
        pixel_spacing_x=pixel_spacing_x,
        pixel_spacing_y=pixel_spacing_y,
        pixel_spacing_xy=format_float_tuple([pixel_spacing_x, pixel_spacing_y]),
        slice_thickness=safe_float(getattr(ds, "SliceThickness", None)),
        spacing_between_slices=safe_float(getattr(ds, "SpacingBetweenSlices", None)),
        bits_allocated=getattr(ds, "BitsAllocated", None),
        bits_stored=getattr(ds, "BitsStored", None),
        high_bit=getattr(ds, "HighBit", None),
        pixel_representation=getattr(ds, "PixelRepresentation", None),
        dtype_guess=dtype_guess_from_dicom(getattr(ds, "BitsAllocated", None), getattr(ds, "PixelRepresentation", None)),
        photometric_interpretation=safe_str(getattr(ds, "PhotometricInterpretation", "")),
        samples_per_pixel=getattr(ds, "SamplesPerPixel", None),
        manufacturer=safe_str(getattr(ds, "Manufacturer", "")),
        manufacturer_model_name=safe_str(getattr(ds, "ManufacturerModelName", "")),
        magnetic_field_strength=safe_float(getattr(ds, "MagneticFieldStrength", None)),
        study_description=safe_str(getattr(ds, "StudyDescription", "")),
        accession_number=safe_str(getattr(ds, "AccessionNumber", "")),
        body_part_examined=safe_str(getattr(ds, "BodyPartExamined", "")),
        transfer_syntax_uid=transfer_syntax_uid,
        representative_dicom=str(dcm_files[0]),
    )



def build_patient_summary(series_df: pd.DataFrame) -> pd.DataFrame:
    patient_rows = []
    for patient_id, group in series_df.groupby("patient_id"):
        categories = set(group["sequence_category"].dropna().tolist())
        patient_rows.append(
            {
                "patient_id": patient_id,
                "patient_dir": mode_or_unknown(group["patient_dir"]),
                "n_study_dirs": group["study_dir_name"].nunique(),
                "n_series_dirs": group["series_dir_name"].nunique(),
                "total_dicom_files": int(group["n_dicom_files"].sum()),
                "unique_sequence_categories": ", ".join(sorted(categories)),
                "has_t1": "t1" in categories,
                "has_t1_gd": "t1_gd" in categories,
                "has_t2": "t2" in categories,
                "has_t2_flair": "t2_flair" in categories,
                "has_perfusion": "perfusion" in categories,
                "has_diffusion": "diffusion" in categories,
                "has_segmentation": "segmentation" in categories,
                "has_complete_structural_set": all(
                    item in categories for item in ["t1", "t1_gd", "t2", "t2_flair"]
                ),
            }
        )
    return pd.DataFrame(patient_rows).sort_values("patient_id").reset_index(drop=True)



def build_dataset_summary(series_df: pd.DataFrame, patient_df: pd.DataFrame, dataset_path: Path) -> dict:
    summary: dict[str, object] = {}

    mr_df = series_df[series_df["modality"].fillna("").eq("MR")].copy()
    stats_df = mr_df if not mr_df.empty else series_df

    summary["dataset_path"] = str(dataset_path)
    summary["total_patient_folders_analyzed"] = int(patient_df.shape[0])
    summary["total_study_dirs"] = int(series_df["study_dir_name"].nunique())
    summary["total_series_dirs"] = int(series_df.shape[0])
    summary["total_dicom_files"] = int(series_df["n_dicom_files"].sum())
    summary["all_series_modality_mr"] = bool(series_df["modality"].fillna("").eq("MR").all())

    summary["most_common_matrix_size"] = mode_or_unknown(stats_df["matrix_size"])
    summary["most_common_slice_count"] = numeric_mode(stats_df["slice_count"])
    summary["most_common_in_plane_spacing"] = mode_or_unknown(stats_df["pixel_spacing_xy"])
    summary["most_common_slice_thickness"] = numeric_mode(stats_df["slice_thickness"])
    summary["most_common_spacing_between_slices"] = numeric_mode(stats_df["spacing_between_slices"])
    summary["most_common_plane"] = mode_or_unknown(stats_df["plane"])
    summary["most_common_dtype_guess"] = mode_or_unknown(stats_df["dtype_guess"])
    summary["most_common_bits_allocated"] = numeric_mode(stats_df["bits_allocated"])
    summary["most_common_bits_stored"] = numeric_mode(stats_df["bits_stored"])
    summary["most_common_photometric_interpretation"] = mode_or_unknown(stats_df["photometric_interpretation"])
    summary["most_common_manufacturer"] = mode_or_unknown(stats_df["manufacturer"])
    summary["most_common_manufacturer_model"] = mode_or_unknown(stats_df["manufacturer_model_name"])
    summary["most_common_magnetic_field_strength"] = numeric_mode(stats_df["magnetic_field_strength"])

    summary["processed_captk_series_count"] = int(series_df["is_processed_captk"].sum())
    summary["raw_or_non_processed_series_count"] = int((~series_df["is_processed_captk"]).sum())

    summary["modality_counts"] = series_df["modality"].replace("", "Unknown").value_counts(dropna=False).to_dict()
    summary["sequence_category_counts"] = series_df["sequence_category"].value_counts(dropna=False).to_dict()
    summary["plane_counts"] = series_df["plane"].value_counts(dropna=False).to_dict()
    summary["manufacturer_counts"] = series_df["manufacturer"].replace("", "Unknown").value_counts().head(10).to_dict()
    summary["top_series_descriptions"] = (
        series_df["series_description"]
        .replace("", "Unknown")
        .value_counts()
        .head(15)
        .to_dict()
    )
    summary["patients_with_complete_structural_set"] = int(patient_df["has_complete_structural_set"].sum())
    summary["patients_with_segmentation"] = int(patient_df["has_segmentation"].sum())

    return summary


def write_summary_text(summary: dict, out_path: Path) -> None:
    lines = [
        f"Dataset path: {summary['dataset_path']}",
        f"Total patient folders analyzed: {summary['total_patient_folders_analyzed']}",
        f"Total study dirs: {summary['total_study_dirs']}",
        f"Total series dirs: {summary['total_series_dirs']}",
        f"Total DICOM files: {summary['total_dicom_files']}",
        f"All series modality MR: {summary['all_series_modality_mr']}",
        f"Most common matrix size: {summary['most_common_matrix_size']}",
        f"Most common slice count: {summary['most_common_slice_count']}",
        f"Most common in-plane spacing: {summary['most_common_in_plane_spacing']}",
        f"Most common slice thickness: {summary['most_common_slice_thickness']}",
        f"Most common spacing between slices: {summary['most_common_spacing_between_slices']}",
        f"Most common plane: {summary['most_common_plane']}",
        f"Most common dtype guess: {summary['most_common_dtype_guess']}",
        f"Most common bits allocated: {summary['most_common_bits_allocated']}",
        f"Most common bits stored: {summary['most_common_bits_stored']}",
        f"Most common photometric interpretation: {summary['most_common_photometric_interpretation']}",
        f"Most common manufacturer: {summary['most_common_manufacturer']}",
        f"Most common manufacturer model: {summary['most_common_manufacturer_model']}",
        f"Most common magnetic field strength: {summary['most_common_magnetic_field_strength']}",
        f"Processed CaPTk series count: {summary['processed_captk_series_count']}",
        f"Raw/non-processed series count: {summary['raw_or_non_processed_series_count']}",
        f"Patients with complete structural set (T1, T1-Gd, T2, T2-FLAIR): {summary['patients_with_complete_structural_set']}",
        f"Patients with segmentation-like series detected: {summary['patients_with_segmentation']}",
        "",
        "Modality counts:",
    ]

    for key, value in summary["modality_counts"].items():
        lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("Sequence category counts:")
    for key, value in summary["sequence_category_counts"].items():
        lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("Plane counts:")
    for key, value in summary["plane_counts"].items():
        lines.append(f"  - {key}: {value}")

    lines.append("")
    lines.append("Top series descriptions:")
    for key, value in summary["top_series_descriptions"].items():
        lines.append(f"  - {key}: {value}")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# ------------------------------
# CLI
# ------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze UPENN-GBM raw DICOM dataset structure and metadata.")
    parser.add_argument("--dataset_path", required=True, help="Root folder of the dataset or a single patient folder.")
    parser.add_argument("--output_dir", default="outputs/reports", help="Folder where reports will be saved.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    patient_dirs = list_patient_dirs(dataset_path)
    if not patient_dirs:
        raise RuntimeError(
            "No patient folders found. Expected folders named like 'UPENN-GBM-00001' or a folder containing DICOM series."
        )

    records: list[SeriesRecord] = []
    for patient_dir in patient_dirs:
        series_dirs = find_series_dirs(patient_dir)
        for series_dir in series_dirs:
            record = build_series_record(series_dir, patient_dir)
            if record is not None:
                records.append(record)

    if not records:
        raise RuntimeError("No readable DICOM series were found.")

    series_df = pd.DataFrame([asdict(record) for record in records]).sort_values(
        ["patient_id", "study_dir_name", "series_dir_name"]
    ).reset_index(drop=True)

    patient_df = build_patient_summary(series_df)
    summary = build_dataset_summary(series_df, patient_df, dataset_path.resolve())

    series_csv = output_dir / "upenn_gbm_series_metadata.csv"
    patient_csv = output_dir / "upenn_gbm_patient_summary.csv"
    summary_txt = output_dir / "upenn_gbm_dataset_summary.txt"
    summary_json = output_dir / "upenn_gbm_dataset_summary.json"

    series_df.to_csv(series_csv, index=False, encoding="utf-8")
    patient_df.to_csv(patient_csv, index=False, encoding="utf-8")
    write_summary_text(summary, summary_txt)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Dataset path: {dataset_path}")
    print(f"Total patient folders analyzed: {summary['total_patient_folders_analyzed']}")
    print(f"Total study dirs: {summary['total_study_dirs']}")
    print(f"Total series dirs: {summary['total_series_dirs']}")
    print(f"Total DICOM files: {summary['total_dicom_files']}")
    print(f"All series modality MR: {summary['all_series_modality_mr']}")
    print(f"Most common matrix size: {summary['most_common_matrix_size']}")
    print(f"Most common slice count: {summary['most_common_slice_count']}")
    print(f"Most common in-plane spacing: {summary['most_common_in_plane_spacing']}")
    print(f"Most common slice thickness: {summary['most_common_slice_thickness']}")
    print(f"Most common plane: {summary['most_common_plane']}")
    print(f"Most common dtype guess: {summary['most_common_dtype_guess']}")
    print(f"Most common bits stored: {summary['most_common_bits_stored']}")
    print(f"Output folder: {output_dir.resolve()}")


if __name__ == "__main__":
    main()








