from __future__ import annotations

import argparse
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom


PATIENT_DIR_PATTERN = re.compile(r"UPENN-GBM-\d{5}$", re.IGNORECASE)
CATEGORY_PRIORITY = ["t1", "t1_gd", "t2", "t2_flair", "perfusion", "diffusion", "segmentation", "other"]
CATEGORY_LABELS = {
    "t1": "T1",
    "t1_gd": "T1-Gd / T1C",
    "t2": "T2",
    "t2_flair": "T2-FLAIR",
    "perfusion": "Perfusion",
    "diffusion": "Diffusion",
    "segmentation": "Segmentation",
    "other": "Other",
}


@dataclass
class SeriesCandidate:
    patient_id: str
    series_path: Path
    series_description: str
    protocol_name: str
    image_type: str
    category: str
    plane: str
    is_processed_captk: bool
    rows: Optional[int]
    columns: Optional[int]
    n_slices: int


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



def list_patient_dirs(dataset_path: Path) -> list[Path]:
    dataset_path = dataset_path.resolve()
    if PATIENT_DIR_PATTERN.search(dataset_path.name):
        return [dataset_path]

    patient_dirs = sorted(
        p for p in dataset_path.iterdir()
        if p.is_dir() and PATIENT_DIR_PATTERN.search(p.name)
    )
    if patient_dirs:
        return patient_dirs

    if any(p.is_dir() for p in dataset_path.iterdir()):
        return [dataset_path]
    return []



def find_series_dirs(patient_dir: Path) -> list[Path]:
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

    if any(token in text for token in ["t1gd", "t1-gd", "t1 gd", "t1c", "post contrast", "post-contrast", "postcontrast", "gad", "+c"]):
        return "t1_gd"

    if "t2" in text:
        return "t2"

    if "t1" in text:
        return "t1"

    return "other"



def is_processed(series_description: str, protocol_name: str, image_type: str) -> bool:
    text = normalize_text_for_matching(series_description, protocol_name, image_type)
    return ("processed_captk" in text) or text.startswith("derived") or ("derived" in text)


# ------------------------------
# Series loading
# ------------------------------


def sort_key_for_dataset(ds: pydicom.dataset.FileDataset) -> tuple:
    """
    Sorting priority:
    1. InstanceNumber
    2. z from ImagePositionPatient
    3. AcquisitionNumber
    """
    instance_number = getattr(ds, "InstanceNumber", None)
    try:
        instance_number = int(instance_number)
    except Exception:
        instance_number = 0

    z_position = 0.0
    try:
        ipp = getattr(ds, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            z_position = float(ipp[2])
    except Exception:
        pass

    acquisition_number = getattr(ds, "AcquisitionNumber", None)
    try:
        acquisition_number = int(acquisition_number)
    except Exception:
        acquisition_number = 0

    return (instance_number, z_position, acquisition_number)



def build_candidate(series_dir: Path, patient_dir: Path) -> Optional[SeriesCandidate]:
    dcm_files = dicom_files_in_dir(series_dir)
    if not dcm_files:
        return None

    try:
        ds = pydicom.dcmread(str(dcm_files[0]), stop_before_pixels=True, force=True)
    except Exception:
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

    return SeriesCandidate(
        patient_id=safe_str(getattr(ds, "PatientID", "")) or patient_dir.name,
        series_path=series_dir,
        series_description=series_description,
        protocol_name=protocol_name,
        image_type=image_type,
        category=categorize_series(series_description, protocol_name, sequence_name, image_type),
        plane=infer_plane_from_iop(getattr(ds, "ImageOrientationPatient", None)),
        is_processed_captk=is_processed(series_description, protocol_name, image_type),
        rows=getattr(ds, "Rows", None),
        columns=getattr(ds, "Columns", None),
        n_slices=len(dcm_files),
    )



def candidate_score(candidate: SeriesCandidate) -> tuple:
    """
    Higher score means a more visualization-friendly candidate.
    Structural sequences prefer processed CaPTk outputs if present.
    """
    structural = candidate.category in {"t1", "t1_gd", "t2", "t2_flair"}
    return (
        1 if structural and candidate.is_processed_captk else 0,
        1 if candidate.plane == "axial" else 0,
        candidate.n_slices,
        (candidate.rows or 0) * (candidate.columns or 0),
    )



def select_representative_candidates(candidates: list[SeriesCandidate], max_series: int) -> list[SeriesCandidate]:
    selected: list[SeriesCandidate] = []
    seen_categories: set[str] = set()

    # First pass: one best series per category.
    for category in CATEGORY_PRIORITY:
        same_category = [c for c in candidates if c.category == category]
        if not same_category:
            continue
        best = sorted(same_category, key=candidate_score, reverse=True)[0]
        selected.append(best)
        seen_categories.add(category)
        if len(selected) >= max_series:
            return selected[:max_series]

    # Second pass: fill remaining slots with best leftovers.
    leftovers = [c for c in candidates if c not in selected]
    leftovers = sorted(leftovers, key=candidate_score, reverse=True)
    for candidate in leftovers:
        selected.append(candidate)
        if len(selected) >= max_series:
            break

    return selected[:max_series]



def load_series_volume(series_dir: Path) -> tuple[np.ndarray, list[pydicom.dataset.FileDataset]]:
    datasets = []
    for dcm_path in dicom_files_in_dir(series_dir):
        ds = pydicom.dcmread(str(dcm_path), force=True)
        datasets.append(ds)

    datasets = sorted(datasets, key=sort_key_for_dataset)

    slices = []
    for ds in datasets:
        try:
            pixel = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise RuntimeError(
                f"Could not decode pixel data for {series_dir}. "
                f"You may need 'pylibjpeg', 'pylibjpeg-libjpeg', or 'gdcm'. Original error: {exc}"
            ) from exc

        slope = safe_float(getattr(ds, "RescaleSlope", None))
        intercept = safe_float(getattr(ds, "RescaleIntercept", None))
        if slope is not None:
            pixel = pixel * slope
        if intercept is not None:
            pixel = pixel + intercept

        if pixel.ndim == 2:
            slices.append(pixel)
        elif pixel.ndim == 3:
            # Handle multi-frame DICOM such as DICOM SEG.
            if len(datasets) == 1:
                return pixel, datasets
            for frame in pixel:
                slices.append(frame)
        else:
            raise RuntimeError(f"Unsupported pixel array shape: {pixel.shape}")

    volume = np.stack(slices, axis=0)
    return volume, datasets



def choose_slice_index(volume: np.ndarray, slice_mode: str) -> int:
    depth = volume.shape[0]
    if depth == 0:
        return 0

    if slice_mode == "middle":
        return depth // 2

    if slice_mode == "max_signal":
        # Choose the slice with the highest standard deviation as a simple proxy
        # for most informative content.
        scores = [float(np.nanstd(volume[idx])) for idx in range(depth)]
        return int(np.argmax(scores))

    raise ValueError(f"Unsupported slice_mode: {slice_mode}")



def robust_normalize(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)

    lo = float(np.percentile(finite, 1))
    hi = float(np.percentile(finite, 99))
    if hi <= lo:
        hi = float(np.max(finite))
        lo = float(np.min(finite))
    if hi <= lo:
        return np.zeros_like(image, dtype=np.float32)

    image = np.clip(image, lo, hi)
    image = (image - lo) / (hi - lo)
    return image



def resolve_patient_dir(dataset_path: Path, patient_id: Optional[str]) -> Path:
    patient_dirs = list_patient_dirs(dataset_path)
    if not patient_dirs:
        raise RuntimeError("No patient folders found.")

    if patient_id is None:
        return patient_dirs[0]

    for patient_dir in patient_dirs:
        if patient_dir.name.lower() == patient_id.lower():
            return patient_dir

    raise RuntimeError(f"Patient folder not found: {patient_id}")


# ------------------------------
# CLI
# ------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize representative series from UPENN-GBM raw DICOM dataset.")
    parser.add_argument("--dataset_path", required=True, help="Root dataset path or a single patient folder.")
    parser.add_argument("--patient_id", default=None, help="Example: UPENN-GBM-00001. If omitted, the first patient folder is used.")
    parser.add_argument("--slice_mode", choices=["middle", "max_signal"], default="middle")
    parser.add_argument("--max_series", type=int, default=5, help="Maximum number of series to show in the output figure.")
    parser.add_argument("--output_dir", default="outputs/reports", help="Folder where the visualization figure and CSV will be saved.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    patient_dir = resolve_patient_dir(dataset_path, args.patient_id)
    series_dirs = find_series_dirs(patient_dir)

    candidates = []
    for series_dir in series_dirs:
        candidate = build_candidate(series_dir, patient_dir)
        if candidate is not None:
            candidates.append(candidate)

    if not candidates:
        raise RuntimeError("No readable DICOM series found for visualization.")

    selected = select_representative_candidates(candidates, max_series=args.max_series)
    if not selected:
        raise RuntimeError("No representative series could be selected.")

    n = len(selected)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.5 * ncols, 5.2 * nrows))
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])
    axes = axes.flatten()

    selected_rows = []

    for ax, candidate in zip(axes, selected):
        try:
            volume, _ = load_series_volume(candidate.series_path)
            slice_index = choose_slice_index(volume, args.slice_mode)
            image = robust_normalize(volume[slice_index])
            ax.imshow(image, cmap="gray")
            ax.set_title(
                f"{candidate.patient_id} | {CATEGORY_LABELS.get(candidate.category, candidate.category)}\n"
                f"Slice {slice_index + 1}/{volume.shape[0]} | {candidate.plane}",
                fontsize=11,
            )
            subtitle = candidate.series_description or candidate.protocol_name or candidate.series_path.name
            ax.set_xlabel(subtitle[:90], fontsize=9)
            ax.axis("off")

            selected_rows.append(
                {
                    "patient_id": candidate.patient_id,
                    "category": candidate.category,
                    "category_label": CATEGORY_LABELS.get(candidate.category, candidate.category),
                    "plane": candidate.plane,
                    "series_path": str(candidate.series_path),
                    "series_description": candidate.series_description,
                    "protocol_name": candidate.protocol_name,
                    "image_type": candidate.image_type,
                    "is_processed_captk": candidate.is_processed_captk,
                    "rows": candidate.rows,
                    "columns": candidate.columns,
                    "n_slices": candidate.n_slices,
                    "selected_slice_index": int(slice_index),
                }
            )
        except Exception as exc:
            ax.text(0.5, 0.5, f"Visualization failed\n{candidate.series_path.name}\n{exc}", ha="center", va="center")
            ax.axis("off")

    for ax in axes[len(selected):]:
        ax.axis("off")

    patient_id_for_name = selected[0].patient_id if selected else patient_dir.name
    fig.suptitle(f"UPENN-GBM Representative Series | {patient_id_for_name}", fontsize=15)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    figure_path = output_dir / f"{patient_id_for_name}_representative_series.png"
    csv_path = output_dir / f"{patient_id_for_name}_representative_series.csv"

    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame(selected_rows).to_csv(csv_path, index=False, encoding="utf-8")

    print(f"Patient dir: {patient_dir}")
    print(f"Representative figure saved to: {figure_path.resolve()}")
    print(f"Representative CSV saved to: {csv_path.resolve()}")


if __name__ == "__main__":
    main()




