r"""
Check BraTS-PEDs-v1 dataset structure.

Expected default Windows path:
    C:\Users\Atabey\Downloads\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1

Expected structure:
    BraTS-PEDs-v1/
        Training/
            BraTS-PED-00001-000/
                BraTS-PED-00001-000-t1n.nii.gz
                BraTS-PED-00001-000-t1c.nii.gz
                BraTS-PED-00001-000-t2w.nii.gz
                BraTS-PED-00001-000-t2f.nii.gz
                BraTS-PED-00001-000-seg.nii.gz
        Validation/
            BraTS-PED-00267-000/
                BraTS-PED-00267-000-t1n.nii.gz
                BraTS-PED-00267-000-t1c.nii.gz
                BraTS-PED-00267-000-t2w.nii.gz
                BraTS-PED-00267-000-t2f.nii.gz
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError as exc:
    raise SystemExit(
        "nibabel is required for reading .nii.gz files. Install it with:\n"
        "    pip install nibabel\n"
    ) from exc


DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "PKG - BraTS-PEDs-v1" / "BraTS-PEDs-v1"
MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def parse_args():
    parser = argparse.ArgumentParser(description="Check BraTS-PEDs-v1 dataset structure")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--full_scan",
        action="store_true",
        help="If set, reads every training segmentation mask to count tumor-containing slices.",
    )
    parser.add_argument(
        "--shape_check_cases",
        type=int,
        default=10,
        help="Number of cases per split to inspect for image shape/header. Use 0 to skip.",
    )
    return parser.parse_args()


def get_file(case_dir: Path, suffix: str) -> Path:
    return case_dir / f"{case_dir.name}-{suffix}.nii.gz"


def case_status(case_dir: Path, split: str) -> Dict:
    required = MODALITIES + (["seg"] if split.lower() == "training" else [])
    files = {suffix: get_file(case_dir, suffix) for suffix in required}
    missing = [suffix for suffix, path in files.items() if not path.exists()]
    return {
        "split": split,
        "case_id": case_dir.name,
        "case_path": str(case_dir),
        "missing": ",".join(missing),
        "is_complete": len(missing) == 0,
        **{f"has_{suffix}": files[suffix].exists() for suffix in required},
    }


def inspect_nifti(path: Path) -> Dict:
    img = nib.load(str(path))
    header = img.header
    zooms = header.get_zooms()[:3]
    return {
        "shape": tuple(int(v) for v in img.shape[:3]),
        "zooms": tuple(float(v) for v in zooms),
        "dtype": str(header.get_data_dtype()),
    }


def scan_training_seg(case_dir: Path) -> Dict:
    seg_path = get_file(case_dir, "seg")
    if not seg_path.exists():
        return {
            "case_id": case_dir.name,
            "seg_exists": False,
            "seg_shape": None,
            "labels": None,
            "tumor_voxels": None,
            "tumor_slices": None,
        }

    seg = np.asarray(nib.load(str(seg_path)).dataobj)
    labels = sorted([int(x) for x in np.unique(seg)])
    tumor_mask = seg > 0
    tumor_slices = np.where(tumor_mask.any(axis=(0, 1)))[0]
    return {
        "case_id": case_dir.name,
        "seg_exists": True,
        "seg_shape": tuple(int(v) for v in seg.shape[:3]),
        "labels": labels,
        "tumor_voxels": int(tumor_mask.sum()),
        "tumor_slices": int(len(tumor_slices)),
        "first_tumor_slice": int(tumor_slices[0]) if len(tumor_slices) else None,
        "last_tumor_slice": int(tumor_slices[-1]) if len(tumor_slices) else None,
    }


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "outputs" / "dataset_check"
    output_dir.mkdir(parents=True, exist_ok=True)

    training_dir = dataset_path / "Training"
    validation_dir = dataset_path / "Validation"

    print("=== BraTS-PED Dataset Check ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Dataset exists: {dataset_path.exists()}")
    print(f"Training exists: {training_dir.exists()}")
    print(f"Validation exists: {validation_dir.exists()}")

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    if not training_dir.exists():
        raise FileNotFoundError(f"Training folder not found: {training_dir}")
    if not validation_dir.exists():
        raise FileNotFoundError(f"Validation folder not found: {validation_dir}")

    train_cases = sorted([p for p in training_dir.iterdir() if p.is_dir()])
    val_cases = sorted([p for p in validation_dir.iterdir() if p.is_dir()])

    print(f"Training cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")

    rows: List[Dict] = []
    for case_dir in train_cases:
        rows.append(case_status(case_dir, "Training"))
    for case_dir in val_cases:
        rows.append(case_status(case_dir, "Validation"))

    case_df = pd.DataFrame(rows)
    complete_train = int(case_df[(case_df["split"] == "Training") & (case_df["is_complete"])].shape[0])
    complete_val = int(case_df[(case_df["split"] == "Validation") & (case_df["is_complete"])].shape[0])

    print(f"Complete training cases: {complete_train}/{len(train_cases)}")
    print(f"Complete validation cases: {complete_val}/{len(val_cases)}")

    shape_rows = []
    if args.shape_check_cases > 0:
        for split_name, cases in [("Training", train_cases), ("Validation", val_cases)]:
            for case_dir in cases[: args.shape_check_cases]:
                for suffix in MODALITIES + (["seg"] if split_name == "Training" else []):
                    path = get_file(case_dir, suffix)
                    if not path.exists():
                        continue
                    info = inspect_nifti(path)
                    shape_rows.append({
                        "split": split_name,
                        "case_id": case_dir.name,
                        "suffix": suffix,
                        "shape": str(info["shape"]),
                        "zooms": str(info["zooms"]),
                        "dtype": info["dtype"],
                    })
        shape_df = pd.DataFrame(shape_rows)
        if not shape_df.empty:
            print("\nSample inspected shapes:")
            print(shape_df.head(12).to_string(index=False))
            print("\nMost common shapes:")
            print(shape_df["shape"].value_counts().head().to_string())
    else:
        shape_df = pd.DataFrame()

    seg_df = pd.DataFrame()
    if args.full_scan:
        print("\nScanning all training segmentation masks for tumor-containing slices...")
        seg_rows = [scan_training_seg(case_dir) for case_dir in train_cases]
        seg_df = pd.DataFrame(seg_rows)
        print("Seg labels found:")
        label_sets = seg_df["labels"].astype(str).value_counts().head(10)
        print(label_sets.to_string())
        print("\nTumor-slice statistics:")
        print(seg_df["tumor_slices"].describe().to_string())
        print(f"Total tumor-containing axial slices: {int(seg_df['tumor_slices'].sum())}")

    summary = {
        "dataset_path": str(dataset_path),
        "training_cases": len(train_cases),
        "validation_cases": len(val_cases),
        "complete_training_cases": complete_train,
        "complete_validation_cases": complete_val,
        "modalities": MODALITIES,
        "training_has_seg": True,
        "validation_has_seg": False,
        "full_scan": bool(args.full_scan),
    }

    if args.full_scan and not seg_df.empty:
        summary.update({
            "total_tumor_slices": int(seg_df["tumor_slices"].sum()),
            "mean_tumor_slices_per_case": float(seg_df["tumor_slices"].mean()),
            "median_tumor_slices_per_case": float(seg_df["tumor_slices"].median()),
        })

    case_df.to_csv(output_dir / "case_completeness.csv", index=False)
    if not shape_df.empty:
        shape_df.to_csv(output_dir / "shape_check.csv", index=False)
    if not seg_df.empty:
        seg_df.to_csv(output_dir / "segmentation_slice_summary.csv", index=False)
    (output_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
