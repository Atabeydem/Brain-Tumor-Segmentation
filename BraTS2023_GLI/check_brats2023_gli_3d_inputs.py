import argparse
import json
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:
    raise ImportError(
        "nibabel is required for this script. Install it with: pip install nibabel"
    ) from exc


DEFAULT_SPLIT_JSON = Path(r".\outputs\splits\brats2023_gli_split_seed42.json")
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def load_nifti_info(path: Path):
    img = nib.load(str(path))
    data = img.get_fdata(dtype=np.float32)
    return {
        "shape": tuple(int(x) for x in data.shape),
        "zooms": tuple(float(x) for x in img.header.get_zooms()[:3]),
        "min": float(np.nanmin(data)),
        "max": float(np.nanmax(data)),
        "mean_nonzero": float(data[data != 0].mean()) if np.any(data != 0) else 0.0,
        "nonzero_voxels": int(np.count_nonzero(data)),
    }, data


def inspect_case(case_item, modalities):
    case_id = case_item["case_id"]
    files = {k: Path(v) for k, v in case_item["files"].items()}

    result = {
        "case_id": case_id,
        "ok": True,
        "errors": [],
        "modalities": {},
        "seg": {},
    }

    shapes = []
    zooms = []

    for mod in modalities:
        path = files.get(mod)
        if path is None or not path.exists():
            result["ok"] = False
            result["errors"].append(f"Missing modality: {mod}")
            continue

        try:
            info, _ = load_nifti_info(path)
            result["modalities"][mod] = info
            shapes.append(info["shape"])
            zooms.append(info["zooms"])
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"Could not load {mod}: {exc}")

    seg_path = files.get("seg")
    if seg_path is None or not seg_path.exists():
        result["ok"] = False
        result["errors"].append("Missing segmentation file.")
    else:
        try:
            seg_info, seg = load_nifti_info(seg_path)
            labels = np.unique(seg.astype(np.int16))
            tumor_voxels = int(np.count_nonzero(seg > 0))
            tumor_slices_z = int(np.count_nonzero(np.any(seg > 0, axis=(0, 1))))
            result["seg"] = {
                **seg_info,
                "labels": [int(x) for x in labels.tolist()],
                "tumor_voxels_seg_gt_0": tumor_voxels,
                "tumor_axial_slices": tumor_slices_z,
            }
            shapes.append(seg_info["shape"])
            zooms.append(seg_info["zooms"])
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"Could not load seg: {exc}")

    unique_shapes = sorted(set(shapes))
    unique_zooms = sorted(set(zooms))
    result["unique_shapes_in_case"] = unique_shapes
    result["unique_zooms_in_case"] = unique_zooms

    if len(unique_shapes) > 1:
        result["ok"] = False
        result["errors"].append(f"Shape mismatch in case: {unique_shapes}")

    return result


def summarize(results):
    ok_count = sum(1 for r in results if r["ok"])
    shape_counter = Counter()
    zoom_counter = Counter()
    label_counter = Counter()
    tumor_slices = []
    tumor_voxels = []

    for r in results:
        for shape in r.get("unique_shapes_in_case", []):
            shape_counter[str(shape)] += 1
        for zoom in r.get("unique_zooms_in_case", []):
            zoom_counter[str(zoom)] += 1
        labels = r.get("seg", {}).get("labels", [])
        label_counter[str(labels)] += 1
        if "tumor_axial_slices" in r.get("seg", {}):
            tumor_slices.append(r["seg"]["tumor_axial_slices"])
        if "tumor_voxels_seg_gt_0" in r.get("seg", {}):
            tumor_voxels.append(r["seg"]["tumor_voxels_seg_gt_0"])

    summary = {
        "n_cases": len(results),
        "n_ok_cases": ok_count,
        "n_problem_cases": len(results) - ok_count,
        "shape_counts": dict(shape_counter),
        "zoom_counts": dict(zoom_counter),
        "seg_label_pattern_counts": dict(label_counter),
    }

    if tumor_slices:
        arr = np.array(tumor_slices)
        summary["tumor_axial_slices"] = {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "total": int(arr.sum()),
        }

    if tumor_voxels:
        arr = np.array(tumor_voxels)
        summary["tumor_voxels_seg_gt_0"] = {
            "min": int(arr.min()),
            "max": int(arr.max()),
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "total": int(arr.sum()),
        }

    return summary


def main():
    parser = argparse.ArgumentParser(description="Check BraTS2023-GLI 3D NIfTI inputs for 3D training.")
    parser.add_argument("--split_json", type=str, default=str(DEFAULT_SPLIT_JSON))
    parser.add_argument("--output_dir", type=str, default=r".\outputs\3d_checks")
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--max_detailed_cases", type=int, default=5)
    args = parser.parse_args()

    split_json = Path(args.split_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not split_json.exists():
        raise FileNotFoundError(f"Split JSON not found: {split_json}")

    split = json.loads(split_json.read_text(encoding="utf-8"))
    train_cases = split["train_cases"]
    val_cases = split["val_cases"]

    print("=== BraTS2023-GLI 3D Input Check ===")
    print(f"Split JSON: {split_json.resolve()}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Modalities: {args.modalities}")

    all_results = {
        "split_json": str(split_json),
        "modalities": args.modalities,
        "train": [],
        "val": [],
    }

    for split_name, cases in [("train", train_cases), ("val", val_cases)]:
        print(f"\nInspecting {split_name} cases...")
        for idx, case_item in enumerate(cases, start=1):
            result = inspect_case(case_item, args.modalities)
            all_results[split_name].append(result)

            if idx <= args.max_detailed_cases:
                print(f"- {result['case_id']} | ok={result['ok']} | shapes={result.get('unique_shapes_in_case')} | labels={result.get('seg', {}).get('labels')} | tumor_slices={result.get('seg', {}).get('tumor_axial_slices')}")

            if not result["ok"]:
                print(f"  PROBLEM: {result['case_id']} -> {result['errors']}")

    all_results["summary"] = {
        "train": summarize(all_results["train"]),
        "val": summarize(all_results["val"]),
        "all": summarize(all_results["train"] + all_results["val"]),
    }

    out_json = output_dir / "brats2023_gli_3d_input_check.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print("\n=== Summary ===")
    for split_name in ["train", "val", "all"]:
        s = all_results["summary"][split_name]
        print(f"{split_name}: {s['n_ok_cases']}/{s['n_cases']} OK")
        print(f"  Shapes: {s['shape_counts']}")
        print(f"  Zooms: {s['zoom_counts']}")
        print(f"  Label patterns: {s['seg_label_pattern_counts']}")
        if "tumor_axial_slices" in s:
            ts = s["tumor_axial_slices"]
            print(f"  Tumor axial slices: min={ts['min']}, max={ts['max']}, mean={ts['mean']:.2f}, median={ts['median']:.2f}, total={ts['total']}")

    print(f"\nSaved detailed check: {out_json.resolve()}")


if __name__ == "__main__":
    main()

