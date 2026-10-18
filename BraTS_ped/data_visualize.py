import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np


MODALITIES = ["t1n", "t1c", "t2w", "t2f", "seg"]


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize one BraTS case.")
    parser.add_argument(
        "--case_dir",
        type=str,
        required=True,
        help="Path to a single case folder, e.g. .../Training/BraTS-PED-00001-000",
    )
    parser.add_argument(
        "--slice_index",
        type=int,
        default=None,
        help="Optional manual slice index. If not given, the best tumor slice will be selected from seg.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional custom output image path. If not given, the figure will be saved under BraTS/outputs/figures.",
    )
    return parser.parse_args()


def load_nifti(file_path: Path):
    img = nib.load(str(file_path))
    data = np.asanyarray(img.dataobj)
    return data


def normalize_for_display(slice_2d):
    slice_2d = slice_2d.astype(np.float32)
    min_val = np.min(slice_2d)
    max_val = np.max(slice_2d)

    if max_val - min_val < 1e-8:
        return np.zeros_like(slice_2d)

    return (slice_2d - min_val) / (max_val - min_val)


def find_best_slice(seg_volume):
    areas = []
    for i in range(seg_volume.shape[2]):
        area = np.count_nonzero(seg_volume[:, :, i])
        areas.append(area)

    best_slice = int(np.argmax(areas))
    return best_slice


def main():
    args = parse_args()
    case_dir = Path(args.case_dir)

    script_dir = Path(__file__).resolve().parent
    if args.save_path is None:
        save_path = script_dir / "outputs" / "figures" / "sample_case_visualization.png"
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not case_dir.exists():
        raise FileNotFoundError(f"Case folder not found: {case_dir}")

    case_name = case_dir.name
    data_dict = {}

    for modality in MODALITIES:
        file_path = case_dir / f"{case_name}-{modality}.nii.gz"

        if not file_path.exists():
            if modality == "seg":
                data_dict[modality] = None
                continue
            raise FileNotFoundError(f"Missing file: {file_path}")

        data_dict[modality] = load_nifti(file_path)

    if args.slice_index is not None:
        slice_idx = args.slice_index
    else:
        if data_dict["seg"] is not None:
            slice_idx = find_best_slice(data_dict["seg"])
        else:
            slice_idx = data_dict["t1n"].shape[2] // 2

    print(f"Case: {case_name}")
    print(f"Selected slice index: {slice_idx}")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, modality in zip(axes, MODALITIES):
        volume = data_dict[modality]

        if volume is None:
            ax.set_title(f"{modality}\n(Not available)")
            ax.axis("off")
            continue

        slice_2d = volume[:, :, slice_idx]

        if modality == "seg":
            ax.imshow(slice_2d.T, cmap="gray", origin="lower")
        else:
            ax.imshow(normalize_for_display(slice_2d).T, cmap="gray", origin="lower")

        ax.set_title(modality)
        ax.axis("off")

    plt.suptitle(f"{case_name} | Slice {slice_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()