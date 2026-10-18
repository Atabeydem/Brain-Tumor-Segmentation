import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MODALITY_ORDER = ["t1n", "t1c", "t2w", "t2f", "seg"]


def detect_modality(file_path: Path) -> str:
    name = file_path.name.lower()
    for suffix in ["seg", "t1c", "t1n", "t2f", "t2w"]:
        if name.endswith(f"-{suffix}.nii.gz"):
            return suffix
    return "unknown"


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros_like(arr, dtype=np.float32)

    valid = arr[finite_mask]
    low, high = np.percentile(valid, [1, 99])
    if high <= low:
        low, high = valid.min(), valid.max()
        if high <= low:
            return np.zeros_like(arr, dtype=np.float32)

    arr = np.clip(arr, low, high)
    arr = (arr - low) / (high - low)
    return arr


def choose_slice_index(seg_volume: np.ndarray, mode: str = "max_tumor") -> int:
    depth = seg_volume.shape[2]
    if mode == "middle":
        return depth // 2

    tumor_pixels_per_slice = (seg_volume > 0).sum(axis=(0, 1))
    if tumor_pixels_per_slice.max() == 0:
        return depth // 2
    return int(np.argmax(tumor_pixels_per_slice))


def load_case_files(case_dir: Path):
    modality_to_path = {}
    for file_path in sorted(case_dir.glob("*.nii.gz")):
        modality = detect_modality(file_path)
        modality_to_path[modality] = file_path
    return modality_to_path


def load_volume(file_path: Path) -> np.ndarray:
    img = nib.load(str(file_path))
    return np.asarray(img.dataobj)


def get_case_dir(dataset_path: Path, case_id: str | None):
    if case_id:
        case_dir = dataset_path / case_id
        if not case_dir.exists():
            raise FileNotFoundError(f"Case folder not found: {case_dir}")
        return case_dir

    for candidate in sorted(dataset_path.iterdir()):
        if candidate.is_dir() and list(candidate.glob("*.nii.gz")):
            return candidate

    raise FileNotFoundError("No case folders with .nii.gz files were found.")


def save_modalities_figure(case_id, volumes, slice_index, output_path: Path):
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for ax, modality in zip(axes, MODALITY_ORDER):
        if modality not in volumes:
            ax.axis("off")
            ax.set_title(f"{modality}\nmissing")
            continue

        slice_img = volumes[modality][:, :, slice_index]
        if modality == "seg":
            ax.imshow(slice_img, cmap="gray")
        else:
            ax.imshow(normalize_for_display(slice_img), cmap="gray")
        ax.set_title(modality)
        ax.axis("off")

    fig.suptitle(f"{case_id} | Slice {slice_index}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_overlay_figure(case_id, base_volume, seg_volume, slice_index, output_path: Path, base_modality: str):
    base_slice = normalize_for_display(base_volume[:, :, slice_index])
    seg_slice = seg_volume[:, :, slice_index]
    tumor_mask = seg_slice > 0

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(base_slice, cmap="gray")
    axes[0].set_title(f"{base_modality}")
    axes[0].axis("off")

    axes[1].imshow(seg_slice, cmap="gray")
    axes[1].set_title("seg")
    axes[1].axis("off")

    axes[2].imshow(base_slice, cmap="gray")
    axes[2].imshow(np.ma.masked_where(~tumor_mask, tumor_mask), alpha=0.45, cmap="autumn")
    axes[2].set_title("overlay")
    axes[2].axis("off")

    fig.suptitle(f"{case_id} | Slice {slice_index}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize one BraTS 2023 GLI case.")
    parser.add_argument("--dataset_path", required=True, help="Path to dataset root folder.")
    parser.add_argument("--case_id", default=None, help="Optional case folder name, e.g. BraTS-GLI-00000-000")
    parser.add_argument("--slice_index", type=int, default=None, help="Optional fixed axial slice index.")
    parser.add_argument(
        "--slice_mode",
        choices=["max_tumor", "middle"],
        default="max_tumor",
        help="How to choose slice when --slice_index is not given.",
    )
    parser.add_argument(
        "--base_modality",
        choices=["t1n", "t1c", "t2w", "t2f"],
        default="t1c",
        help="Base modality for overlay image.",
    )
    parser.add_argument(
        "--output_dir",
        default="brats2023_gli_visualizations",
        help="Directory where PNG outputs will be saved.",
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_dir = get_case_dir(dataset_path, args.case_id)
    case_id = case_dir.name
    modality_to_path = load_case_files(case_dir)

    volumes = {}
    for modality, path in modality_to_path.items():
        volumes[modality] = load_volume(path)

    if "seg" in volumes:
        seg_volume = volumes["seg"]
        slice_index = args.slice_index if args.slice_index is not None else choose_slice_index(seg_volume, args.slice_mode)
    else:
        any_modality = next(iter(volumes.values()))
        slice_index = args.slice_index if args.slice_index is not None else any_modality.shape[2] // 2
        seg_volume = None

    modalities_png = output_dir / f"{case_id}_modalities_slice_{slice_index}.png"
    save_modalities_figure(case_id, volumes, slice_index, modalities_png)
    print(f"Saved: {modalities_png}")

    if seg_volume is not None and args.base_modality in volumes:
        overlay_png = output_dir / f"{case_id}_{args.base_modality}_overlay_slice_{slice_index}.png"
        save_overlay_figure(case_id, volumes[args.base_modality], seg_volume, slice_index, overlay_png, args.base_modality)
        print(f"Saved: {overlay_png}")


if __name__ == "__main__":
    main()
