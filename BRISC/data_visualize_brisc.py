import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "brisc_Dataset2025" / "brisc2025"


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize one BRISC sample.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=str(DEFAULT_DATASET_PATH),
        help=(
            "Path to the BRISC dataset root folder. "
            f"Default: {DEFAULT_DATASET_PATH}"
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default="segmentation",
        choices=["classification", "segmentation"],
        help="Task to visualize.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Split to search in.",
    )
    parser.add_argument(
        "--tumor_label",
        type=str,
        default="glioma",
        choices=["glioma", "meningioma", "pituitary", "no_tumor"],
        help="Tumor label to visualize.",
    )
    parser.add_argument(
        "--plane_label",
        type=str,
        default="axial",
        choices=["axial", "coronal", "sagittal"],
        help="Plane label to visualize.",
    )
    parser.add_argument(
        "--sample_rank",
        type=int,
        default=1,
        help="1-based index among matching samples.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Optional custom output image path.",
    )
    return parser.parse_args()


def normalize_for_display(image_array):
    image_array = image_array.astype(np.float32)
    min_val = np.min(image_array)
    max_val = np.max(image_array)

    if max_val - min_val < 1e-8:
        return np.zeros_like(image_array)

    return (image_array - min_val) / (max_val - min_val)


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_path)

    if args.save_path is None:
        save_path = Path(__file__).resolve().parent / "outputs" / "figures" / "sample_brisc_visualization.png"
    else:
        save_path = Path(args.save_path)

    save_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

    manifest_path = dataset_path / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.csv not found in: {dataset_path}")

    manifest_df = pd.read_csv(manifest_path)

    filtered = manifest_df[
        (manifest_df["task"] == args.task)
        & (manifest_df["split"] == args.split)
        & (manifest_df["tumor_label"] == args.tumor_label)
        & (manifest_df["plane_label"] == args.plane_label)
        & (~manifest_df["is_mask"].astype(bool))
    ].sort_values("index")

    if filtered.empty:
        raise ValueError("No sample matches the selected filters.")

    if args.sample_rank < 1 or args.sample_rank > len(filtered):
        raise ValueError(f"sample_rank must be between 1 and {len(filtered)}.")

    selected_row = filtered.iloc[args.sample_rank - 1]
    image_relative_path = str(selected_row["relative_path"]).replace("\\", "/")
    image_path = dataset_path / image_relative_path

    image = Image.open(image_path)
    image_array = np.asarray(image)

    mask_array = None

    if args.task == "segmentation":
        mask_candidates = manifest_df[
            (manifest_df["task"] == "segmentation")
            & (manifest_df["is_mask"].astype(bool))
            & (
                manifest_df["linked_image"].astype(str).str.replace("\\", "/", regex=False)
                == image_relative_path
            )
        ]

        if not mask_candidates.empty:
            mask_row = mask_candidates.iloc[0]
            mask_relative_path = str(mask_row["relative_path"]).replace("\\", "/")
            mask_path = dataset_path / mask_relative_path
            mask = Image.open(mask_path)
            mask_array = np.asarray(mask)

    print(f"Selected file: {selected_row['filename']}")
    print(f"Task: {selected_row['task']}")
    print(f"Split: {selected_row['split']}")
    print(f"Tumor label: {selected_row['tumor_label']}")
    print(f"Plane label: {selected_row['plane_label']}")

    if image_array.ndim == 3 and image_array.shape[2] == 3:
        display_image = normalize_for_display(image_array)
        is_rgb = True
    else:
        display_image = normalize_for_display(image_array)
        is_rgb = False

    if mask_array is not None:
        binary_mask = (mask_array > 127).astype(np.uint8)

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        if is_rgb:
            axes[0].imshow(display_image)
        else:
            axes[0].imshow(display_image, cmap="gray")
        axes[0].set_title("Image")
        axes[0].axis("off")

        axes[1].imshow(mask_array, cmap="gray")
        axes[1].set_title("Mask")
        axes[1].axis("off")

        if is_rgb:
            axes[2].imshow(display_image)
        else:
            axes[2].imshow(display_image, cmap="gray")
        axes[2].imshow(binary_mask, cmap="Reds", alpha=0.35)
        axes[2].set_title("Overlay")
        axes[2].axis("off")
    else:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        if is_rgb:
            ax.imshow(display_image)
        else:
            ax.imshow(display_image, cmap="gray")
        ax.set_title("Image")
        ax.axis("off")

    plt.suptitle(
        f"{selected_row['filename']} | {selected_row['tumor_label']} | {selected_row['plane_label']}",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Saved visualization to: {save_path}")


if __name__ == "__main__":
    main()
