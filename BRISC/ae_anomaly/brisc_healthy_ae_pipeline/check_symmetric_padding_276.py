"""Non-destructive 256 -> 276 -> 256 symmetric-padding sanity check.

Run without inputs for a synthetic test, or with --image PATH to verify a real
local image (never resizes or writes the input). This does NOT train a model.
"""
from __future__ import annotations

import argparse
import numpy as np

PATCH = 256
PAD = 10
PADDED = PATCH + 2 * PAD


def symmetric_pad(patch: np.ndarray) -> np.ndarray:
    assert patch.shape[:2] == (PATCH, PATCH), patch.shape
    pad_width = ((PAD, PAD), (PAD, PAD)) + ((0, 0),) * (patch.ndim - 2)
    result = np.pad(patch, pad_width, mode="symmetric")
    assert result.shape[:2] == (PADDED, PADDED)
    assert np.array_equal(result[PAD:-PAD, PAD:-PAD, ...], patch)
    return result


def unpad(reconstruction_276: np.ndarray) -> np.ndarray:
    assert reconstruction_276.shape[:2] == (PADDED, PADDED), reconstruction_276.shape
    return reconstruction_276[PAD:-PAD, PAD:-PAD, ...]


def split_and_reassemble_identity(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Replicates OLD edge geometry, only adds symmetric padding per patch.

    Model output is intentionally mocked by identity; verifies preprocessing,
    border handling and stitching, NOT reconstruction quality or seam removal.
    """
    height, width = image.shape[:2]
    pad_bottom = (-height) % PATCH
    pad_right = (-width) % PATCH
    outer_padding = ((0, pad_bottom), (0, pad_right)) + ((0, 0),) * (image.ndim - 2)
    canvas = np.pad(image, outer_padding, mode="edge")
    stitched = np.empty_like(canvas)
    count = 0
    for y in range(0, canvas.shape[0], PATCH):
        for x in range(0, canvas.shape[1], PATCH):
            original_patch = canvas[y:y + PATCH, x:x + PATCH, ...]
            model_input_276 = symmetric_pad(original_patch)
            # Once the real model has passed the shape check, replace this line
            # with its actual (276, 276) reconstruction output:
            mock_model_output_276 = model_input_276.copy()
            output_256 = unpad(mock_model_output_276)
            assert np.array_equal(output_256, original_patch)
            stitched[y:y + PATCH, x:x + PATCH, ...] = output_256
            count += 1
    restored = stitched[:height, :width, ...]
    assert restored.shape == image.shape
    assert np.array_equal(restored, image)
    return restored, count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Optional path to a local MRI image (loaded as grayscale)")
    args = parser.parse_args()

    if args.image:
        from PIL import Image
        with Image.open(args.image) as image_file:
            image = np.asarray(image_file.convert("L"), dtype=np.uint8)
    else:
        # 369x369 forces partial border patches, like the BRISC data.
        image = np.arange(369 * 369, dtype=np.uint32).reshape(369, 369) % 256
        image = image.astype(np.uint8)

    restored, count = split_and_reassemble_identity(image)
    print(f"Original image: {image.shape}, dtype: {image.dtype}")
    print(f"Fixed tile: {PATCH}x{PATCH}, symmetric pad: {PAD}px per side")
    print(f"Autoencoder expected input/output: {PADDED}x{PADDED}")
    print(f"Patch count: {count}; exact round-trip: {np.array_equal(image, restored)}")
    print("PASS: padding, unpadding, old edge geometry, and reassembly preserve pixels")
    print("NEXT: check that the unchanged ResNet really outputs 276x276 for 276x276 input")


if __name__ == "__main__":
    main()
