import argparse
import json
import random
from pathlib import Path
from datetime import datetime


DEFAULT_DATASET_PATH = Path(r"C:\Users\Atabey\Downloads\PKG - BraTS-PEDs-v1\BraTS-PEDs-v1")
DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def find_file(case_dir: Path, suffix: str) -> Path:
    """Find a modality or segmentation file inside a BraTS-PED case folder."""
    matches = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f"Multiple files found for suffix '{suffix}' in {case_dir}: {matches}")

    # Fallback for unusual naming
    matches = sorted(case_dir.glob(f"*{suffix}*.nii.gz"))
    if len(matches) == 1:
        return matches[0]

    return case_dir / f"{case_dir.name}-{suffix}.nii.gz"


def collect_complete_cases(dataset_path: Path, modalities, split_name: str = "Training", require_seg: bool = True):
    split_dir = dataset_path / split_name
    if not split_dir.exists():
        raise FileNotFoundError(f"Folder not found: {split_dir}")

    case_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    complete_cases = []
    incomplete_cases = []

    for case_dir in case_dirs:
        files = {mod: find_file(case_dir, mod) for mod in modalities}
        if require_seg:
            files["seg"] = find_file(case_dir, "seg")

        missing = [key for key, path in files.items() if not path.exists()]
        item = {
            "case_id": case_dir.name,
            "case_dir": str(case_dir),
            "files": {key: str(path) for key, path in files.items()},
        }

        if missing:
            item["missing"] = missing
            incomplete_cases.append(item)
        else:
            complete_cases.append(item)

    return complete_cases, incomplete_cases


def main():
    parser = argparse.ArgumentParser(description="Create a fixed BraTS-PED train/validation case split.")
    parser.add_argument("--dataset_path", type=str, default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output_dir", type=str, default=r".\outputs\splits")
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--require_seg", action="store_true", default=True)
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    complete_cases, incomplete_cases = collect_complete_cases(
        dataset_path=dataset_path,
        modalities=args.modalities,
        split_name="Training",
        require_seg=args.require_seg,
    )

    if len(complete_cases) == 0:
        raise RuntimeError("No complete labeled Training cases found.")

    rng = random.Random(args.seed)
    shuffled = list(complete_cases)
    rng.shuffle(shuffled)

    val_count = max(1, int(len(shuffled) * args.val_ratio))
    val_cases = sorted(shuffled[:val_count], key=lambda x: x["case_id"])
    train_cases = sorted(shuffled[val_count:], key=lambda x: x["case_id"])

    split = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset_path": str(dataset_path),
        "source_split": "Training",
        "task": "BraTS-PED binary tumor segmentation, seg > 0",
        "modalities": args.modalities,
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "n_complete_cases": len(complete_cases),
        "n_incomplete_cases": len(incomplete_cases),
        "n_train_cases": len(train_cases),
        "n_val_cases": len(val_cases),
        "train_cases": train_cases,
        "val_cases": val_cases,
        "incomplete_cases": incomplete_cases,
    }

    json_path = output_dir / f"brats_ped_split_seed{args.seed}.json"
    train_txt_path = output_dir / f"brats_ped_train_cases_seed{args.seed}.txt"
    val_txt_path = output_dir / f"brats_ped_val_cases_seed{args.seed}.txt"

    json_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    train_txt_path.write_text("\n".join([c["case_id"] for c in train_cases]) + "\n", encoding="utf-8")
    val_txt_path.write_text("\n".join([c["case_id"] for c in val_cases]) + "\n", encoding="utf-8")

    print("=== BraTS-PED Fixed Split Created ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Modalities: {args.modalities}")
    print(f"Seed: {args.seed}")
    print(f"Complete labeled cases: {len(complete_cases)}")
    print(f"Incomplete cases: {len(incomplete_cases)}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved train list: {train_txt_path.resolve()}")
    print(f"Saved val list: {val_txt_path.resolve()}")

    if incomplete_cases:
        print("\nWARNING: Some incomplete cases were found. They were excluded.")
        for item in incomplete_cases[:10]:
            print(f"- {item['case_id']}: missing {item['missing']}")


if __name__ == "__main__":
    main()
