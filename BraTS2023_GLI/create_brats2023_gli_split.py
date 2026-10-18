import argparse
import json
import random
from pathlib import Path


DEFAULT_MODALITIES = ["t1n", "t1c", "t2w", "t2f"]


def find_file(case_dir: Path, suffix: str):
    matches = sorted(case_dir.glob(f"*-{suffix}.nii.gz"))
    if len(matches) == 1:
        return matches[0]
    return None


def collect_complete_cases(dataset_path: Path, modalities):
    complete_cases = []
    incomplete_cases = []

    expected = list(modalities) + ["seg"]

    for case_dir in sorted(dataset_path.iterdir()):
        if not case_dir.is_dir():
            continue

        files = {}
        missing = []

        for suffix in expected:
            path = find_file(case_dir, suffix)
            if path is None:
                missing.append(suffix)
            else:
                files[suffix] = str(path)

        item = {
            "case_id": case_dir.name,
            "case_dir": str(case_dir),
            "files": files,
        }

        if missing:
            item["missing"] = missing
            incomplete_cases.append(item)
        else:
            complete_cases.append(item)

    return complete_cases, incomplete_cases


def main():
    parser = argparse.ArgumentParser(description="Create fixed train/validation split for BraTS2023-GLI.")
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--modalities", nargs="+", default=DEFAULT_MODALITIES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.20)
    parser.add_argument("--output_dir", type=str, default=r".\outputs\splits")
    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    complete_cases, incomplete_cases = collect_complete_cases(dataset_path, args.modalities)

    rng = random.Random(args.seed)
    shuffled = complete_cases[:]
    rng.shuffle(shuffled)

    n_val = int(round(len(shuffled) * args.val_ratio))
    val_cases = sorted(shuffled[:n_val], key=lambda x: x["case_id"])
    train_cases = sorted(shuffled[n_val:], key=lambda x: x["case_id"])

    split = {
        "dataset_name": "BraTS2023-GLI",
        "dataset_path": str(dataset_path),
        "modalities": args.modalities,
        "label_file": "seg",
        "label_mode": "whole_tumor_binary_seg_gt_0",
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

    json_path = output_dir / f"brats2023_gli_split_seed{args.seed}.json"
    train_txt = output_dir / f"brats2023_gli_train_cases_seed{args.seed}.txt"
    val_txt = output_dir / f"brats2023_gli_val_cases_seed{args.seed}.txt"

    json_path.write_text(json.dumps(split, indent=2), encoding="utf-8")
    train_txt.write_text("\n".join(c["case_id"] for c in train_cases), encoding="utf-8")
    val_txt.write_text("\n".join(c["case_id"] for c in val_cases), encoding="utf-8")

    print("=== BraTS2023-GLI Fixed Split Created ===")
    print(f"Dataset path: {dataset_path}")
    print(f"Modalities: {args.modalities}")
    print(f"Seed: {args.seed}")
    print(f"Complete labeled cases: {len(complete_cases)}")
    print(f"Incomplete cases: {len(incomplete_cases)}")
    print(f"Train cases: {len(train_cases)}")
    print(f"Validation cases: {len(val_cases)}")
    print(f"Saved JSON: {json_path.resolve()}")
    print(f"Saved train list: {train_txt.resolve()}")
    print(f"Saved val list: {val_txt.resolve()}")


if __name__ == "__main__":
    main()
