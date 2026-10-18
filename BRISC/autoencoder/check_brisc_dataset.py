from pathlib import Path
import pandas as pd

DEFAULT_DATASET_PATH = Path.home() / "Downloads" / "brisc_Dataset2025" / "brisc2025"


def to_bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        })
        .fillna(False)
        .astype(bool)
    )


def main():
    dataset_path = DEFAULT_DATASET_PATH

    print("Dataset path:", dataset_path)
    print("Exists:", dataset_path.exists())

    manifest_path = dataset_path / "manifest.csv"
    print("manifest.csv exists:", manifest_path.exists())

    if not manifest_path.exists():
        raise SystemExit("manifest.csv bulunamadı. Dataset path yanlış olabilir.")

    df = pd.read_csv(manifest_path)
    df["is_mask"] = to_bool_series(df["is_mask"])

    print("Columns:", list(df.columns))
    print("Total rows:", len(df))
    print("Splits:", df["split"].value_counts().to_dict())
    print("Tasks:", df["task"].value_counts().to_dict())
    print("Image rows:", int((~df["is_mask"]).sum()))
    print("Mask rows:", int(df["is_mask"].sum()))

    sample = df[~df["is_mask"]].iloc[0]
    sample_path = dataset_path / str(sample["relative_path"]).replace("\\", "/")

    print("Sample image:", sample_path)
    print("Sample exists:", sample_path.exists())


if __name__ == "__main__":
    main()