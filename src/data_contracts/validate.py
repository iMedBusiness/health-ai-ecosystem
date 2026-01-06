import pandas as pd
from .specs import DATASET_SPECS


def validate_df(df: pd.DataFrame, dataset_name: str) -> None:
    if dataset_name not in DATASET_SPECS:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    required_cols = set(DATASET_SPECS[dataset_name])
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(
            f"{dataset_name} missing columns: {missing}"
        )

    if df.empty:
        raise ValueError(f"{dataset_name} is empty")

    # soft checks
    if "quantity_on_hand" in df.columns:
        if (df["quantity_on_hand"] < 0).any():
            print(f"⚠ Warning: negative stock detected in {dataset_name}")
