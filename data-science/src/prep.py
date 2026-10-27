# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import mlflow

FEATURE_COLUMNS = (
    "Segment",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
)
TARGET_COLUMN = "price"
NUMERIC_COLUMNS = FEATURE_COLUMNS[1:] + (TARGET_COLUMN,)


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw data")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")
    parser.add_argument(
        "--test_train_ratio",
        type=float,
        default=0.2,
        help="Test-train ratio",
    )
    return parser.parse_args()


def clean_data(data):
    """Clean and validate the raw vehicle-pricing dataset."""
    required_columns = set(FEATURE_COLUMNS) | {TARGET_COLUMN}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing_columns)}")

    cleaned_data = data[list(FEATURE_COLUMNS) + [TARGET_COLUMN]].copy()
    cleaned_data["Segment"] = (
        cleaned_data["Segment"].astype("string").str.strip().str.lower()
    )
    for column in NUMERIC_COLUMNS:
        cleaned_data[column] = pd.to_numeric(cleaned_data[column], errors="coerce")

    valid_rows = cleaned_data["Segment"].ne("") & cleaned_data.notna().all(axis=1)
    valid_rows &= cleaned_data[list(NUMERIC_COLUMNS)].gt(0).all(axis=1)
    cleaned_data = cleaned_data.loc[valid_rows].copy()
    if cleaned_data.empty:
        raise ValueError("No valid rows remain after data cleaning")

    cleaned_data["Segment"] = LabelEncoder().fit_transform(cleaned_data["Segment"])
    return cleaned_data


def main(args):
    """Clean, preprocess, split, and save datasets."""
    if not 0 < args.test_train_ratio < 1:
        raise ValueError("--test_train_ratio must be greater than 0 and less than 1")

    raw_data = pd.read_csv(args.raw_data)
    cleaned_data = clean_data(raw_data)
    train_data, test_data = train_test_split(
        cleaned_data, test_size=args.test_train_ratio, random_state=42
    )

    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path / "train.csv", index=False)
    test_data.to_csv(test_path / "test.csv", index=False)

    mlflow.log_metrics(
        {
            "raw_row_count": len(raw_data),
            "cleaned_row_count": len(cleaned_data),
            "train_row_count": len(train_data),
            "test_row_count": len(test_data),
        }
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")
    with mlflow.start_run():
        main(args)
