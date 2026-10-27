# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, required=True, help="Training data directory")
    parser.add_argument("--test_data", type=str, required=True, help="Testing data directory")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum tree depth")
    parser.add_argument("--model_output", type=str, required=True, help="Model output directory")
    return parser.parse_args()


def main(args):
    """Read, train, evaluate, and save the model."""
    if args.n_estimators <= 0:
        raise ValueError("--n_estimators must be greater than 0")
    if args.max_depth <= 0:
        raise ValueError("--max_depth must be greater than 0")

    train_path = Path(args.train_data) / "train.csv"
    test_path = Path(args.test_data) / "test.csv"
    if not train_path.is_file():
        raise FileNotFoundError(f"Training dataset was not found: {train_path}")
    if not test_path.is_file():
        raise FileNotFoundError(f"Test dataset was not found: {test_path}")
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    target_column = "price"
    if target_column not in train_data or target_column not in test_data:
        raise ValueError(f"Both datasets must contain the '{target_column}' target column")
    if train_data.empty or test_data.empty:
        raise ValueError("Training and test datasets must both contain at least one row")

    x_train = train_data.drop(columns=target_column)
    y_train = train_data[target_column]
    x_test = test_data.drop(columns=target_column)
    y_test = test_data[target_column]
    if list(x_train.columns) != list(x_test.columns):
        raise ValueError("Training and test feature columns must match")
    if x_train.isna().any().any() or x_test.isna().any().any():
        raise ValueError("Training and test features must not contain missing values")

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    metrics = {
        "MSE": mse,
        "RMSE": math.sqrt(mse),
        "MAE": mean_absolute_error(y_test, predictions),
        "R2": r2_score(y_test, predictions),
    }

    mlflow.log_params(
        {"n_estimators": args.n_estimators, "max_depth": args.max_depth}
    )
    mlflow.log_metrics(metrics)

    model_path = Path(args.model_output)
    model_path.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, model_path)
    (model_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Train dataset input path: {args.train_data}")
    print(f"Test dataset input path: {args.test_data}")
    print(f"Model output path: {args.model_output}")
    print(f"Number of estimators: {args.n_estimators}")
    print(f"Max depth: {args.max_depth}")
    with mlflow.start_run():
        main(args)
