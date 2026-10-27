# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
from pathlib import Path

import json
import mlflow
import mlflow.sklearn


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser("register")
    parser.add_argument("--model_name", type=str, required=True, help="Registered model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model directory")
    parser.add_argument(
        "--model_info_output_path",
        type=str,
        required=True,
        help="Directory for model registration metadata",
    )
    return parser.parse_args()

def main(args):
    """Load, log, register, and describe the selected model."""
    print(f"Registering {args.model_name}")
    model_path = Path(args.model_path)
    if not model_path.is_dir():
        raise FileNotFoundError(f"Model directory was not found: {model_path}")
    metrics_path = model_path / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Model metrics were not found: {metrics_path}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    model = mlflow.sklearn.load_model(model_path)
    logged_model = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_price_regressor",
    )
    registered_model = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=args.model_name,
    )

    output_path = Path(args.model_info_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "model_info.json").write_text(
        json.dumps(
            {
                "model_name": registered_model.name,
                "model_version": registered_model.version,
                "model_uri": logged_model.model_uri,
                "metrics": metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")
    with mlflow.start_run():
        main(args)