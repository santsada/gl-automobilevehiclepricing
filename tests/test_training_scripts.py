import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

SOURCE_DIR = Path(__file__).parents[1] / "data-science" / "src"
sys.path.insert(0, str(SOURCE_DIR))

import prep  # noqa: E402
import train  # noqa: E402


class PreparationTests(unittest.TestCase):
    def test_clean_data_normalizes_categories_and_removes_invalid_rows(self):
        raw_data = pd.DataFrame(
            {
                "Segment": [" Luxury ", "non-luxury", "", "luxury"],
                "Kilometers_Driven": [10000, 20000, 30000, 0],
                "Mileage": [15.0, 18.0, 12.0, 14.0],
                "Engine": [2000, 1500, 1800, 2200],
                "Power": [150, 100, 120, 160],
                "Seats": [5, 5, 5, 5],
                "price": [20.0, 10.0, 12.0, 22.0],
            }
        )

        cleaned_data = prep.clean_data(raw_data)

        self.assertEqual(len(cleaned_data), 2)
        self.assertTrue(pd.api.types.is_integer_dtype(cleaned_data["Segment"]))


class TrainingTests(unittest.TestCase):
    def test_main_logs_evaluation_metrics_and_writes_metrics_file(self):
        train_data = pd.DataFrame(
            {
                "Segment": [0, 0, 1, 1],
                "Kilometers_Driven": [10000, 20000, 10000, 20000],
                "Mileage": [15.0, 18.0, 14.0, 16.0],
                "Engine": [2000, 1500, 2200, 1800],
                "Power": [150, 100, 160, 130],
                "Seats": [5, 5, 5, 7],
                "price": [20.0, 10.0, 25.0, 15.0],
            }
        )
        test_data = train_data.iloc[:2].copy()

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_directory = root / "train"
            test_directory = root / "test"
            model_directory = root / "model"
            train_directory.mkdir()
            test_directory.mkdir()
            train_data.to_csv(train_directory / "train.csv", index=False)
            test_data.to_csv(test_directory / "test.csv", index=False)
            args = SimpleNamespace(
                train_data=str(train_directory),
                test_data=str(test_directory),
                n_estimators=10,
                max_depth=3,
                model_output=str(model_directory),
            )

            with patch.object(train.mlflow, "log_params"), patch.object(
                train.mlflow, "log_metrics"
            ) as log_metrics, patch.object(train.mlflow.sklearn, "save_model"):
                train.main(args)

            metrics = json.loads((model_directory / "metrics.json").read_text())
            self.assertEqual(set(metrics), {"MSE", "RMSE", "MAE", "R2"})
            log_metrics.assert_called_once_with(metrics)
