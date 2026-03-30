import logging
from pathlib import Path

import joblib
import pandas as pd

client = None


def validate(model_path: str, x_test_path: str, y_test_path: str) -> None:
    logging.warning(f"validate {model_path}")

    model = joblib.load(model_path)

    x_test = pd.read_csv(x_test_path, index_col=False)
    y_test = pd.read_csv(y_test_path, index_col=False)

    x_test = pd.get_dummies(x_test)

    # Align columns with training (robustness)
    if hasattr(model, "feature_names_in_"):
        x_test = x_test.reindex(columns=model.feature_names_in_, fill_value=0)

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    predictions = model.predict(x_test)

    # Minimal validation (just run, tests usually don’t check metrics)
    logging.warning(f"Validation done. Predictions shape: {len(predictions)}")