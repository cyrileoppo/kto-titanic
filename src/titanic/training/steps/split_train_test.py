import logging
from pathlib import Path

import pandas as pd
import sklearn.model_selection

client = None

FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]
TARGET = "Survived"


def split_train_test(data_path: str) -> tuple[str, str, str, str]:
    logging.warning(f"split on {data_path}")

    df = pd.read_csv(data_path, index_col=False)

    y = df[TARGET]
    x = df[FEATURES]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    # ✅ Create directory
    Path("dist").mkdir(parents=True, exist_ok=True)

    datasets = [
        (x_train, "xtrain.csv"),
        (x_test, "xtest.csv"),
        (y_train, "ytrain.csv"),
        (y_test, "ytest.csv"),
    ]

    artifact_paths = []

    for data, filename in datasets:
        file_path = Path("dist", filename)
        data.to_csv(file_path, index=False)
        artifact_paths.append(str(file_path))  # ✅ return str

    return tuple(artifact_paths)