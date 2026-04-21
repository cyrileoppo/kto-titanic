import logging
from pathlib import Path

import mlflow
import pandas as pd
import sklearn.model_selection

client = mlflow.MlflowClient()

FEATURES = ["Pclass", "Sex", "SibSp", "Parch"]
TARGET = "Survived"


def split_train_test(data_path: str) -> list[str]:
    logging.warning(f"split on {data_path}")

    run_id = mlflow.active_run().info.run_id
    local_data_path = client.download_artifacts(run_id, data_path)

    df = pd.read_csv(local_data_path, index_col=False)

    y = df[TARGET]
    x = df[FEATURES]

    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    datasets = [
        (x_train, "xtrain"),
        (x_test, "xtest"),
        (y_train, "ytrain"),
        (y_test, "ytest"),
    ]

    artifact_paths: list[str] = []

    for data, folder in datasets:
        Path(folder).mkdir(parents=True, exist_ok=True)

        file_path = Path(folder, f"{folder}.csv")
        data.to_csv(file_path, index=False)
        mlflow.log_artifact(str(file_path), artifact_path=folder)

        artifact_paths.append(str(file_path))

    return artifact_paths