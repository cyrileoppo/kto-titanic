import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

client = None

ARTIFACT_PATH = "model_trained"


def train(
    x_train_path: str,
    y_train_path: str,
    n_estimators: int,
    max_depth: int,
    random_state: int,
) -> str:
    logging.warning(f"train {x_train_path} {y_train_path}")

    x_train = pd.read_csv(x_train_path, index_col=False)
    y_train = pd.read_csv(y_train_path, index_col=False)

    x_train = pd.get_dummies(x_train)

    # ✅ Flatten y if needed
    if y_train.shape[1] == 1:
        y_train = y_train.iloc[:, 0]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    # ✅ Correct directory
    Path(ARTIFACT_PATH).mkdir(parents=True, exist_ok=True)

    model_path = Path(ARTIFACT_PATH, "model.joblib")
    joblib.dump(model, model_path)

    return str(model_path)