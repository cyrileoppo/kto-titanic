import logging

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn import metrics

client = None


def validate(model_path: str, x_test_path: str, y_test_path: str) -> None:
    logging.warning(f"validate {model_path}")

    run_id = mlflow.active_run().info.run_id
    local_model_path = client.download_artifacts(run_id, model_path)
    local_x_test_path = client.download_artifacts(run_id, x_test_path)
    local_y_test_path = client.download_artifacts(run_id, y_test_path)

    model = joblib.load(local_model_path)

    x_test = pd.read_csv(local_x_test_path, index_col=False)
    y_test = pd.read_csv(local_y_test_path, index_col=False)

    x_test = pd.get_dummies(x_test)

    # Align columns with training (robustness)
    if hasattr(model, "feature_names_in_"):
        x_test = x_test.reindex(columns=model.feature_names_in_, fill_value=0)

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    predictions = model.predict(x_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    mae = metrics.mean_absolute_error(y_test, predictions)
    r2 = metrics.r2_score(y_test, predictions)
    medae = metrics.median_absolute_error(y_test, predictions)

    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("medae", medae)

    if hasattr(model, "feature_importances_"):
        feature_importance = dict(zip(x_test.columns.tolist(), model.feature_importances_.tolist(), strict=False))
        mlflow.log_dict(feature_importance, "feature_importance.json")

    signature = infer_signature(x_test, predictions)
    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        signature=signature,
        input_example=x_test.head(5),
    )
    mlflow.register_model(model_info.model_uri, "titanic-rf")

    logging.warning(f"Validation done. Predictions shape: {len(predictions)}")