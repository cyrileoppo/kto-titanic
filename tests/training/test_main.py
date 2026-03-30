from unittest.mock import patch
from titanic.training.main import workflow


def test_workflow_runs_all_steps():
    with (
        patch("titanic.training.main.load_data") as mock_load_data,
        patch("titanic.training.main.split_train_test") as mock_split_train_test,
        patch("titanic.training.main.train") as mock_train,
        patch("titanic.training.main.validate") as mock_validate,
    ):
        mock_load_data.return_value = "path_output/data.csv"
        mock_split_train_test.return_value = [
            "xtrain/xtrain.csv",
            "xtest/xtest.csv",
            "ytrain/ytrain.csv",
            "ytest/ytest.csv",
        ]
        mock_train.return_value = "model_trained/model.joblib"

        workflow("data/all_titanic.csv", n_estimators=10, max_depth=3, random_state=42)

        mock_load_data.assert_called_once_with("data/all_titanic.csv")
        mock_split_train_test.assert_called_once_with("path_output/data.csv")
        mock_train.assert_called_once_with(
            "xtrain/xtrain.csv",
            "ytrain/ytrain.csv",
            10,
            3,
            42,
        )
        mock_validate.assert_called_once_with(
            "model_trained/model.joblib",
            "xtest/xtest.csv",
            "ytest/ytest.csv",
        )