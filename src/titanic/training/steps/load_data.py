import logging
import os
from pathlib import Path
import boto3
import pandas as pd
from ydata_profiling import ProfileReport

ARTIFACT_PATH = "path_output"
PROFILING_PATH = "profiling_reports"

def load_data(path: str) -> str:
    logging.warning(f"load_data on path : {path}")

    Path("./dist/").mkdir(parents=True, exist_ok=True)
    local_path = Path("./dist/", "data.csv")

    # 👉 NEW: handle local files
    if Path(path).exists():
        df = pd.read_csv(path)
        df.to_csv(local_path, index=False)
    else:
        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        s3_client.download_file("kto-titanic", path, str(local_path))
        df = pd.read_csv(local_path)

    profile = ProfileReport(df, title=f"Profiling Report - {local_path.stem}")
    profile_path = Path("./dist/", "profile.html")
    profile.to_file(profile_path)

    return str(local_path) 
    # TODO : Dans un second temps, ajouter les logs mlflow, notamment les artifacts du profiling
    # Mais aussi logger l'artifact du fichier csv.