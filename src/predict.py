import json

import pandas as pd
from catboost import CatBoostRegressor

from src.data_processing.build_features import feature_engineering

# load configs
with open("configs/path_config.json", "r") as f:
    path_config = json.load(f)


def predict(
    df: pd.DataFrame,
    model_config: str = "catboost_model",
):
    """Very simple prediction pipeline"""
    # load data
    df = feature_engineering(df.copy())
    X, _ = df.drop(columns=["target"]), df["target"]
    # load model
    model = CatBoostRegressor()
    model.load_model(path_config[model_config])
    # predict
    pred = model.predict(X[:42])

    return pred
