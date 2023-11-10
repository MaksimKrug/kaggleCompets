from pathlib import PosixPath
from typing import Dict, Optional, Tuple

import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error

from src.data_processing.build_features import feature_engineering
from src.data_processing.load_data import get_timesteps, load_dataset


def prepare_data(data_path: PosixPath | str) -> (pd.DataFrame, pd.arrays.DatetimeArray):
    """Prepare data for training"""
    # load data
    df = load_dataset(data_path)
    df = feature_engineering(df.copy())
    # get timesteps
    timesteps = get_timesteps(df)

    return df, timesteps


def fit_model(
    X: pd.DataFrame,
    y: pd.Series,
    config: Optional[Dict] = None,
    n_jobs: int = 1,
    verbose: int = 0,
) -> Tuple[CatBoostRegressor, pd.Series]:
    """Training pipeline"""
    model = CatBoostRegressor(
        objective="MAE",
        thread_count=n_jobs,
        verbose=verbose,
        cat_features=["county", "is_business", "product_type", "is_consumption"],
    )
    # Set config if needed
    if config:
        model.set_params(**config)

    return model.fit(X, y)


def get_metric(
    config: Dict,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame | pd.Series,
    y_test: pd.DataFrame | pd.Series,
) -> float:
    """Fit model for one split"""
    # fit
    model = fit_model(X_train, y_train, config, n_jobs=4)

    # generate predictions
    y_pred = model.predict(X_test)

    return mean_absolute_error(y_test, y_pred)
