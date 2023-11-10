import os
from pathlib import PosixPath
from warnings import filterwarnings

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

filterwarnings("ignore")


def load_dataset(data_path: PosixPath) -> pd.DataFrame:
    """Read csv file"""
    assert os.path.exists(data_path), "Data path not exists"
    df = pd.read_csv(data_path, parse_dates=["datetime"])

    return df


def get_timesteps(df: pd.DataFrame) -> pd.arrays.DatetimeArray:
    """Calculate timesteps for time splitting"""
    return np.sort(pd.to_datetime(df[["year", "month"]].assign(day=1)))


def prepare_splits(
    X: pd.DataFrame,
    y: pd.Series | pd.DataFrame,
    timesteps: pd.arrays.DatetimeArray,
    n_splits: int = 3,
) -> (pd.DataFrame, pd.Series):
    """
    Prepare splits for cross-validation

    Parameters
    ----------
    X : pd.DataFrame
        Features
    y : pd.Series | pd.DataFrame
        Target
    timestemps : pd.arrays.DatetimeArray
        Datetime array with unique year + month combinations
    n_splits : int
        Number of splits
    """
    # Split data
    first_dates_month = pd.to_datetime(X[["year", "month"]].assign(day=1))
    cv = TimeSeriesSplit(n_splits=n_splits)
    for idx, (train_index, test_index) in enumerate(cv.split(timesteps)):
        # split data by timesteps
        year_month_train, year_month_test = (
            timesteps[train_index],
            timesteps[test_index],
        )
        train_index = first_dates_month.isin(year_month_train)
        test_index = first_dates_month.isin(year_month_test)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        yield idx, X_train, X_test, y_train, y_test
