import numpy as np
import pandas as pd


def feature_engineering(df: pd.DataFrame, time_col: str = "datetime") -> pd.DataFrame:
    """Feature engineering for train data"""
    # Remove nans
    df = df.dropna(subset=["target"])

    # CATEGORY
    cat_features = ["county", "is_business", "product_type", "is_consumption"]
    for col in cat_features:
        df[col] = df[col].astype("category")

    # left only cateories and datetime
    df = df[cat_features + ["datetime", "target"]]

    # TIME FEATURES
    min_time = df[time_col].min()
    date2range = {
        "month": [1, 12],
        "day": [1, 31],
        "dayofyear": [1, 366],
        "dayofweek": [0, 6],
        "hour": [0, 23],
    }

    # Year
    df["year"] = df[time_col].dt.year

    date_cols = ["month", "day", "dayofyear", "dayofweek", "hour", "delta"]
    for col in date_cols:
        # Get feature
        if col == "delta":
            df["date_delta"] = (df[time_col] - min_time).dt.days
        else:
            df[col] = getattr(df[time_col].dt, col)
            min_, max_ = date2range[col]
            angles = 2 * np.pi * (df[col] - min_) / (max_ - min_ + 1)
            df[col + "_sine"] = np.sin(angles).astype("float")
            df[col + "_cosine"] = np.cos(angles).astype("float")

    # Remove time col
    del df[time_col]

    return df
