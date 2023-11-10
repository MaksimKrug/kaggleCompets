import argparse
import json

import joblib
import numpy as np
import optuna

from src.data_processing.load_data import prepare_splits
from src.training.utils import get_metric, prepare_data

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--trials", type=int, default=10)
parser.add_argument("--timeout", type=int, default=60 * 60 * 1)
args = parser.parse_args()

# load configs
with open("configs/path_config.json", "r") as f:
    path_config = json.load(f)


def objective(trial, X, y, timesteps):
    """Objective function for optuna"""
    # Configs
    config = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 200),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.95, log=True),
        "depth": trial.suggest_int("depth", 3, 10, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-8, 100, log=True),
        "model_size_reg": trial.suggest_float("model_size_reg", 1e-8, 100, log=True),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.1, 1),
        "subsample": trial.suggest_float("subsample", 0.7, 1),
    }

    # Averaging by splits
    cv_mae = [None] * 3
    for idx, *data in prepare_splits(X, y, timesteps):
        cv_mae[idx] = get_metric(config, *data)

    return np.mean(cv_mae)


if __name__ == "__main__":
    # Preprocess data
    data, timesteps = prepare_data(args.data_path)
    X, y = data.drop(columns=["target"]), data["target"]
    # Get the best params
    study = optuna.create_study(directions=["minimize"], study_name="catboost")
    study.optimize(
        lambda trial: objective(trial, X, y, timesteps),
        n_trials=args.trials,
        timeout=args.timeout,
    )
    # dump artifacts
    _ = joblib.dump(study, path_config["catboost_hyperopt"])
    with open(path_config["catboost_best_params"], "w") as f:
        json.dump(study.best_params, f)
    with open(path_config["catboost_best_score"], "w") as f:
        json.dump({"MAE": study.best_value}, f)
    fig = optuna.visualization.plot_optimization_history(
        study, target_name="Validation MAE"
    )
    fig.write_image(path_config["catboost_optimization_history"])
