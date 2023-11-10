import argparse
import json
import os

import joblib
import pandas as pd

from src.training.utils import fit_model, prepare_data
from src.utils import logger

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str)
parser.add_argument("--study_path", type=str)
args = parser.parse_args()

# load configs
with open("configs/path_config.json", "r") as f:
    path_config = json.load(f)

if __name__ == "__main__":
    logger.info("START TRAINING")
    # load data
    logger.info("LOAD DATA")
    assert os.path.exists(args.data_path)
    data, timesteps = prepare_data(args.data_path)
    X, y = data.drop(columns=["target"]), data["target"]
    # load optuna study
    logger.info("LOAD OPTUNA CONFIGS")
    assert os.path.exists(args.study_path)
    study = joblib.load(args.study_path)
    # train model
    logger.info("FIT MODEL")
    model = fit_model(X, y, n_jobs=4, config=study.best_params, verbose=50)
    model.save_model(path_config["catboost_model"], format="cbm")
    # save feature importance
    feat_imp = pd.Series(model.feature_importances_, X.columns).sort_values(
        ascending=True
    )
    fig = feat_imp.plot(kind="barh").get_figure()
    fig.savefig(path_config["catboost_feature_importance"], bbox_inches="tight")
    # sanity check
    model.predict(X)
    logger.info("TRAINING FINISHED SUCCESSFULLY")
