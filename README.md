# [Enefit - Predict Energy Behavior of Prosumers](https://www.kaggle.com/competitions/predict-energy-behavior-of-prosumers/overview)
Predict Prosumer Energy Patterns and Minimize Imbalance Costs.\

## Overview

The goal of the competition is to create an energy prediction model of prosumers to reduce energy imbalance costs.

This competition aims to tackle the issue of energy imbalance, a situation where the energy expected to be used doesn't line up with the actual energy used or produced. Prosumers, who both consume and generate energy, contribute a large part of the energy imbalance. Despite being only a small part of all consumers, their unpredictable energy use causes logistical and financial problems for the energy companies.

### Description

The number of prosumers is rapidly increasing, and solving the problems of energy imbalance and their rising costs is vital. If left unaddressed, this could lead to increased operational costs, potential grid instability, and inefficient use of energy resources. If this problem were effectively solved, it would significantly reduce the imbalance costs, improve the reliability of the grid, and make the integration of prosumers into the energy system more efficient and sustainable. Moreover, it could potentially incentivize more consumers to become prosumers, knowing that their energy behavior can be adequately managed, thus promoting renewable energy production and use.

### Dataset Description

Your challenge in this competition is to predict the amount of electricity produced and consumed by Estonian energy customers who have installed solar panels. You'll have access to weather data, the relevant energy prices, and records of the installed photovoltaic capacity.

This is a forecasting competition using the time series API. The private leaderboard will be determined using real data gathered after the submission period closes.


## Repository structure
------------
    ├── .dvc                   <- DVC utils
    │
    ├── configs
    │   └── path_config.json   <- Configs for paths
    │ 
    ├── data
    │   └── *.dvc              <- .dvc files for DVC
    │
    ├── notebooks
    │   └── EDA.ipynb          <- EDA notebook (saved outputs)
    │
    ├── src                
    │   ├── artifacts          <- All artifacts like images, .pkl files or models
    │   │
    │   ├── data_processing    <- Data processing scripts
    │   │    
    │   ├── logs               <- logs folder
    │   │
    │   ├── training           <- Python scripts for training
    │   │
    │   ├──  utils.py          <- Utils (logger only)
    |   |
    │   ├──  predict.py        <- Predict script (placeholder)
    |
    ├── tests                  <- Pytests (only two)
    |
    ├── .dvcignore
    ├── .gitignore
    ├── .pylintrc
    ├── Dockerfile
    ├── README.md
    ├── requirements.txt
    ├── setup.py   

--------

## Reproduce the pipeline
### Locally
install dependencies
```
pip install -r requirement.txt
```

download all .csv files from gdrive
```
dvc pull
```

optimizer pipeline
```
python -m src.training.hyperopt_optuna --data_path="./data/train.csv" --trials=3 --timeout=180
```

train model
```
python -m src.training.train_model --data_path="./data/train.csv" --study_path="./src/artifacts/catboost_hyperopt.pkl"
```
tests
```
pytest tests
```

### docker
run docker container
```
docker build --rm -t enefit .
docker run --rm -v $PWD/data:/workdir/data -it enefit 
```
<b>Run local commands inside docker container</b>

