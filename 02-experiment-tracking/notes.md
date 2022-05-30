## What is ML-flow?

MLflow is a python package whose main functionality is to organize, optimize and collect/log information about
our model's iterations to facilitate reproducibility.

## Getting started with MLflow

First, create a conda environment, so that the requirements config doest not mess up your local configuration.

```
conda create -n exp-tracking python=3.9
```

Next, install the required packages

```python 
pip install -r requirements.txt
``` 
Start the *MLflow* UI with the following command: 

```sh
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

We will be working on the notebook *duration-prediction.ipynb*, make sure to run your code on the environment *exp-tracking*. 

Import *MLflow* and set the tracking uri as well as the experiment name: 

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
mlflow.set_experiment("nyc_taxi_exp")
```

Run the notebook and for the last model, log it's information as follows: 

```python
with mlflow.start_run():
    mlflow.set_tag("Dev", "Sebastian")
    
    mlflow.log_param("train-data-path", "./data/green_tripdata_2021-01.parquet")
    mlflow.log_param("val-data-path", "./data/green_tripdata_2021-02.parquet")
    
    alpha = 0.01
    
    mlflow.log_param("alpha", alpha)
    lasso = Lasso(alpha)
    lasso.fit(X_train, y_train)

    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    mlflow.log_metric("rmse", rmse)
```

* `mlflow.set_tag()` allows to track information on who did a particular run.

* `mlflow.log_param()` tracks particular values of certain parameters (data path in this instance) to facilitate reproducibility.

* `mlflow.log_metric()` logs values of hyperparameters that the model takes as inputs.

The end result of the (simple) experiment tracking looks something like this:

![alt text](https://github.com/sebastian2296/mlops-zoomcamp/blob/main/02-experiment-tracking/img/mlflow_getting_started.png)


## Experiment tracking with MLflow

### Add hyperparameter tuning to the notebook 

We'll use Xgboost because it's much easier to have multiple runs for each hyperparameter.

We'll start by importing all the libraries needed, like so:

```python
import xgboost as xgb

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
```

*Hyperopt* is a library that uses bayesian methods to find the best set of hyperparameters.

* `fmin` method to minimize the output of the object function. 
* `tpe` algorithm that controls the above logic.
* `hp` contains different methods that set the search spaces for each hyperparameter.
* `STATUS_OK` signal that we'll sent to *hyperopt* to tell that the current run completed succesfully.
* `Trials` keeps track of information for each run.

We'll define a function that we'll pass to the `fmin` method in order to log all the results of running *xgboost* with different sets of hyperparameters. 

```python
def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("model", "xgboost")
        mlflow.log_params(params)
        booster = xgb.train(
            params=params,
            dtrain=train,
            num_boost_round=1000,
            evals=[(valid, 'validation')],
            early_stopping_rounds=50
        )
        y_pred = booster.predict(valid)
        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mlflow.log_metric("rmse", rmse)

    return {'loss': rmse, 'status': STATUS_OK}
```

Lastly, we'll iterate 50 times (50 different sets of hyperparemeters) to find the specfication that minimizes the `RMSE`:

* Define the search space (range of values each hyperparameter can take):

```python
search_space = {
    'max_depth': scope.int(hp.quniform('max_depth', 4, 100, 1)),
    'learning_rate': hp.loguniform('learning_rate', -3, 0),
    'reg_alpha': hp.loguniform('reg_alpha', -5, -1),
    'reg_lambda': hp.loguniform('reg_lambda', -6, -1),
    'min_child_weight': hp.loguniform('min_child_weight', -1, 3),
    'objective': 'reg:linear',
    'seed': 42
}
```
* Minimize the output value (`RMSE`):
 ```python
best_result = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=50,
    trials=Trials()
)
```
### How this looks in MLflow

A contour plot that shows different RMSE results for multiple hyperparameter sets (model runs):

![alt text](https://github.com/sebastian2296/mlops-zoomcamp/blob/main/02-experiment-tracking/img/mlflow_hyperparameter_tuning.png)

### Select the best model

There's no gold standard to choose the best model, it will depend on each particular case and what you're looking for in the model. 

In general, we can stick to pick the one with the lowest `RMSE`:

![alt text](https://github.com/sebastian2296/mlops-zoomcamp/blob/main/02-experiment-tracking/img/lowest_rmse.png)

The parameters that yielded the lowest `RMSE`:

![alt text](https://github.com/sebastian2296/mlops-zoomcamp/blob/main/02-experiment-tracking/img/params.png)

### Autolog

Let's use the "best params" of the previous iterations to showcase the autolog feature. 

The autolog feature allows us to track multiple characteristics of our models runs like *feature importance* , the dependecies and the model artifacts. This feature can be activated by adding: 

`mlflow.xgboost.autolog()` before the model train:

```python
params = {
    "earning_rate":0.24049559707914156,
    "max_depth":19,
    "min_child_weight":1.5402061054017244,
    "objective": "reg:linear",
    "reg_alpha": 0.2314863926934718,
    "reg_lambda":0.10273237284095527,
    "seed": 42
    }

#We can log results with the previous approach "with mlflow.start_run" but xgboost allows us to use autolog

mlflow.xgboost.autolog()

booster = xgb.train(
    params=params,
    dtrain=train,
    num_boost_round=1000,
    evals=[(valid, 'validation')],
    early_stopping_rounds=50
)
```