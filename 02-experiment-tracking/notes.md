## What is ML-flow

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