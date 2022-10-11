# Lightweight MLOps Zoomcamp

This is stripped-down version of
[MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) 


In this workshop, we will:

* Track experiments
* Create a training pipeline
* Register our model in the 
* Serve the model
* Monitor the performance

In MLOps Zoomcamp we show how to use specific tools for achieving
this. But in this workshop we'll focus more on the concepts. 
We'll use one tool - ML flow, but the principles will apply to
any tool.



## What's MLOps 

* https://datatalks.club/blog/mlops-10-minutes.html


## Preparation

* We'll start with the model [we already trained](duration-prediction-starter.ipynb)
* Copy this notebook to "duration-prediction.ipynb"
* This model is used for preducting the duration of a taxi trip

We'll start with preparing the environement for the workshop

```bash
pip install pipenv 
```

Create the env:

```bash
pipenv --python=3.9
```

Install the dependencies

```bash
pipenv install scikit-learn==1.1.2 pandas pyarrow seaborn
pipenv install --dev jupyter
```

Download the data from the [NYC TLC website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

```bash
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
wget https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet
```

Run the notebook

```bash
pipenv run jupyter notebook
```

## Experiment tracking

First, let's add mlflow for tracking experiments 

```bash
pipenv install mlflow boto3
```

Run MLFlow locally (replace it with your bucket name)

```bash
pipenv run mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root s3://mlflow-models-alexey
```

Open it at http://localhost:5000/


Connect to the server from the notebook

```python
import mflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")
```

Log the experiment:

```python
with mlflow.start_run():
    categorical = ['PULocationID', 'DOLocationID']
    numerical = ['trip_distance']

    train_dicts = df_train[categorical + numerical].to_dict(orient='records')
    val_dicts = df_val[categorical + numerical].to_dict(orient='records')

    mlflow.log_params({
        'categorical': categorical,
        'numerical': numerical,
    })
    
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts)
    X_val = dv.transform(val_dicts)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_val)

    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(rmse)
    mlflow.log_metric('rmse', rmse)
    
    with open('dict_vectorizer.bin', 'wb') as f_out:
        pickle.dump(dv, f_out)
    mlflow.log_artifact('dict_vectorizer.bin')
    
    mlflow.sklearn.log_model(lr, 'model')
```

Replace it with a pipeline:


```python
from sklearn.pipeline import make_pipeline

pipeline = make_pipeline(
    DictVectorizer(),
    LinearRegression()
)

pipeline.fit(train_dicts, y_train)
y_pred = pipeline.predict(val_dicts)

mlflow.sklearn.log_model(pipeline, 'model')
```

## Training pipeline

Convert the notebook to a script 

```bash
pipenv run jupyter nbconvert --to=script duration-prediction.ipynb
```

Rename the file to `train.py` and clean it

Run it:

```bash 
pipenv run python train.py
```


## Model registry

Register the model as "trip_duration" model, stage "staging"

Let's get this model

```python
model = mlflow.pyfunc.load_model('models:/trip_duration/staging')
```

And use it:

```python
y_pred = model.predict(val_dicts)
```

In some cases we don't want to depend on the MLFlow model registry
to be always available. In this case, we can get the S3 path
of the model and use it directly for initializing the model

```bash
MODEL_METADATA=$(pipenv run python storage_uri.py \
    --tracking-uri http://localhost:5000 \
    --model-name trip_duration \
    --stage-name staging)
echo ${MODEL_METADATA}
```

Now we can use the storage URL to load the model:

```python
model = mlflow.pyfunc.load_model(storage_url)
y_pred = model.predict(val_dicts)
```

## Serving 

Now let's go to the `serve` folder and create a virtual 
environment

```bash
pipenv --python=3.9
pipenv install scikit-learn==1.1.2 mlflow==1.29.0 boto3 flask gunicorn
```

Create a simple flask app (see [`serve.py`](serve.py))






## Monitoring



## What's next

* Use Prefect/Airflow/etc for orchestration
* Use BentoML/KServe/Seldon for deployment
* Use Evidently for monitoring
