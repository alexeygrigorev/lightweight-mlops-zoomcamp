# Lightweight MLOps Zoomcamp

This is stripped-down version of
[MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) 


About the instructor:

* Founder of [DataTalks.Club](https://datatalks.club/) - community of 40k+ data enthusiasts
* Author of [ML Bookcamp](https://mlbookcamp.com/)
* Instructor of [ML Zoomcamp](http://mlzoomcamp.com/) and [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) 
* Connect with me on [LinkedIn](https://www.linkedin.com/in/agrigorev/) and [Twitter](https://twitter.com/Al_Grigor)
* Do you want to run a similar workshop at your company? Write me at alexey@datatalks.club


In this workshop, we will:

* Track experiments
* Create a training pipeline
* Register our model in the registry
* Serve the model
* Monitor the performance

In MLOps Zoomcamp we show how to use specific tools for achieving
this. But in this workshop we'll focus more on the concepts. 
We'll use one tool - ML flow, but the principles will apply to
any tool.


## Plan

* 5 min: Discuss what's MLOps and how it helps with the entire ML project lifecycle 
* 10 min: Prepare the environment and train our model (ride duration prediction)
* 10 min: Install and run MLFlow for experiment tracking
* 10 min: Use Scikit-Learn pipelines to make model management simpler
* 10 min: Convert a notebook for training a model to a Python script
* 15 min: Save and load the model with MLFlow model registry (and without)
* 15 min: Serve the model as a web service
* 10 min: Monitor the predictive performance of this model
* 5 min: Summary & wrapping up


## What's MLOps 

Poll: What's MLOps?

* https://datatalks.club/blog/mlops-10-minutes.html


## Preparation

* We'll start with the model [we already trained](train/duration-prediction-starter.ipynb)
* Copy this notebook to "duration-prediction.ipynb"
* This model is used for preducting the duration of a taxi trip

You can use any environment for running the content. In the workshop,
we rented an EC2 instance on AWS:

- Name: "mlops-workshop-2023"
- Ubuntu 22.04 64 bit
- Instance type: t2.xlarge
- 30 gb disk space
- Give it an IAM role with S3 read/write access
- We will need to forward ports 8888 and 5000


Script for preparing the instance:


```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b
rm Miniconda3-latest-Linux-x86_64.sh

# add it to .bashrc
export PATH="$HOME/miniconda3/bin:$PATH"

sudo apt install jq
```

We'll start with preparing the environement for the workshop

```bash
pip install pipenv 
```

Create the env in a separate folder (e.g. "train"):

```bash
pipenv --python=3.11
```

Install the dependencies

```bash
pipenv install scikit-learn==1.3.1 pandas pyarrow seaborn
pipenv install --dev jupyter
```

On Linux you might also need to instal `pexpect` for jupyter:

```bash
pipenv install --dev jupyter pexpect
```

Run poll: "Which virtual environment managers have you used"

Options:

- Conda
- Python venv
- Pipenv
- Poetry
- Other
- Didn't use any

We will use the data from the [NYC TLC website](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page):

* Train: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-01.parquet
* Validation: https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-02.parquet

Download the starter notebook:

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/lightweight-mlops-zoomcamp/main/train/duration-prediction-starter.ipynb
mv duration-prediction-starter.ipynb duration-prediction.ipynb
```

Run the notebook

```bash
pipenv run jupyter notebook
```

## Experiment tracking

First, let's add mlflow for tracking experiments 

```bash
pipenv install mlflow==2.7.1 boto3
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
import mlflow

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

or

```python
trip = {
    'PULocationID': '43',
    'DOLocationID': '238',
    'trip_distance': 1.16
}

model.predict(trip)
```

In some cases we don't want to depend on the MLFlow model registry
to be always available. In this case, we can get the S3 path
of the model and use it directly for initializing the model

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/lightweight-mlops-zoomcamp/main/train/storage_uri.py

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

Poll: "What can we use for serving an ML model?"

Now let's go to the `serve` folder and create a virtual 
environment

```bash
pipenv --python=3.11
pipenv install \
    scikit-learn==1.3.1 \
    mlflow==2.7.1 \
    boto3 \
    flask \
    gunicorn
```

Create a simple flask app (see [`serve.py`](serve/serve.py))

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/lightweight-mlops-zoomcamp/main/serve/serve.py
```

Run it:

```bash
echo ${MODEL_METADATA} | jq

export MODEL_VERSION=$(echo ${MODEL_METADATA} | jq -r ".run_id")
export MODEL_URI=$(echo ${MODEL_METADATA} | jq -r ".source")

pipenv run python serve.py
```

Test it:

```bash
REQUEST='{
    "PULocationID": 100,
    "DOLocationID": 102,
    "trip_distance": 30
}'
URL="http://localhost:9696/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL}
```

Now package the model with Docker and deploy it
(outside of the scope for this tutorial).


## Monitoring

Now let's add logging to our model. For that, we will save all the 
predictions somewhere. Later we will load these preditions and 
see if the features are drifting. 

First, we need to have a way to correlate the request and the response.
For that, each request needs to have an ID. 

We can change the request to look like that:

```json
{
    "ride_id": "ride_xyz",
    "ride": {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 30
    }
}
```

Let's change the `serve.py` file to handle this and also return 
the ID along with the predictions (see [`serve_v2.py`](serve/serve_v2.py))

```bash
wget https://raw.githubusercontent.com/alexeygrigorev/lightweight-mlops-zoomcamp/main/serve/serve_v2.py
```

New request: 

```bash
REQUEST='{
    "ride_id": "ride_xyz",
    "ride": {
        "PULocationID": 100,
        "DOLocationID": 102,
        "trip_distance": 30
    }
}'
URL="http://localhost:9696/predict"

curl -X POST \
    -d "${REQUEST}" \
    -H "Content-Type: application/json" \
    ${URL}
```


Now let's log the predictions - see the `log` function in `serve_v2.py`

Here we will use filesystem, but in practice you should never do it
and use tools like logstash, kafka, kinesis, mongo and so on.

A good approach could be writing results to Kinesis and then
dumping using Kinesis Firehose to save the results to S3
(see an example [here](https://github.com/alexeygrigorev/hands-on-mlops-workshop))

Now we start sending the requests and collect enough of them for a couple of days.

Let's analyze them (see the notebook in the [monitor](train/monitor.ipynb) notebook in the [train](train/) folder).

For that let's pretend we saved all the predictions, but in reality 
we'll just run them in our notebook

Let's use these datasets:

* https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-03.parquet
* https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2022-06.parquet
* https://d37ci6vzurychx.cloudfront.net/trip-data/green_tripdata_2020-06.parquet

After that, we can extract this notebook to a script, and if we
detect an issue, send an alert, re-train the model or do
some other actions.


## What's next

* Use Prefect/Airflow/etc for orchestration
* Use BentoML/KServe/Seldon for deployment
* Use Evidently/whylogs/Seldon for monitoring

If you want to learn how to do it -
check our [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course
