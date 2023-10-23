import os
import json

import time
from datetime import datetime
from pathlib import Path

import mlflow
from flask import Flask, request, jsonify


MODEL_VERSION = os.getenv('MODEL_VERSION')
MODEL_URI = os.getenv('MODEL_URI')

model = mlflow.pyfunc.load_model(MODEL_URI)


def prepare_features(ride):
    features = {}
    features['PULocationID'] = str(ride['PULocationID'])
    features['DOLocationID'] = str(ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features


def predict(features):
    preds = model.predict(features)
    return float(preds[0])


def log(result):
    # IMPORTANT:
    # Saving to files is a very bad practice. Don't do it
    # This is just an illustration - to make the workshop simpler
    # In practice, use logstash, kafka, kinesis, mongo, etc for that

    now = datetime.now()

    date_now = now.strftime('%Y-%m-%d')
    time_now = int(time.mktime(now.timetuple()))
    print(date_now)
    print(time_now)

    path = Path('logs') / date_now / f'{time_now}.json'
    print(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('wt', encoding='utf-8') as f_out:
        json.dump(result, f_out)



app = Flask('duration-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    body = request.get_json()
    ride = body['ride']
    ride_id = body['ride_id']

    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'preduction': {
            'duration': pred,
        },
        'ride_id': ride_id,
        'model_version': MODEL_VERSION
    }

    log(result)

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)