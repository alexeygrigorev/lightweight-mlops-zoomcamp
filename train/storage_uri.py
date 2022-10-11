import sys
import json
import argparse

from mlflow.tracking import MlflowClient


def find_stage(model, name='staging'):
    for version in model.latest_versions:
        if version.current_stage.lower() == name:
            return version
    return None


parser = argparse.ArgumentParser()

parser.add_argument('--tracking-uri', required=True)
parser.add_argument('--model-name', required=True)
parser.add_argument('--stage-name', required=True)


args = parser.parse_args()

tracking_uri = args.tracking_uri
model_name = args.model_name
stage_name = args.stage_name

client = MlflowClient(tracking_uri=tracking_uri)

model_metadata = client.get_registered_model(model_name)

model_version = find_stage(model_metadata, stage_name)

if model_version is not None:
    result = {
        'run_id': model_version.run_id,
        'source': model_version.source
    }
    print(json.dumps(result))
else:
    sys.exit(1)