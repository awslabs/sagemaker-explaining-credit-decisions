"""
DEPLOYMENT FUNCTIONS: these functions executed when a request is received
by the endpoint
"""
import numpy as np
from pathlib import Path
import json
import joblib

from package.data import schemas


def model_fn(model_dir):
    model_dir = Path(model_dir)
    preprocessor = joblib.load(Path(model_dir, "preprocessor.joblib"))
    classifier = joblib.load(Path(model_dir, "classifier.joblib"))
    data_schema = schemas.from_json_schema(
        Path(model_dir, "data.schema.json")
    )
    model_assets = {
        "data_schema": data_schema,
        "preprocessor": preprocessor,
        "classifier": classifier,
    }
    return model_assets


def input_fn(request_body_str, request_content_type):
    assert (
        request_content_type == "application/json"
    ), "content_type must be 'application/json'"
    request_body = json.loads(request_body_str)
    return request_body


def preprocess_fn(request_body, model_assets):
    model_assets["data_schema"].validate(request_body)
    data = model_assets["data_schema"].transform(request_body)
    data = np.expand_dims(data, axis=0)
    features = model_assets["preprocessor"].transform(data)
    return features


def predict_fn(request_body, model_assets):
    features = preprocess_fn(request_body, model_assets)
    prediction = model_assets["classifier"].predict_proba(features)
    # take first sample (idx=0)
    # and second probability (idx=1) corresponding to the positive class
    prediction = prediction[0][1]
    return {"prediction": prediction}


def output_fn(prediction, response_content_type):
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(prediction)
    return response_body_str
