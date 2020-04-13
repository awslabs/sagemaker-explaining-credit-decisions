import argparse
import numpy as np
import os
from pathlib import Path
import json
import joblib
from sklearn.pipeline import Pipeline
# from xgboost import XGBClassifier
# from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

import sys

from package import preprocessing
from package import schemas
from package import datasets
from package import training


# TRAINING FUNCTIONS

# Note: this file in run in 'script mode' when `.fit` is called from the
# notebook. `parse_args` and `train_fn` are called in the
# `if __name__ =='__main__'` block.


def parse_args(sys_args):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tree-max-depth",
        type=int,
        default=10
    )
    parser.add_argument(
        "--tree-num-leaves",
        type=int,
        default=31
    )
    parser.add_argument(
        "--tree-boosting-type",
        type=str,
        default="gbdt"
    )
    parser.add_argument(
        "--tree-min-child-samples",
        type=int,
        default=20
    )
    parser.add_argument(
        "--tree-n-estimators",
        type=int,
        default=100
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=os.environ.get("SM_MODEL_DIR")
    )
    parser.add_argument(
        "--schemas",
        type=str,
        default=os.environ.get("SM_CHANNEL_SCHEMAS")
    )
    parser.add_argument(
        "--data-train",
        type=str,
        default=os.environ.get("SM_CHANNEL_DATA_TRAIN"),
    )
    parser.add_argument(
        "--label-train",
        type=str,
        default=os.environ.get("SM_CHANNEL_LABEL_TRAIN"),
    )
    parser.add_argument(
        "--data-test",
        type=str,
        default=os.environ.get("SM_CHANNEL_DATA_TEST")
    )
    parser.add_argument(
        "--label-test",
        type=str,
        default=os.environ.get("SM_CHANNEL_LABEL_TEST"),
    )

    args, _ = parser.parse_known_args(sys_args)
    return args


def load_schemas(schemas_folder):
    data_schema_filepath = Path(schemas_folder, "data.schema.json")
    data_schema = schemas.from_json_schema(data_schema_filepath)
    label_schema_filepath = Path(schemas_folder, "label.schema.json")
    label_schema = schemas.from_json_schema(label_schema_filepath)
    return data_schema, label_schema


def train_fn(args):
    # # load data
    data_schema, label_schema = load_schemas(args.schemas)
    X_train = datasets.read_csv_dataset(args.data_train, data_schema)
    y_train = datasets.read_csv_dataset(args.label_train, label_schema)
    X_test = datasets.read_csv_dataset(args.data_test, data_schema)
    y_test = datasets.read_csv_dataset(args.label_test, label_schema)

    # convert from column vector to 1d array
    y_train = y_train[:, 0]
    y_test = y_test[:, 0]

    # create components
    preprocessor = preprocessing.create_preprocessor(data_schema)
    classifier = LGBMClassifier(
        max_depth=args.tree_max_depth,
        num_leaves=args.tree_num_leaves,
        boosting_type=args.tree_boosting_type,
        min_child_samples=args.tree_min_child_samples,
        n_estimators=args.tree_n_estimators
    )

    # create pipeline
    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("classifier", classifier)]
    )
    training.train_pipeline(pipeline, X_train, y_train, args.cv_splits)
    features_schema = preprocessing.transform_schema(preprocessor, data_schema)
    training.test_pipeline(pipeline, X_test, y_test)

    # save components
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(preprocessor, Path(model_dir, "preprocessor.joblib"))
    joblib.dump(classifier, Path(model_dir, "classifier.joblib"))
    data_schema.save(Path(model_dir, "data.schema.json"))
    features_schema.save(Path(model_dir, "features.schema.json"))


# DEPLOYMENT FUNCTIONS
# these functions executed when a request is received by the endpoint
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


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
