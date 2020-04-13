import shap
from pathlib import Path
import sys

import entry_point
from package import schemas


def parse_args(sys_args):
    return entry_point.parse_args(sys_args)


def train_fn(args):
    return entry_point.train_fn(args)


def model_fn(model_dir):
    model_dir = Path(model_dir)
    model_assets = entry_point.model_fn(model_dir)
    # explainer wraps the classifier
    explainer = shap.TreeExplainer(model_assets['classifier'])
    model_assets['explainer'] = explainer
    # used for explanation descriptions
    features_schema = schemas.from_json_schema(
        Path(model_dir, "features.schema.json")
    )
    model_assets["features_schema"] = features_schema
    return model_assets


def input_fn(request_body_str, request_content_type):
    return entry_point.input_fn(request_body_str, request_content_type)


def predict_fn(data, model_assets):
    features = entry_point.preprocess_fn(data, model_assets)
    # take second probability (idx=1) corresponding to the positive class
    # and first sample (idx=0)
    shap_values = model_assets["explainer"].shap_values(features)[1][0]
    # take second probability (idx=1) corresponding to the positive class
    expected_value = model_assets["explainer"].expected_value[1]
    prediction = entry_point.predict_fn(data, model_assets)
    prediction['explanation'] = {
        "shap_values": shap_values.tolist(),
        "expected_value": expected_value,
        "feature_values": features[0].tolist(),  # take first sample (idx=0)
        "feature_names": model_assets["features_schema"].item_titles,
        "feature_descriptions": model_assets["features_schema"].item_descriptions  # noqa
    }
    return prediction


def output_fn(prediction, response_content_type):
    return entry_point.output_fn(prediction, response_content_type)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
