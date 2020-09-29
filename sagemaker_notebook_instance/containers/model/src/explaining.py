"""
DEPLOYMENT FUNCTIONS: these functions executed when a request is received
by the endpoint
"""
import numpy as np
from pathlib import Path
import json
import joblib
import shap
import warnings

from package.data import schemas


def model_fn(model_dir):
    model_dir = Path(model_dir)
    # load schemas
    data_schema_path = Path(model_dir, "data.schema.json")
    data_schema = schemas.from_json_schema(data_schema_path)
    features_schema_path = Path(model_dir, "features.schema.json")
    features_schema = schemas.from_json_schema(features_schema_path)
    # load preprocessor and classifier
    preprocessor = joblib.load(Path(model_dir, "preprocessor.joblib"))
    classifier = joblib.load(Path(model_dir, "classifier.joblib"))
    # create explainer (wraps classifier)
    explainer = shap.TreeExplainer(classifier)
    # combine into single dict
    model_assets = {
        "data_schema": data_schema,
        "features_schema": features_schema,
        "preprocessor": preprocessor,
        "classifier": classifier,
        "explainer": explainer
    }
    return model_assets


def assert_json(content_type):
    assert (
        content_type == "application/json"
    ), "content_type must be 'application/json'"


def parse_entities(entities):
    entities = entities.strip()
    entities = entities.split('=')
    assert entities[0] == 'entities', 'Unexpected field in content type.'
    entities = entities[1]
    entities = entities.split(',')
    return entities


def input_fn(request_body_str, request_content_type):
    fields = request_content_type.split(';')
    if len(fields) == 1:
        content_type = request_content_type[0]
        entities = ["predictions"]  # default entity
    elif len(fields) == 2:
        content_type, entities = fields
        entities = parse_entities(entities)
    else:
        raise Exception('Unexpected field in content_type.')
    assert_json(content_type)
    request = {
        'data': json.loads(request_body_str),
        'entities': entities
    }
    return request


def preprocess_fn(data, model_assets):
    model_assets["data_schema"].validate(data)
    data = model_assets["data_schema"].transform(data)
    data = np.expand_dims(data, axis=0)
    features = model_assets["preprocessor"].transform(data)
    return features


def predict_fn(request, model_assets):
    data = request['data']
    entities = request['entities']
    response = {}
    if 'data' in entities:
        response['data'] = data
    features = preprocess_fn(data, model_assets)
    if 'features' in entities:
        feature_names = model_assets["features_schema"].item_titles
        feature_values = features[0].tolist()
        response['features'] = {k: v for k, v in zip(feature_names, feature_values)}
    if 'descriptions' in entities:
        response['descriptions'] = model_assets["features_schema"].item_descriptions_dict
    if 'prediction' in entities:
        prediction = model_assets["classifier"].predict_proba(features)
        # take first sample (idx=0)
        # and second probability (idx=1) corresponding to the positive class
        response['prediction'] = prediction[0][1].tolist()
    if ('explanation_shap_values' in entities) or ('explanation_shap_interaction_values' in entities):
        explanation = {}
        expected_value = model_assets["explainer"].expected_value
        # see https://github.com/slundberg/shap/issues/729: handle both cases
        if expected_value.shape == (1,):
            explanation['expected_value'] = expected_value[0].tolist()
        else:
            explanation['expected_value'] = expected_value[1].tolist()
        if 'explanation_shap_values' in entities:
            # second probability (idx=1) corresponding to the positive class
            # and take first sample (idx=0)
            feature_names = model_assets["features_schema"].item_titles
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                shap_values = model_assets["explainer"].shap_values(features)[1][0]
            explanation['shap_values'] = {k: v for k, v in zip(feature_names, shap_values.tolist())}
        if 'explanation_shap_interaction_values' in entities:
            labels = model_assets["features_schema"].item_titles
            # take first sample (idx=0)
            values = model_assets["explainer"].shap_interaction_values(features)[0].tolist()
            explanation['shap_interaction_values'] = {
                'labels': labels,
                'values': values
            }
        # see https://github.com/slundberg/shap/issues/729: setting back to original
        model_assets["explainer"].expected_value = expected_value
        response['explanation'] = explanation
    return response


def output_fn(response, response_content_type):
    assert (
        response_content_type == "application/json"
    ), "accept must be 'application/json'"
    response_body_str = json.dumps(response)
    return response_body_str
