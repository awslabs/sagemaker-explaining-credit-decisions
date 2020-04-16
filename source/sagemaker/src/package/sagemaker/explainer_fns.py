import shap
from pathlib import Path

from package.data import schemas
from package.sagemaker import predictor_fns


def model_fn(model_dir):
    model_dir = Path(model_dir)
    model_assets = predictor_fns.model_fn(model_dir)
    # explainer wraps the classifier
    explainer = shap.TreeExplainer(model_assets['classifier'])
    model_assets['explainer'] = explainer
    # used for explanation descriptions
    features_schema = schemas.from_json_schema(
        Path(model_dir, "features.schema.json")
    )
    model_assets["features_schema"] = features_schema
    return model_assets


def predict_fn(data, model_assets):
    features = predictor_fns.preprocess_fn(data, model_assets)
    # take second probability (idx=1) corresponding to the positive class
    # and first sample (idx=0)
    shap_values = model_assets["explainer"].shap_values(features)[1][0]
    # take second probability (idx=1) corresponding to the positive class
    expected_value = model_assets["explainer"].expected_value[1]
    prediction = predictor_fns.predict_fn(data, model_assets)
    prediction['explanation'] = {
        "shap_values": shap_values.tolist(),
        "expected_value": expected_value,
        "feature_values": features[0].tolist(),  # take first sample (idx=0)
        "feature_names": model_assets["features_schema"].item_titles,
        "feature_descriptions": model_assets["features_schema"].item_descriptions  # noqa
    }
    return prediction
