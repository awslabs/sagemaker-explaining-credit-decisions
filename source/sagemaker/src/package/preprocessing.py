from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

from . import schemas


NUMERICAL_TYPES = set(["boolean", "integer", "number"])
CATEGORICAL_TYPES = set(["string"])


class AsTypeFloat32(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.astype("float32")


def get_numerical_idxs(data_schema):
    idxs = get_idxs(data_schema, NUMERICAL_TYPES)
    return idxs


def get_categorical_idxs(data_schema):
    idxs = get_idxs(data_schema, CATEGORICAL_TYPES)
    return idxs


def get_idxs(data_schema, types):
    idxs = []
    for idx, type in enumerate(data_schema.item_types):
        if type in types:
            idxs.append(idx)
    return idxs


def create_preprocessor(data_schema) -> ColumnTransformer:
    numerical_idxs = get_numerical_idxs(data_schema)
    numerical_transformer = AsTypeFloat32()
    categorical_idxs = get_categorical_idxs(data_schema)
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("numerical", numerical_transformer, numerical_idxs),
            ("categorical", categorical_transformer, categorical_idxs),
        ],
        remainder="drop",
    )
    return preprocessor


def preprocess_numerical_schema(preprocessor, data_schema):
    num_idx = [e[0] for e in preprocessor.transformers].index("numerical")
    numerical_idxs = get_numerical_idxs(data_schema)
    numerical_items = [data_schema.items[idx] for idx in numerical_idxs]
    features = []
    for item in numerical_items:
        feature = {
            "title": item["title"],
            "description": item["description"],
            "type": "number"
        }
        features.append(feature)
    return num_idx, features


def preprocess_categorical_schema(preprocessor, data_schema):
    cat_idx = [e[0] for e in preprocessor.transformers].index("categorical")
    categorical_idxs = get_categorical_idxs(data_schema)
    categorical_items = [data_schema.items[idx] for idx in categorical_idxs]
    features = []
    ohe = preprocessor.transformers_[cat_idx][1]
    for item, categories in zip(categorical_items, ohe.categories_):
        for category in categories:
            feature = {
                "title": "{}__{}".format(item["title"], category),
                "description": "{} is '{}' if value is 1.0.".format(
                    item["description"].strip('.'), category
                ),
                "type": "number"
            }
            features.append(feature)
    return cat_idx, features


def transform_schema(preprocessor, data_schema):
    num_idx, num_features = preprocess_numerical_schema(preprocessor, data_schema)  # noqa
    cat_idx, cat_features = preprocess_categorical_schema(preprocessor, data_schema)  # noqa
    assert num_idx < cat_idx, "Ordering should be numerical, then categorical."
    features = num_features + cat_features

    array_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "minItems": len(features),
        "maxItems": len(features),
        "items": features,
        "title": data_schema.title,
        "description": data_schema.description.replace(
            "items", "features"
        ),
    }
    return schemas.Schema(array_schema)
