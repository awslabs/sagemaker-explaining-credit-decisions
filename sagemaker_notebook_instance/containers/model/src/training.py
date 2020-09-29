"""
TRAINING FUNCTIONS: this file in run in 'script mode' when `.fit` is called
from the notebook. `parse_args` and `train_fn` are called in the
`if __name__ =='__main__'` block.
"""
import argparse
import joblib
from lightgbm import LGBMClassifier
import os
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from package.data import schemas, datasets


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


def load_schemas(schemas_folder):
    data_schema_filepath = Path(schemas_folder, "data.schema.json")
    data_schema = schemas.from_json_schema(data_schema_filepath)
    label_schema_filepath = Path(schemas_folder, "label.schema.json")
    label_schema = schemas.from_json_schema(label_schema_filepath)
    return data_schema, label_schema


def log_cross_val_auc(clf, X, y, cv_splits, log_prefix):
    cv_auc = cross_val_score(clf, X, y, cv=cv_splits, scoring='roc_auc')
    cv_auc_mean = cv_auc.mean()
    cv_auc_error = cv_auc.std() * 2
    log = "{}_auc_cv: {:.5f} (+/- {:.5f})"
    print(log.format(log_prefix, cv_auc_mean, cv_auc_error))


def log_auc(clf, X, y, log_prefix):
    y_pred_proba = clf.predict_proba(X)
    auc = roc_auc_score(y, y_pred_proba[:, 1])
    log = '{}_auc: {:.5f}'
    print(log.format(log_prefix, auc))


def train_pipeline(pipeline, X, y, cv_splits):
    # fit pipeline to cross validation splits
    if cv_splits > 1:
        log_cross_val_auc(pipeline, X, y, cv_splits, 'train')
    # fit pipeline to all training data
    pipeline.fit(X, y)
    log_auc(pipeline, X, y, 'train')
    return pipeline


def test_pipeline(pipeline, X, y):
    log_auc(pipeline, X, y, 'test')


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


def train_fn(args):
    # # load data
    data_schema, label_schema = load_schemas(args.schemas)
    X_train = datasets.read_json_dataset(args.data_train, data_schema)
    y_train = datasets.read_json_dataset(args.label_train, label_schema)
    X_test = datasets.read_json_dataset(args.data_test, data_schema)
    y_test = datasets.read_json_dataset(args.label_test, label_schema)

    # convert from column vector to 1d array of int
    y_train = y_train[:, 0].astype('int')
    y_test = y_test[:, 0].astype('int')

    # create components
    preprocessor = create_preprocessor(data_schema)
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
    train_pipeline(pipeline, X_train, y_train, args.cv_splits)
    features_schema = transform_schema(preprocessor, data_schema)
    test_pipeline(pipeline, X_test, y_test)

    # save components
    model_dir = Path(args.model_dir)
    model_dir.mkdir(exist_ok=True, parents=True)
    joblib.dump(preprocessor, Path(model_dir, "preprocessor.joblib"))
    joblib.dump(classifier, Path(model_dir, "classifier.joblib"))
    data_schema.save(Path(model_dir, "data.schema.json"))
    features_schema.save(Path(model_dir, "features.schema.json"))
