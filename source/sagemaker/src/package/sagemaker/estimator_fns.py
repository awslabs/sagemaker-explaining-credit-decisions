"""
TRAINING FUNCTIONS: this file in run in 'script mode' when `.fit` is called
from the notebook. `parse_args` and `train_fn` are called in the
`if __name__ =='__main__'` block.
"""
import argparse
import os
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from package.data import schemas, datasets
from package.machine_learning import preprocessing, training


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
