from pathlib import Path
import sys
import boto3
import sagemaker

from package import config, utils
from package.data import datasets

current_folder = utils.get_current_folder(globals())
src_path = Path(current_folder, "../src").resolve()
sys.path.append(str(src_path))

import entry_point as ep


def test_train_fn():
    boto_session = boto3.session.Session(region_name=config.AWS_REGION)
    sagemaker_session = sagemaker.Session(boto_session)
    current_folder = utils.get_current_folder(globals())
    datasets_folder = Path(current_folder, "../datasets").resolve()
    datasets.clear_datasets(datasets_folder)
    sagemaker_session.download_data(
        path=str(datasets_folder),
        bucket=config.S3_BUCKET,
        key_prefix=config.DATASETS_S3_PREFIX,
    )

    sys_args = [
        "--model-dir",
        str(Path(current_folder, "../models").resolve()),
        "--schemas",
        str(Path(current_folder, "../schemas").resolve()),
        "--data-train",
        str(Path(current_folder, "../datasets/data_train").resolve()),
        "--label-train",
        str(Path(current_folder, "../datasets/label_train").resolve()),
        "--data-test",
        str(Path(current_folder, "../datasets/data_test").resolve()),
        "--label-test",
        str(Path(current_folder, "../datasets/label_test").resolve())
    ]
    args = ep.parse_args(sys_args)
    ep.train_fn(args)

    model_assets = ep.model_fn(args.model_dir)
    data = {
        'contact__has_telephone': False,
        'credit__amount': 433,
        'credit__coapplicant': 1,
        'credit__duration': 18,
        'credit__guarantor': 0,
        'credit__installment_rate': 3,
        'credit__purpose': 'car',
        'employment__duration': 0,
        'employment__permit': 'foreign',
        'employment__type': 'professional',
        'finance__accounts__checking__balance': 'no_account',
        'finance__accounts__savings__balance': 'high',
        'finance__credits__other_banks': 0,
        'finance__credits__other_stores': 0,
        'finance__credits__this_bank': 1,
        'finance__other_assets': 'real_estate',
        'finance__repayment_history': 'poor',
        'personal__age': 50,
        'personal__gender': 'male',
        'personal__num_dependents': 1,
        'personal__relationship_status': 'married',
        'residence__duration': 4,
        'residence__type': 'rent'
    }
    output = ep.predict_fn(data, model_assets)
    assert isinstance(output, dict)
    assert False
