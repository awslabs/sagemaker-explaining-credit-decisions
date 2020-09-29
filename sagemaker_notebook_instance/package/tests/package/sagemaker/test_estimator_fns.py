from package.sagemaker import estimator_fns


def test_parse_args():
    sys_args = [
        "--cv-splits", "1",
        "--model-dir", "test2",
        "--schemas", "test3",
        "--data-train", "test4",
        "--label-train", "test5",
        "--data-test", "test6",
        "--label-test",  "test7"
    ]
    args = estimator_fns.parse_args(sys_args)
    assert args.cv_splits == 1
    assert args.model_dir == "test2"
    assert args.schemas == "test3"
    assert args.data_train == "test4"
    assert args.label_train == "test5"
    assert args.data_test == "test6"
    assert args.label_test == "test7"
