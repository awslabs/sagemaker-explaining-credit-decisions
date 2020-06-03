# flake8: noqa: F401
import os
import sys

# import training function
from package.sagemaker.estimator_fns import parse_args, train_fn

# import deployment functions
from package.sagemaker.predictor_fns import model_fn, predict_fn, input_fn, output_fn


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    train_fn(args)
