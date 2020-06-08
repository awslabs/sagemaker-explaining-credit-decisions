from dotenv import load_dotenv
import os
from pathlib import Path

from package import utils


current_folder = utils.get_current_folder(globals())
dotenv_filepath = Path(current_folder, '../../.env').resolve()
if dotenv_filepath.exists():
    load_dotenv()

AWS_ACCOUNT_ID = os.environ['AWS_ACCOUNT_ID']
AWS_REGION = os.environ['AWS_REGION']

STACK_NAME = os.environ['EXPLAIN_STACK_NAME']

S3_BUCKET = os.environ['EXPLAIN_S3_BUCKET']
DATASETS_S3_PREFIX = 'datasets'
SCHEMAS_S3_PREFIX = 'schemas'
OUTPUTS_S3_PREFIX = 'outputs'
EXPLANATIONS_S3_PREFIX = 'explanations'

SAGEMAKER_IAM_ROLE = os.environ['EXPLAIN_SAGEMAKER_IAM_ROLE']

GLUE_DATABASE = os.environ['EXPLAIN_GLUE_DATABASE']
GLUE_WORKFLOW = os.environ['EXPLAIN_GLUE_WORKFLOW']

ECR_REPOSITORY = os.environ['EXPLAIN_ECR_REPOSITORY']
DOCKER_CONFIG = '/home/ec2-user/.docker/config.json'
