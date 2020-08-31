import json
import os
from pathlib import Path

from package import utils


current_folder = utils.get_current_folder(globals())
cfn_stack_outputs_filepath = Path(current_folder, '../../stack_outputs.json').resolve()
with open(cfn_stack_outputs_filepath) as f:
    cfn_stack_outputs = json.load(f)

AWS_ACCOUNT_ID = cfn_stack_outputs['aws_account_id']
AWS_REGION = cfn_stack_outputs['aws_region']

RESOURCE_NAME = cfn_stack_outputs['explain_resource_name']

S3_BUCKET = cfn_stack_outputs['explain_s3_bucket']
DATASETS_S3_PREFIX = 'datasets'
SCHEMAS_S3_PREFIX = 'schemas'
OUTPUTS_S3_PREFIX = 'outputs'
EXPLANATIONS_S3_PREFIX = 'explanations'

SAGEMAKER_IAM_ROLE = cfn_stack_outputs['explain_sagemaker_iam_role']

GLUE_DATABASE = cfn_stack_outputs['explain_glue_database']
GLUE_WORKFLOW = cfn_stack_outputs['explain_glue_workflow']

ECR_REPOSITORY = cfn_stack_outputs['explain_ecr_repository']
DOCKER_CONFIG = '/home/ec2-user/.docker/config.json'

TAG_KEY = 'sagemaker_solution'
TAG_VALUE = 'explaining_credit_decisions'
