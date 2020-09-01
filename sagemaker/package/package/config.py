import json
import os
from pathlib import Path

from package import utils


current_folder = utils.get_current_folder(globals())
cfn_stack_outputs_filepath = Path(current_folder, '../../stack_outputs.json').resolve()
with open(cfn_stack_outputs_filepath) as f:
    cfn_stack_outputs = json.load(f)

AWS_ACCOUNT_ID = cfn_stack_outputs['AwsAccountId']
AWS_REGION = cfn_stack_outputs['AwsRegion']

RESOURCE_NAME = cfn_stack_outputs['ResourceName']

S3_BUCKET = cfn_stack_outputs['S3Bucket']
DATASETS_S3_PREFIX = 'datasets'
SCHEMAS_S3_PREFIX = 'schemas'
OUTPUTS_S3_PREFIX = 'outputs'
EXPLANATIONS_S3_PREFIX = 'explanations'

SAGEMAKER_IAM_ROLE = cfn_stack_outputs['SagemakerIamRole']

GLUE_DATABASE = cfn_stack_outputs['GlueDatabase']
GLUE_WORKFLOW = cfn_stack_outputs['GlueWorkflow']

ECR_REPOSITORY = cfn_stack_outputs['EcrRepository']
DOCKER_CONFIG = '/home/ec2-user/.docker/config.json'

TAG_KEY = 'sagemaker-solution'