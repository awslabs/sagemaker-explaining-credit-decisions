import json
import os
from pathlib import Path

from package import utils


current_folder = utils.get_current_folder(globals())
cfn_stack_outputs_filepath = Path(current_folder, '../../stack_outputs.json').resolve()
with open(cfn_stack_outputs_filepath) as f:
    cfn_stack_outputs = json.load(f)

SAGEMAKER_MODE = cfn_stack_outputs['SagemakerMode']

AWS_ACCOUNT_ID = cfn_stack_outputs['AwsAccountId']
AWS_REGION = cfn_stack_outputs['AwsRegion']
AWS_PROFILE = 'sm-soln' if SAGEMAKER_MODE == 'Studio' else 'default' 

SOLUTION_PREFIX = cfn_stack_outputs['SolutionPrefix']

S3_BUCKET = cfn_stack_outputs['S3Bucket']
DATASETS_S3_PREFIX = 'datasets'
SCHEMAS_S3_PREFIX = 'schemas'
OUTPUTS_S3_PREFIX = 'outputs'
EXPLANATIONS_S3_PREFIX = 'explanations'

IAM_ROLE = cfn_stack_outputs['IamRole']

GLUE_DATABASE = cfn_stack_outputs['GlueDatabase']
GLUE_WORKFLOW = cfn_stack_outputs['GlueWorkflow']

ECR_REPOSITORY = cfn_stack_outputs['EcrRepository']
ECR_IMAGE = f"{AWS_ACCOUNT_ID}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY}:latest"
CODE_BUILD_PROJECT = cfn_stack_outputs['ContainerBuildProject']
DOCKER_CONFIG = '/home/ec2-user/.docker/config.json'

TAG_KEY = 'sagemaker-solution'

NOTEBOOKS_PATH = Path(current_folder, '../../notebooks').resolve()