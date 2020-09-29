import boto3
import sys
import time
from .logs import logs_for_build

from package import config


codebuild_session = boto3.Session(region_name=config.AWS_REGION, profile_name=config.AWS_PROFILE)
codebuild_client = codebuild_session.client("codebuild")


def build(project_name, log=True):
    print("Starting a build job for CodeBuild project: {}".format(project_name))
    id = _start_build(project_name)
    if log:
        logs_for_build(id, wait=True, session=codebuild_session)
    else:
        _wait_for_build(id)


def _start_build(project_name):
    args = {"projectName": project_name}
    response = codebuild_client.start_build(**args)
    return response["build"]["id"]


def _wait_for_build(build_id, poll_seconds=10):
    status = codebuild_client.batch_get_builds(ids=[build_id])
    while status["builds"][0]["buildStatus"] == "IN_PROGRESS":
        print(".", end="")
        sys.stdout.flush()
        time.sleep(poll_seconds)
        status = codebuild_client.batch_get_builds(ids=[build_id])
    print()
    print(f"Build complete, status = {status['builds'][0]['buildStatus']}")
    print(f"Logs at {status['builds'][0]['logs']['deepLink']}")
