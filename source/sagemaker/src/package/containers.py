import boto3
import sagemaker
import docker
from pathlib import Path
import json
import subprocess

from . import config


ecr_client = boto3.client("ecr", region_name=config.AWS_REGION)
docker_client = docker.APIClient()


class Image:
    def __init__(self, registry, repository_name, tag="latest"):
        self.registry = registry
        self.repository_name = repository_name
        self.tag = tag
        self._check_credential_manager()
        self._configure_credentials()

    def __str__(self):
        return "{}/{}:{}".format(self.registry, self.repository_name, self.tag)

    @property
    def repository(self):
        return "{}/{}".format(self.registry, self.repository_name)

    @property
    def short_name(self):
        return self.repository_name

    @staticmethod
    def _check_credential_manager():
        try:
            subprocess.run(
                ["docker-credential-ecr-login", "version"],
                stdout=subprocess.DEVNULL,
            )
        except Exception:
            raise Exception(
                "Couldn't run 'docker-credential-ecr-login'. "
                "Make sure it is installed and configured correctly."
            )

    def _configure_credentials(self):
        docker_config_filepath = Path(config.DOCKER_CONFIG)
        if docker_config_filepath.exists():
            with open(docker_config_filepath, "r") as openfile:
                docker_config = json.load(openfile)
        else:
            docker_config = {}
        if "credHelpers" not in docker_config:
            docker_config["credHelpers"] = {}
        docker_config["credHelpers"][self.registry] = "ecr-login"
        docker_config_filepath.parent.mkdir(exist_ok=True, parents=True)
        with open(docker_config_filepath, "w") as openfile:
            json.dump(docker_config, openfile, indent=4)

    def build(self, dockerfile, buildargs):
        path = Path(dockerfile).parent
        for line in docker_client.build(
            path=str(path),
            buildargs=buildargs,
            tag=self.repository_name,
            decode=True,
        ):
            if "error" in line:
                raise Exception(line["error"])
            else:
                print(line)

    def push(self):
        docker_client.tag(
            self.repository_name, self.repository, self.tag, force=True
        )
        for line in docker_client.push(
            self.repository, self.tag, stream=True, decode=True
        ):
            print(line)


def scikit_learn_image():
    registry = sagemaker.fw_registry.registry(
        region_name=config.AWS_REGION, framework="scikit-learn"
    )
    repository_name = "sagemaker-scikit-learn"
    tag = "0.20.0-cpu-py3"
    return Image(registry, repository_name, tag)


def custom_image():
    registry = "{}.dkr.ecr.{}.amazonaws.com".format(
        config.AWS_ACCOUNT_ID, config.AWS_REGION
    )
    repository_name = config.ECR_REPOSITORY
    tag = "latest"
    return Image(registry, repository_name, tag)
