import boto3
from time import sleep

from . import config
from . import schemas


glue_client = boto3.client("glue", region_name=config.AWS_REGION)


def start_workflow(workflow_name):
    workflow_run = glue_client.start_workflow_run(Name=workflow_name)
    workflow_run_id = workflow_run["RunId"]
    return workflow_run_id


def workflow_finished(workflow_name, workflow_run_id):
    status = glue_client.get_workflow_run(
        Name=workflow_name, RunId=workflow_run_id
    )
    finished = status["Run"]["Status"] == "COMPLETED"
    if finished:
        successful = (
            status["Run"]["Statistics"]["TotalActions"]
            == status["Run"]["Statistics"]["SucceededActions"]
        )
        if successful:
            return True
        else:
            raise Exception(
                "AWS Glue Workflow failed. "
                "Check the workflow logs on the console."
            )
    else:
        return False


def wait_for_workflow_finished(
    workflow_name, workflow_run_id, polling_frequency=10
):
    while not workflow_finished(workflow_name, workflow_run_id):
        sleep(polling_frequency)
        print('.', end='', flush=True)
    print('\nAWS Glue Workflow has finished successfully.')


GLUE_TO_JSON_TYPES = {
    "boolean": "boolean",
    "tinyint": "integer",
    "smallint": "integer",
    "int": "integer",
    "bigint": "integer",
    "float": "number",
    "double": "number",
    "decimal": "number",
    "string": "string",
    "char": "string",
    "varchar": "string"
}


def get_table_schema(database_name, table_name):
    glue_client = boto3.client(
        "glue", region_name=config.AWS_REGION
    )  # use client, since no resource for glue.
    glue_table = glue_client.get_table(
        DatabaseName=database_name, Name=table_name
    )
    glue_table_schema = glue_table["Table"]["StorageDescriptor"]["Columns"]
    items = []
    for glue_column in glue_table_schema:
        item = {
            "title": glue_column["Name"],
            "type": GLUE_TO_JSON_TYPES[glue_column["Type"]],
        }
        items.append(item)
    array_schema = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "array",
        "minItems": len(items),
        "maxItems": len(items),
        "items": items,
    }
    return schemas.Schema(array_schema)
