import streamlit as st
from pathlib import Path
import json
import boto3

from package import utils, config


@st.cache
def list_explanation_groups():
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(
        Bucket=config.S3_BUCKET,
        Delimiter='/',
        Prefix=config.EXPLANATIONS_S3_PREFIX + '/'
    )
    prefixes = [p['Prefix'] for p in response['CommonPrefixes']]
    return prefixes


@st.cache
def load_explanation_group(prefix):
    s3_client = boto3.client('s3')
    response = s3_client.list_objects_v2(
        Bucket=config.S3_BUCKET,
        Prefix=prefix
    )
    keys = [c['Key'] for c in response['Contents']]
    keys = [k for k in keys if k.endswith('.out')]
    explanations = []
    for key in keys:
        obj = s3_client.get_object(
            Bucket=config.S3_BUCKET,
            Key=key
        )
        json_lines = obj['Body'].read().decode('utf-8').split('\n')
        for line in json_lines:
            if line:
                explanation = json.loads(line)
                explanations.append(explanation)
    return explanations
