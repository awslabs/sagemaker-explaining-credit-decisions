# %% [markdown]
# # Endpoint for Explanations
#
# In this notebook, we'll deploy the model explainer to a HTTP endpoint
# using Amazon SageMaker and visualize the explanations.
#
# You can bring also
# bring your own trained models to explain. See the customizing section for
# more details.
#
# <p align="center">
#   <img src="https://github.com/awslabs/sagemaker-explaining-credit-decisions/raw/master/docs/architecture_diagrams/stage_3.png" width="1000px">
# </p>

# %% [markdown]
# We start by setting up the environment (e.g. install packages, etc) if this has
# not been done already.

# %%
# !python ../env_setup.py

# %% [markdown]
# We then import a variety of packages that will be used throughout
# the notebook. One of the most important packages used throughout this
# solution is the Amazon SageMaker Python SDK (i.e. `import sagemaker`). We
# also import modules from our own custom package that can be found at
# `./package`.

# %%
from bokeh.plotting import output_notebook
import boto3
import sagemaker
from pathlib import Path
from sagemaker.sklearn import SKLearnModel
import sys

sys.path.append('../package')
from package import config, utils, visuals
from package.data import schemas
from package.sagemaker import predictors

# %% [markdown]
# Up next, we define the current folder, a sagemaker session and a
# sagemaker client (from `boto3`).

# %%
current_folder = utils.get_current_folder(globals())
sagemaker_session = sagemaker.Session()
sagemaker_client = boto3.client('sagemaker')

# %% [markdown]
# We define a couple of functions below to retrive the model data (i.e.
# `model.tar.gz`) from the most recent trained model (from the last stage).

# %%
def get_latest_training_job(name_contains):
    response = sagemaker_client.list_training_jobs(
        NameContains=name_contains,
        StatusEquals='Completed'
    )
    training_jobs = response['TrainingJobSummaries']
    assert len(training_jobs) > 0, "Couldn't find any completed training jobs with '{}' in name.".format(name_contains)
    latest_training_job = training_jobs[0]['TrainingJobName']
    return latest_training_job


def get_model_data(training_job):
    response = sagemaker_client.describe_training_job(TrainingJobName=training_job)
    assert 'ModelArtifacts' in response, "Couldn't find ModelArtifacts for training job."
    return response['ModelArtifacts']['S3ModelArtifacts']

# %%
latest_training_job = get_latest_training_job(config.SOLUTION_PREFIX)
model_data = get_model_data(latest_training_job)

# %% [markdown]
# Our model explainer endpoint will be named as per the `explainer_name`
# variable. AWS CloudFormation will delete this endpoint (and endpoint
# configuration) during stack deletion if the `endpoint_name` is kept as
# is. You will need to manually delete the endpoint (and endpoint
# configuration) after stack deletion if you change this.

# %%
explainer_name = "{}-explainer".format(config.SOLUTION_PREFIX)

# %% [markdown]
# We define the model to deploy which includes the explainer logic.

# %%
model = SKLearnModel(
    name=explainer_name,
    model_data=model_data,
    role=config.IAM_ROLE,
    entry_point='entry_point.py',
    source_dir=str(Path(current_folder, '../containers/model/src').resolve()),
    dependencies=[str(Path(current_folder, '../package/package').resolve())],
    image=config.ECR_IMAGE,
    code_location='s3://' + str(Path(config.S3_BUCKET, config.OUTPUTS_S3_PREFIX))
)

# %% [markdown]
# Calling `deploy` will start a container to host the model.
# You can expect this step to take approximately 5 minutes.

# %%
model.deploy(
    endpoint_name=explainer_name,
    instance_type='ml.c5.xlarge',
    initial_instance_count=1,
    tags=[{'Key': config.TAG_KEY, 'Value': config.SOLUTION_PREFIX}]
)

# %% [markdown]
# When you're trying to update the model for development purposes, but
# experiencing issues because the model/endpoint-config/endpoint already
# exists, you can delete the existing model/endpoint-config/endpoint by
# uncommenting and running the following commands:

# %%
# sagemaker_client.delete_endpoint(EndpointName=explainer_name)
# sagemaker_client.delete_endpoint_config(EndpointConfigName=explainer_name)
# sagemaker_client.delete_model(ModelName=explainer_name)

# %% [markdown]
# When calling our new endpoint from the notebook, we use a Amazon
# SageMaker SDK
# [`Predictor`](https://sagemaker.readthedocs.io/en/stable/predictors.html).
# A `Predictor` is used to send data to an endpoint (as part of a request),
# and interpret the response. Creating a `Predictor` does not affect the
# actual endpoint. Our endpoint expects to receive (and also sends) JSON
# formatted objects, and uses `content_type` to specify the entities
# requested (e.g. prediction, features, explanation_shap_values, etc.), so
# we create a custom `Predictor` called `Explainer`. JSON is used because
# it is a standard endpoint format and the endpoint response contains a
# nested data structure.

# %%
explainer = predictors.Explainer(explainer_name, sagemaker_session)

# %% [markdown]
# ## Model Explanations
# We can demonstrate the output of our new `explainer` endpoint with an
# example. One option would be to take a sample from our test set, but
# let's construct a sample by hand. Our example credit application is for
# 6000 EUR and will be put towards buying a used car. You can always come
# back later and make changes to certain values.

# %%
sample = {
    'contact__has_telephone': False,
    'credit__amount': 6000,
    'credit__coapplicant': 1,
    'credit__duration': 36,
    'credit__guarantor': 0,
    'credit__installment_rate': 3,
    'credit__purpose': 'used_car',
    'employment__duration': 0,
    'employment__permit': 'foreign',
    'employment__type': 'professional',
    'finance__accounts__checking__balance': 'no_account',
    'finance__accounts__savings__balance': 'low',
    'finance__credits__other_banks': 0,
    'finance__credits__other_stores': 0,
    'finance__credits__this_bank': 1,
    'finance__other_assets': 'life_insurance',
    'finance__repayment_history': 'good',
    'personal__num_dependents': 1,
    'residence__duration': 4,
    'residence__type': 'own'
}

# %% [markdown]
# We can call `explainer.predict` with features (for a credit application)
# to obtain a prediction and its associated explanation. Using `Explainer`,
# the features will be converted from a Python list into a JSON string
# (using the Amazon SageMaker Python SDK's in-built `json_serializer`).
# Additionally, it will notify to the endpoint that the contents being sent
# are JSON formatted and the explanation entities are required (via
# `content_type`), and a JSON formatted response is requested in return
# (via `accept`). And lastly, the JSON response is converted back into
# Python objects (using `json_deserializer`).
#
# **Caution**: the probability returned by this model has not been
# calibrated. When the model gives a probability of credit default of 20%,
# for example, this does not necessarily mean that 20% of applications with
# a probability of 20% resulted in credit default. Calibration is a useful
# property in certain circumstances, but is not required in cases where
# discrimination between cases of default and non-defult is sufficient.
# [CalibratedClassifierCV](https://scikit-learn.org/stable/modules/generated/sklearn.calibration.CalibratedClassifierCV.html)
# from
# [Scikit-learn](https://scikit-learn.org/stable/modules/calibration.html)
# can be used to calibrate a model. Calibration also has an impact on the
# explanations. Since the calibration process is typically non-linear, it
# breaks the additive property of Shapley Values.
# [`KernelExplainer`](https://shap.readthedocs.io/en/latest/) can handle
# this case, but is typically much slower to compute the explanations.

# %%
output = explainer.predict(sample)
prediction = output['prediction']
print("Credit default risk: {:.2%}".format(prediction))

# %% [markdown]
# ## Visualizing Explanations
# Although `output` contains all the information required to explain
# the machine learning model's prediction, looking at long lists of numbers
# isn't especially helpful. We provide a number of visualization that
# clearly show which features increase and decrease the risk of credit
# default for an individual credit application.
#
# A waterfall chart can be used to show the cumulative effect of each
# feature. Starting with the baseline probability for credit defaults (at
# the bottom of the chart), we can see how each additional feature shifts
# the probability. Green arrows indicate that the feature <span
# style="color:#69AE35">*decreased* the predicted credit default
# risk</span> for the individual credit application. While red arrows
# indicate that the feature <span style="color:#FF5733">*increased* the
# predicted credit default risk</span> for the individual credit
# application. After all features have been considered, we reach the final
# predicted credit default risk (at the top of the chart).
#
# We're using [`bokeh`](https://docs.bokeh.org/en/latest/index.html#) for 
# interactive charts, so let's start by calling `output_notebook` to show the
# plots inside the notebook.

# %%
output_notebook()

# %% [markdown]
# ### Summary Explanation
# As mentioned earlier on in this notebook, our features can be grouped
# together into categories. We can extract the top level category for each
# feature, by extracting the start of the feature name before the level
# seperator. We use two consecutive underscores (`__`) as our level
# separator. Once we have the category for each feature, we can calculate
# the the overall effect for each category. All of this is performed in
# `summarize_explanation`.

# %%
explanation_summary = visuals.summary_explanation(output)

# %% [markdown]
# We then show the associated waterfall chart.

# %%
x_axis_label = 'Credit Default Risk Score (%)'
summary_waterfall = visuals.WaterfallChart(
    baseline=explanation_summary['expected_value'],
    shap_values=explanation_summary['shap_values'],
    names=explanation_summary['feature_names'],
    descriptions=explanation_summary['feature_descriptions'],
    max_features=10,
    x_axis_label=x_axis_label,
)
summary_waterfall.show()

# %% [markdown]
# We can see from the summary waterfall chart above that features related
# to finance have the largest combined effect on the credit default risk.
# Although features realted to finance reduce the credit default risk, the
# features related to employment bring the risk back up again to a certain
# degree.

# %% [markdown]
# ### Detailed Explanation
# After examining the high level explanation, we can drill down into the
# individual features that contribute to the credit default risk score.

# %%
explanation = visuals.detailed_explanation(output)

# %%
detailed_waterfall = visuals.WaterfallChart(
    baseline=explanation['expected_value'],
    shap_values=explanation['shap_values'],
    names=explanation['feature_names'],
    feature_values=explanation['feature_values'],
    descriptions=explanation['feature_descriptions'],
    max_features=10,
    x_axis_label=x_axis_label
)
detailed_waterfall.show()

# %% [markdown]
# We can see from the detailed waterfall chart above that not having a
# checking account with the same bank indicates a lower credit default
# risk. Since this is an influential feature for the model but the reason
# for this effect is not obvious, it may warrant further investigation. We
# can also see that using the credit to purchase a used car is associated
# with a lower credit default risk too. After this we see a number of
# features that increase the credit default risk: a credit amount of 6000
# EUR, a lack of employment and a credit duration of 36 months. Another
# potential area for investigation, would be related to the repayment
# history feature. We can see that *not* having a very poor repayment
# history is associated with a higher credit default risk score. We may
# have artifacts in the datasets that caused the model to use this feature
# in such an unintuitive way.

# %% [markdown]
# ### Counterfactual Example
# And lastly, we switch the value of the checking account balance of the
# applicant from `no_account` to `negative`. We can then see how the
# overall prediction of the model changes, and also see the updated
# contribution of this feature. Clearly, this application has become
# substantially more risky.

# %%
counter_sample = dict(sample)
counter_sample['finance__accounts__checking__balance'] = 'negative'  # from 'no_account'
counter_output = explainer.predict(counter_sample)
counter_explanation = visuals.detailed_explanation(counter_output)
visuals.WaterfallChart(
    baseline=counter_explanation['expected_value'],
    shap_values=counter_explanation['shap_values'],
    names=counter_explanation['feature_names'],
    feature_values=counter_explanation['feature_values'],
    descriptions=counter_explanation['feature_descriptions'],
    max_features=10,
    x_axis_label=x_axis_label,
).show()

# %% [markdown]
# ## Clean Up
#
# When you've finished with the explainer endpoint (and associated
# endpoint-config), make sure that you delete it to avoid accidental
# charges.

# %%
# sagemaker_client.delete_endpoint(EndpointName=explainer_name)

# %% [markdown]
# ## Next Stage
#
# Up next we'll use Amazon SageMaker Batch Transform to obtain explanations
# for our complete dataset.
#
# [Click here to continue.](./4_batch_transform.ipynb)

# %%
