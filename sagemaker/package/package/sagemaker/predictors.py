from sagemaker.predictor import (
    RealTimePredictor,
    json_serializer,
    json_deserializer,
    CONTENT_TYPE_JSON,
)


class Predictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session=None):
        super(Predictor, self).__init__(
            endpoint=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=json_serializer,
            deserializer=json_deserializer,
            content_type="application/json; entities=predictions",
            accept=CONTENT_TYPE_JSON,
        )


class Explainer(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session=None):
        entities = [
            'data',
            'features',
            'descriptions',
            'prediction',
            'explanation_shap_values',
            'explanation_shap_interaction_values'
        ]
        super(Explainer, self).__init__(
            endpoint=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=json_serializer,
            deserializer=json_deserializer,
            content_type="application/json; entities={}".format(",".join(entities)),
            accept=CONTENT_TYPE_JSON,
        )
