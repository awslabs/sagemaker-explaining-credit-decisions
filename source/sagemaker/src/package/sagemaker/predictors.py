from sagemaker.predictor import (
    RealTimePredictor,
    json_serializer,
    json_deserializer,
    CONTENT_TYPE_JSON,
)


class JsonPredictor(RealTimePredictor):
    def __init__(self, endpoint_name, sagemaker_session=None):
        super(JsonPredictor, self).__init__(
            endpoint=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=json_serializer,
            deserializer=json_deserializer,
            content_type=CONTENT_TYPE_JSON,
            accept=CONTENT_TYPE_JSON,
        )
