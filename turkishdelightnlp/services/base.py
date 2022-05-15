from abc import ABC, abstractmethod

from turkishdelightnlp.core.messages import NO_VALID_PAYLOAD
from loguru import logger


class BaseModel(ABC):
    def __init__(self, model_info):
        self.model_info = model_info

    @abstractmethod
    def load_local_model(self):
        pass

    @abstractmethod
    def _pre_process(self, payload):
        pass

    @abstractmethod
    def _post_process(self, prediction):
        pass

    @abstractmethod
    def on_predict(self, features, submodel_type=None):
        pass

    def predict(self, payload, submodel_type=None):
        if payload is None:
            raise ValueError(NO_VALID_PAYLOAD.format(payload))

        pre_processed_payload = self._pre_process(payload)
        prediction = self.on_predict(pre_processed_payload, submodel_type)
        logger.info(prediction)
        post_processed_result = self._post_process(prediction)

        return post_processed_result
