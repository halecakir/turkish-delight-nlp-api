from turkishdelightnlp.models.joint.runtime import (
    load_model,
    predict_dependecy,
    predict_joint,
    predict_pos,
    predict_morphemes,
    predict_morpheme_tags,
)
from turkishdelightnlp.services.base import BaseModel
from turkishdelightnlp.common.enums import JointModelSubTypes


def load_models(models_info: dict):
    types_of_models = {
        "JointModel": JointModel,
        # "SemanticParser": JointModel,
        # "Stemmer": JointModel,
        # "NER": JointModel,
    }
    models = {}
    for type, model_info in models_info.items():
        if type in types_of_models:
            models[type] = types_of_models[type](model_info)
        else:
            # TODO:
            continue
            raise Exception(f"Invalid model type : {type}")
    return models


class JointModel(BaseModel):
    def __init__(self, model_info):
        self.model_info = model_info
        self.model = self.load_local_model()

    def load_local_model(self):
        model_path = self.model_info["model_path"]
        model_opts_path = self.model_info["model_opts_path"]
        return load_model(model_path, model_opts_path)

    def _pre_process(self, payload):
        return payload

    def _post_process(self, prediction):
        return prediction

    def on_predict(self, features, submodel_type=None):
        subtypes = {
            "joint": predict_joint,
            "dependency": predict_dependecy,
            "pos": predict_pos,
            "morph": predict_morphemes,
            "mtag": predict_morpheme_tags,
        }
        if str(submodel_type) in subtypes:
            return subtypes[submodel_type](self.model, features)
        else:
            raise Exception(f"Unknown submodel type {submodel_type}")
