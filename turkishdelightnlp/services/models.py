from turkishdelightnlp.models.joint.runtime import load_model as joint_lm
from turkishdelightnlp.models.joint.runtime import (
    predict_dependecy,
    predict_joint,
    predict_morpheme_tags,
    predict_morphemes,
    predict_pos,
)
from turkishdelightnlp.models.semantic_parser.runtime import (
    load_model as semantic_parser_lm,
)
from turkishdelightnlp.models.semantic_parser.runtime import predict_semantic
from turkishdelightnlp.models.ner.runtime import load_model as ner_lm
from turkishdelightnlp.models.ner.runtime import predict_ner
from turkishdelightnlp.models.stemmer.runtime import load_model as stemmer_lm
from turkishdelightnlp.models.stemmer.runtime import predict_stems
from turkishdelightnlp.services.base import BaseModel


def load_models(models_info: dict):
    types_of_models = {
        "JointModel": JointModel,
        "SemanticParser": SemanticParserModel,
        "Stemmer": StemmerModel,
        "NER": NERModel,
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
        return joint_lm(model_path, model_opts_path)

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


class SemanticParserModel(BaseModel):
    def __init__(self, model_info):
        self.model_info = model_info
        self.model = self.load_local_model()

    def load_local_model(self):
        model_path = self.model_info["model_path"]
        return semantic_parser_lm(model_path)

    def _pre_process(self, payload):
        return payload

    def _post_process(self, prediction):
        return prediction

    def on_predict(self, features, submodel_type=None):
        return predict_semantic(self.model, features)


class NERModel(BaseModel):
    def __init__(self, model_info):
        self.model_info = model_info
        self.model = self.load_local_model()

    def load_local_model(self):
        model_path = self.model_info["model_path"]
        model_opts_path = self.model_info["model_opts_path"]
        return ner_lm(model_path, model_opts_path)

    def _pre_process(self, payload):
        return payload

    def _post_process(self, prediction):
        return prediction

    def on_predict(self, features, submodel_type=None):
        doc = predict_ner(self.model, features)
        return doc


class StemmerModel(BaseModel):
    def __init__(self, model_info):
        self.model_info = model_info
        self.model = self.load_local_model()

    def load_local_model(self):
        model_path = self.model_info["model_path"]
        return stemmer_lm(model_path)

    def _pre_process(self, payload):
        return payload

    def _post_process(self, prediction):
        return prediction

    def on_predict(self, features, submodel_type=None):
        doc = predict_stems(self.model, features)
        return doc
