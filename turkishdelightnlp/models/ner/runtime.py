from .learner import NERModel
from .config import Config


def load_model(model_path, model_opts_path):
    config = Config(model_opts_path)
    model = NERModel(config)
    model.build()
    model.restore_session(model_path)
    return model


def predict_ner(model, sentence):
    doc = {"text": sentence, "ents": [], "title": None}

    tokens = sentence.split()
    preds = model.predict_ner(tokens)
    start = 0
    for t, p in zip(tokens, preds):
        if p != "O":
            doc["ents"].append({"start": start, "end": start + len(t), "label": p})
        start += len(t) + 1
    return doc
