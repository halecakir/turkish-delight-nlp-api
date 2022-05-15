import pickle

from .learner import jPosDepLearner


def load_model(model_path, model_opt_path):
    with open(model_opt_path, "rb") as paramsfp:
        words, w2i, c2i, m2i, t2i, morph_dict, pos, rels, stored_opt = pickle.load(
            paramsfp
        )
        stored_opt.external_embedding = None

    print("Loading pre-trained parser model")
    parser = jPosDepLearner(
        words, pos, rels, w2i, c2i, m2i, t2i, morph_dict, stored_opt
    )
    parser.Load(model_path)
    return parser


def predict_joint(model, sentence):
    result = ""
    doc = {}
    _, morhps = model.predict_morphemes(sentence)
    for entry, morph in zip(model.predict_sentence(sentence), morhps):
        if entry.form == "*root*":
            continue
        doc[entry.form] = morph
        result += str(entry) + "\n"
    return {"conllu": result, "sentence": sentence, "morphemes": doc}


def predict_dependecy(model, sentence):
    result = ""
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        entry.pred_pos = None
        entry.xpos = None
        entry.pred_tags_tokens = None
        entry.feats = None
        result += str(entry) + "\n"
    return {"conllu": result, "sentence": sentence}


def predict_morphemes(model, sentence):
    doc = {}
    tokens, morhps = model.predict_morphemes(sentence)
    for entry, morph in zip(tokens, morhps):
        if entry == "*root*":
            continue
        doc[entry] = morph
    return {"sentence": sentence, "morphemes": doc}


def predict_pos(model, sentence):
    result = ""
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        entry.pred_parent_id = None
        entry.pred_relation = None
        entry.xpos = None
        entry.pred_tags_tokens = None
        entry.feats = None
        result += str(entry) + "\n"
    return {"conllu": result, "sentence": sentence}


def predict_morpheme_tags(model, sentence):
    result = ""
    for entry in model.predict_sentence(sentence):
        if entry.form == "*root*":
            continue
        entry.pred_parent_id = None
        entry.pred_relation = None
        entry.pred_pos = None
        entry.xpos = None
        result += str(entry) + "\n"
    return {"conllu": result, "sentence": sentence}
