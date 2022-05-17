import os
import sys
import string
import torch
import torch.utils.data as Data

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


from ucca.convert import passage2file, split2sentences, to_standard
from ucca.textutil import indent_xml
import xml.etree.ElementTree as ET
from corpus import Corpus
from dataset import collate_fn
from evaluator import UCCA_Evaluator
from ucca_parser import UCCA_Parser
from ucca import core, layer0, layer1, visualization
from ucca.ioutil import (
    get_passages,
    get_passages_with_progress_bar,
    external_write_mode,
)
import matplotlib.pyplot as plt
import stanza

save_path = "dataset"


# reload parser

stanza.download("tr")
nlp = stanza.Pipeline(
    lang="tr", processors="tokenize,mwt,pos,lemma,depparse", verbose=False
)
PUNCTUATION = set(string.punctuation)


def load_model(data_path):
    vocab_path = os.path.join(data_path, "vocab.pt")
    state_path = os.path.join(data_path, "parser.pt")
    config_path = os.path.join(data_path, "config.json")
    print(data_path, config_path)
    ucca_parser = UCCA_Parser.load(vocab_path, config_path, state_path)

    return ucca_parser


def predict_semantic(ucca_parser, sentence):
    p = core.Passage(1)
    l0 = layer0.Layer0(p)
    layer1.Layer1(p)
    doc = nlp(sentence)
    for word in doc.sentences[0].words:
        a = l0.add_terminal(text=word.text, punct=PUNCTUATION.issuperset(word.text))
        a.add_extra(
            pos=word.pos, deprel=word.head, ent_type=word.head, ent_iob=word.head
        )
    c = Corpus(p)

    test_loader = Data.DataLoader(
        dataset=c.generate_inputs(ucca_parser.vocab, False),
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
    )
    ucca_file = prediction(ucca_parser, test_loader)
    return {"ucca_xml" :  convert_xml(ucca_file[0]), "sentence" : split2sentences}


def convert_xml(passage):
    root = to_standard(passage)
    xml_string = ET.tostring(root).decode()
    output = indent_xml(xml_string)
    return output


def prediction(ucca_parser, loader):
    predicted = []
    for batch in loader:
        word_idxs, pos_idxs, dep_idxs, passages, trees, all_nodes, all_remote = batch
        pred_passages = ucca_parser.parse(word_idxs, pos_idxs, dep_idxs, passages)
        predicted.extend(pred_passages)
    return predicted


def visualize(t, save=True):
    for passage in t:
        width = len(passage.layer(layer0.LAYER_ID).all) * 19 / 27
        plt.figure(passage.ID, figsize=(width, width * 10 / 19))
        visualization.draw(passage, node_ids=False)
        plt.savefig(str(passage.ID) + ".png")
        plt.close()
        return str(passage.ID) + ".png"
