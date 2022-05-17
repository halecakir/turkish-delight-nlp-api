import os

import torch

from ucca.convert import to_text, xml2passage
from ucca import textutil
from instance import Instance
from dataset import TensorDataSet
from ucca.convert import split2sentences

class Corpus(object):
    def __init__(self, passage):
        self.passages = [passage]
        self.instances = [Instance(passage) for passage in self.passages]

    @property
    def num_sentences(self):
        return len(self.passages)

    def __repr__(self):
        return "%s : %d sentences" % (self.dic_name, self.num_sentences)

    def __getitem(self, index):
        return self.passages[index]

    @staticmethod
    def read_sentences(path):
        sentences = []
        f = open(path, "r", encoding="utf8")
        for sentence in f.readlines():
            sentences.append(sentence.strip("\n"))
        return sentences
            
    @staticmethod
    def read_passages(path):
        passages = []
        for file in sorted(os.listdir(path)):
            file_path = os.path.join(path, file)
            if os.path.isdir(file_path):
                print(file_path)
            pas = xml2passage(file_path)
            passages.append(pas)
        return passages
    
    @staticmethod
    def read_passages_(path):
        passages = []
        pas = xml2passage(path)
        passages.append(pas)
        return passages

    def generate_inputs(self, vocab, is_training=False):
        word_idxs = []
        pos_idxs, dep_idxs = [], []
        trees, all_nodes, all_remote = [], [], []
        for instance in self.instances:
            _word_idxs = vocab.word2id([vocab.START] + instance.words + [vocab.STOP])
            _pos_idxs = vocab.pos2id([vocab.START] + instance.pos + [vocab.STOP])
            _dep_idxs = vocab.dep2id([vocab.START] + instance.dep + [vocab.STOP])

            word_idxs.append(torch.tensor(_word_idxs))
            pos_idxs.append(torch.tensor(_pos_idxs))
            dep_idxs.append(torch.tensor(_dep_idxs))
            trees.append([])
            all_nodes.append([])
            all_remote.append([])

        return TensorDataSet(
            word_idxs,
            pos_idxs,
            dep_idxs,
            self.passages,
            trees,
            all_nodes,
            all_remote,
        )


class Embedding(object):
    def __init__(self, words, vectors):
        super(Embedding, self).__init__()

        self.words = words
        self.vectors = vectors
        self.pretrained = {w: v for w, v in zip(words, vectors)}

    def __len__(self):
        return len(self.words)

    def __contains__(self, word):
        return word in self.pretrained

    def __getitem__(self, word):
        return self.pretrained[word]

    @property
    def dim(self):
        return len(self.vectors[0])

    @classmethod
    def load(cls, fname, smooth=True):
        with open(fname, 'r', encoding="utf8") as f:
            lines = [line for line in f]
        splits = [line.split() for line in lines[1:]]
        reprs = [(s[0], list(map(float, s[1:]))) for s in splits]
        words, vectors = map(list, zip(*reprs))
        vectors = torch.tensor(vectors)
        if smooth:
            vectors /= torch.std(vectors)
        embedding = cls(words, vectors)

        return embedding