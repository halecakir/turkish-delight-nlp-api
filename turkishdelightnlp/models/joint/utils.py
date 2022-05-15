# coding=utf-8
from collections import Counter
import re
import codecs
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText
import pickle
import numpy as np
import random

random.seed(1)
np.random.seed(1)


class ConllEntry:
    def __init__(
        self,
        id,
        form,
        lemma,
        pos,
        xpos,
        feats=None,
        parent_id=None,
        relation=None,
        deps=None,
        misc=None,
    ):
        self.id = id
        self.form = form
        self.norm = normalize(form)
        self.xpos = xpos
        self.pos = pos
        self.parent_id = parent_id
        self.relation = relation

        self.lemma = lemma
        self.feats = feats
        self.deps = deps
        self.misc = misc

        self.pred_parent_id = None
        self.pred_relation = None
        self.pred_pos = None
        self.pred_tags = None
        self.pred_tags_tokens = None
        self.idChars = []
        self.idMorphs = []
        self.idMorphTags = []
        self.decoder_gold_input = []

    def __str__(self):
        values = [
            str(self.id),
            self.form,
            self.lemma,
            self.pred_pos,
            self.xpos,
            "|".join(self.pred_tags_tokens[1:-1])
            if self.pred_tags_tokens is not None
            else self.feats,
            str(self.pred_parent_id) if self.pred_parent_id is not None else None,
            self.pred_relation,
            self.deps,
            self.misc,
        ]
        return "\t".join(["_" if v is None else v for v in values])


def vocab(conll_path, morph_dict):
    wordsCount = Counter()
    posCount = Counter()
    relCount = Counter()

    # Character vocabulary
    c2i = {}
    c2i["_UNK"] = 0  # unk char
    c2i["<w>"] = 1  # word start
    c2i["</w>"] = 2  # word end index
    c2i["NUM"] = 3
    c2i["EMAIL"] = 4
    c2i["URL"] = 5
    c2i["<start>"] = 6

    m2i = {}
    m2i["UNK"] = 0

    t2i = {}
    t2i["UNK"] = 0
    t2i["<s>"] = 1
    # Create morpheme tag indexes here. (CURSOR)

    root = ConllEntry(
        0, "*root*", "*root*", "ROOT-POS", "ROOT-CPOS", "_", -1, "rroot", "_", "_"
    )
    root.idChars = [1, 2]
    root.idMorphs = [0]
    tokens = [root]

    # create morpheme indexes out of morpheme dictionary
    all_morphs = []
    for word in morph_dict.keys():
        all_morphs += morph_dict[word]
    all_morphs = list(set(all_morphs))
    for idx in range(len(all_morphs)):
        m2i[all_morphs[idx]] = idx + 1

    for line in open(conll_path, "r"):
        tok = line.strip().split("\t")
        if not tok or line.strip() == "":
            if len(tokens) > 1:
                wordsCount.update(
                    [node.norm for node in tokens if isinstance(node, ConllEntry)]
                )
                posCount.update(
                    [node.pos for node in tokens if isinstance(node, ConllEntry)]
                )
                relCount.update(
                    [node.relation for node in tokens if isinstance(node, ConllEntry)]
                )
            tokens = [root]
        else:
            if line[0] == "#" or "-" in tok[0] or "." in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(
                    int(tok[0]),
                    tok[1],
                    tok[2],
                    tok[3],
                    tok[4],
                    tok[5],
                    int(tok[6]) if tok[6] != "_" else -1,
                    tok[7],
                    tok[8],
                    tok[9],
                )

                if entry.norm == "NUM":
                    entry.idChars = [1, 3, 2]
                elif entry.norm == "EMAIL":
                    entry.idChars = [1, 4, 2]
                elif entry.norm == "URL":
                    entry.idChars = [1, 5, 2]
                else:
                    chars_of_word = [1]
                    for char in tok[1]:
                        if char not in c2i:
                            c2i[char] = len(c2i)
                        chars_of_word.append(c2i[char])
                    chars_of_word.append(2)
                    entry.idChars = chars_of_word

                entry.idMorphs = get_morph_gold(entry.norm, morph_dict)

                # entry.idMorphTags = [0]
                for feat in entry.feats.split("|"):
                    if feat not in t2i:
                        t2i[feat] = len(t2i)

                tokens.append(entry)

    if len(tokens) > 1:
        wordsCount.update(
            [node.norm for node in tokens if isinstance(node, ConllEntry)]
        )
        posCount.update([node.pos for node in tokens if isinstance(node, ConllEntry)])
        relCount.update(
            [node.relation for node in tokens if isinstance(node, ConllEntry)]
        )

    return (
        wordsCount,
        {w: i for i, w in enumerate(list(wordsCount.keys()))},
        c2i,
        m2i,
        t2i,
        list(posCount.keys()),
        list(relCount.keys()),
    )


def read_conll(fh, c2i, m2i, t2i, morph_dict):
    # Character vocabulary
    root = ConllEntry(
        0, "*root*", "*root*", "ROOT-POS", "ROOT-CPOS", "_", -1, "rroot", "_", "_"
    )
    root.idChars = [1, 2]
    root.idMorphs = [1, 2]
    root.idMorphTags = [t2i["<s>"], t2i["<s>"]]
    tokens = [root]

    for line in fh:
        tok = line.strip().split("\t")
        if not tok or line.strip() == "":
            if len(tokens) > 1:
                yield tokens
            tokens = [root]
        else:
            if line[0] == "#" or "-" in tok[0] or "." in tok[0]:
                tokens.append(line.strip())
            else:
                entry = ConllEntry(
                    int(tok[0]),
                    tok[1],
                    tok[2],
                    tok[3],
                    tok[4],
                    tok[5],
                    int(tok[6]) if tok[6] != "_" else -1,
                    tok[7],
                    tok[8],
                    tok[9],
                )

                if entry.norm == "NUM":
                    entry.idChars = [1, 3, 2]
                elif entry.norm == "EMAIL":
                    entry.idChars = [1, 4, 2]
                elif entry.norm == "URL":
                    entry.idChars = [1, 5, 2]
                else:
                    if entry.norm == "”" or entry.norm == "’":
                        tok[1] = "''"
                        entry.norm = '"'
                    if entry.norm == "“" or entry.norm == "‘":
                        tok[1] = "``"
                        entry.norm = '"'
                    if "’" in entry.norm:
                        entry.norm = re.sub(r"’", "'", entry.norm)
                        tok[1] = entry.norm
                    if entry.norm == "—":
                        entry.norm = "-"
                        tok[1] = "-"

                    chars_of_word = [1]
                    for char in tok[1]:
                        if char in c2i:
                            chars_of_word.append(c2i[char])
                        else:
                            chars_of_word.append(0)
                    chars_of_word.append(2)
                    entry.idChars = chars_of_word

                entry.idMorphs = get_morph_gold(entry.norm, morph_dict)

                # Create morpheme tag gold data here! (CURSOR)
                # entry.idMorphTags = [0]
                feats_of_word = []
                for feat in entry.feats.split("|"):
                    if feat in t2i:
                        feats_of_word.append(t2i[feat])
                    else:
                        feats_of_word.append(t2i["UNK"])
                entry.idMorphTags = [t2i["<s>"]] + feats_of_word + [t2i["<s>"]]

                tokens.append(entry)

    if len(tokens) > 1:
        yield tokens


def convert_raw_to_conll(sentence, c2i, m2i, t2i, morph_dict):
    root = ConllEntry(
        0, "*root*", "*root*", "ROOT-POS", "ROOT-CPOS", "_", -1, "rroot", "_", "_"
    )
    root.idChars = [1, 2]
    root.idMorphs = [1, 2]
    root.idMorphTags = [t2i["<s>"], t2i["<s>"]]
    tokens = [root]
    splitted = sentence.strip().lower().split()
    for tid, t in enumerate(splitted):
        entry = ConllEntry(tid + 1, t, None, None, None)
        if entry.norm == "NUM":
            entry.idChars = [1, 3, 2]
        elif entry.norm == "EMAIL":
            entry.idChars = [1, 4, 2]
        elif entry.norm == "URL":
            entry.idChars = [1, 5, 2]
        else:
            if entry.norm == "”" or entry.norm == "’":
                t = "''"
                entry.norm = '"'
            if entry.norm == "“" or entry.norm == "‘":
                t = "``"
                entry.norm = '"'
            if "’" in entry.norm:
                entry.norm = re.sub(r"’", "'", entry.norm)
                t = entry.norm
            if entry.norm == "—":
                entry.norm = "-"
                t = "-"

            chars_of_word = [1]
            for char in t:
                if char in c2i:
                    chars_of_word.append(c2i[char])
                else:
                    chars_of_word.append(0)
            chars_of_word.append(2)
            entry.idChars = chars_of_word

        tokens.append(entry)
    return tokens


def write_conll(fn, conll_gen):
    with open(fn, "w") as fh:
        for sentence in conll_gen:
            for entry in sentence[1:]:
                fh.write(str(entry) + "\n")
            fh.write("\n")


numberRegex = re.compile("[0-9]+|[0-9]+\\.[0-9]+|[0-9]+[0-9,]+")


def normalize(word):
    if numberRegex.match(word):
        return "NUM"
    else:
        w = word.lower()
        w = re.sub(r".+@.+", "EMAIL", w)
        w = re.sub(r"(https?://|www\.).*", "URL", w)
        w = re.sub(r"``", '"', w)
        w = re.sub(r"''", '"', w)
        return w


try:
    import lzma
except ImportError:
    from backports import lzma


def load_embeddings_file(file_name, lower=False, type=None):
    if type is None:
        file_type = file_name.rsplit(".", 1)[1] if "." in file_name else None
        if file_type == "p":
            type = "pickle"
        elif file_type == "xz":
            type = "xz"
        elif file_type == "bin":
            type = "word2vec"
        elif file_type == "vec":
            type = "fasttext"
        elif file_name == "txt":
            type = "raw"
        else:
            type = "word2vec"

    if type == "word2vec":
        model = KeyedVectors.load_word2vec_format(
            file_name, binary=True, unicode_errors="ignore"
        )
        words = model.index2entity
    elif type == "fasttext":
        model = FastText.load_fasttext_format(file_name)
        words = [w for w in model.wv.vocab]
    elif type == "pickle":
        with open(file_name, "rb") as fp:
            model = pickle.load(fp)
        words = model.keys()
    elif type == "xz":
        open_func = codecs.open
        if file_name.endswith(".xz"):
            open_func = lzma.open
        else:
            open_func = codecs.open
        model = {}
        with open_func(file_name, "rb") as f:
            reader = codecs.getreader("utf-8")(f, errors="ignore")
            reader.readline()

            count = 0
            for line in reader:
                try:
                    fields = line.strip().split()
                    vec = [float(x) for x in fields[1:]]
                    word = fields[0]
                    if word not in model:
                        model[word] = vec
                except ValueError:
                    # print("Error converting: {}".format(line))
                    pass
        words = model.keys()
    elif type == "raw":
        model = {}
        with open(file_name) as target:
            for line in target:
                fields = line.strip().split()
                vec = [float(x) for x in fields[1:]]
                word = fields[0]
                if word not in model:
                    model[word] = vec
        words = model.keys()

    if lower:
        vectors = {word.lower(): model[word] for word in words}
    else:
        vectors = {word: model[word] for word in words}

    if "UNK" not in vectors:
        unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
        vectors["UNK"] = unk

    return vectors, len(vectors["UNK"])


def save_embeddings(file_name, vectors, type=None):
    if type is None:
        file_type = file_name.rsplit(".", 1)[1] if "." in file_name else None
        if file_type == "p":
            type = "pickle"
        elif file_type == "bin":
            type = "word2vec"
        else:
            type = "word2vec"

    if "UNK" not in vectors:
        unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
        vectors["UNK"] = unk

    if type == "word2vec":
        pass
    elif type == "pickle":
        with open(file_name, "wb") as fp:
            pickle.dump(vectors, fp, protocol=pickle.HIGHEST_PROTOCOL)


def get_morph_dict(segment_file, lowerCase=False):
    if segment_file == "N/A":
        return {}

    morph_dict = {}
    with open(segment_file, encoding="utf8") as text:
        for line in text:
            line = line.strip()
            index = line.split(":")[0].lower() if lowerCase else line.split(":")[0]
            data = line.split(":")[1].split("+")[0]
            if "-" in data:
                morph_dict[index] = data.split("-")
            else:
                morph_dict[index] = [data]
    return morph_dict


def generate_morphs(word, split_points):
    morphs = []
    morph = ""
    for i, split in enumerate(split_points):
        morph += word[i]
        if split > random.random():
            morphs.append(morph)
            morph = ""
    if len(morph) > 0:
        morphs.append(morph)
    return morphs


def get_morph_gold(word, morph_dict):
    split_points = []
    if word in morph_dict:
        for morph in morph_dict[word]:
            for i in range(len(morph)):
                if i == len(morph) - 1:
                    split_points.append(True)
                else:
                    split_points.append(False)
    else:
        for i in range(len(word)):
            if i == len(word) - 1:
                split_points.append(True)
            else:
                split_points.append(False)

    return split_points
