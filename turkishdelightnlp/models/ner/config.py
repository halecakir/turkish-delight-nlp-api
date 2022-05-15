import os
from datetime import datetime as dt

from .general_utils import get_logger
from .data_utils import get_trimmed_word_vectors, load_vocab, get_processing_word


class Config:
    def __init__(self, model_opts_path, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        self.replace_digits = False

        # general config
        self.dir_output = "results/test/"
        self.dir_model = self.dir_output + "model.weights/"
        self.path_log = self.dir_output + "log.txt"
        self.now_str = dt.now().strftime("%d%m%Y_%H%M%S")
        self.conll_eval = "eval/conlleval"
        self.conll_output = "results/conlleval{}.tmp".format(self.now_str)
        self.conll_score = "results/conllscore{}.tmp".format(self.now_str)

        # embeddings
        self.dim_word = 200
        self.dim_morph = 50
        self.dim_char = 30

        # ft, w2v, m2v or None (if you want to use multiple embeddings, provide
        # them comma separated e.g. "ft,m2v"
        self.use_pretrained = "ft,m2v,w2v"
        self.get_ft_vectors_cmd = (
            "http://127.0.0.1:8000/data/embeddings.bin " "< {} > {}"
        )

        # pretrained files
        self.filename_word2vec = f"{model_opts_path}/embeddings/tr-embeddings-w2v.txt"
        self.filename_fasttext = f"{model_opts_path}/embeddings/tr-embeddings-ft.txt"
        self.filename_morph2vec = f"{model_opts_path}/embeddings/tr-embeddings-m2v.txt"

        # trimmed embeddings (created from word2vec_filename with build_data.py)
        self.filename_trimmed_w2v = (
            f"{model_opts_path}/embeddings/emb.w2v.{self.dim_word}d.trimmed.npz"
        )
        self.filename_trimmed_ft = (
            f"{model_opts_path}/embeddings/emb.ft.{self.dim_word}d.trimmed.npz"
        )
        self.filename_trimmed_m2v = (
            f"{model_opts_path}/embeddings/emb.m2v.{self.dim_morph}d.trimmed.npz"
        )

        # dataset
        self.filename_dev = f"{model_opts_path}/dev.tmp"
        self.filename_test = f"{model_opts_path}/test.tmp"
        self.filename_train = f"{model_opts_path}/train.tmp"

        self.filename_dev2 = f"{model_opts_path}/dev2.tmp"
        self.filename_train2 = f"{model_opts_path}/train2.tmp"

        self.max_iter = None  # if not None, max number of examples in Dataset

        # vocab (created from dataset with build_data.py)
        self.filename_words = f"{model_opts_path}/words.tmp"
        self.filename_tags = f"{model_opts_path}/tags.tmp"
        self.filename_chars = f"{model_opts_path}/chars.tmp"

        # training
        self.train_embeddings = True
        self.nepochs = 100
        self.dropout = 0.5
        self.batch_size = 10
        self.lr_method = "sgd"
        self.lr = 0.005
        self.lr_decay = 1.0
        self.clip = 5.0  # if negative, no clipping
        self.nepoch_no_imprv = 999

        # model hyperparameters
        self.hidden_size_char = 30  # lstm on chars
        self.hidden_size_lstm = 250  # lstm on word embeddings

        # NOTE: if both chars and crf, only 1.6x slower on GPU
        self.use_crf = True  # if crf, training is 1.7x slower on CPU
        self.use_chars = "blstm"  # blstm, cnn or None
        self.use_ortho_char = True  # use orthographic chars instead of chars
        self.max_len_of_word = 20  # used only when use_chars = 'cnn'
        self.use_deasciification = False  # TODO

        # directory for training outputs
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # load if requested (default)
        if load:
            self.load()

    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed word2vec
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_chars = load_vocab(self.filename_chars)

        self.nwords = len(self.vocab_words)
        self.nchars = len(self.vocab_chars)
        self.ntags = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(
            self.vocab_words,
            self.vocab_chars,
            lowercase=False,
            chars=(self.use_chars is not None),
            use_ortho_char=self.use_ortho_char,
            replace_digits=self.replace_digits,
        )
        self.processing_tag = get_processing_word(
            self.vocab_tags, lowercase=False, allow_unk=False
        )

        # 3. get pre-trained embeddings
        self.embeddings_w2v = (
            get_trimmed_word_vectors(self.filename_trimmed_w2v)
            if "w2v" in self.use_pretrained
            else None
        )
        self.embeddings_ft = (
            get_trimmed_word_vectors(self.filename_trimmed_ft)
            if "ft" in self.use_pretrained
            else None
        )
        self.embeddings_m2v = (
            get_trimmed_word_vectors(self.filename_trimmed_m2v)
            if "m2v" in self.use_pretrained
            else None
        )
