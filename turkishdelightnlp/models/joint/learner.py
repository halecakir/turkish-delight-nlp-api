# coding=utf-8
import random
import time
import numpy as np
from .mnnl import (
    FFSequencePredictor,
    Layer,
    RNNSequencePredictor,
)
from .decoder import *
from .utils import *
import dynet
import dynet_config

# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)


np.random.seed(1)


class jPosDepLearner:
    def __init__(self, vocab, pos, rels, w2i, c2i, m2i, t2i, morph_dict, options):
        self.model = dynet.ParameterCollection()
        random.seed(1)
        self.trainer = dynet.AdamTrainer(self.model)
        # if options.learning_rate is not None:
        #    self.trainer = AdamTrainer(self.model, alpha=options.learning_rate)
        #    print("Adam initial learning rate:", options.learning_rate)
        self.activations = {
            "tanh": dynet.tanh,
            "sigmoid": dynet.logistic,
            "relu": dynet.rectify,
            "tanh3": (
                lambda x: dynet.tanh(
                    dynet.cwise_multiply(dynet.cwise_multiply(x, x), x)
                )
            ),
        }
        self.activation = self.activations[options.activation]

        self.blstmFlag = options.blstmFlag
        self.labelsFlag = options.labelsFlag
        self.costaugFlag = options.costaugFlag
        self.bibiFlag = options.bibiFlag
        self.morphFlag = options.morphFlag
        self.goldMorphFlag = options.goldMorphFlag
        self.morphTagFlag = options.morphTagFlag
        self.goldMorphTagFlag = options.goldMorphTagFlag
        self.lowerCase = options.lowerCase
        self.mtag_encoding_composition_type = options.mtag_encoding_composition_type
        self.morph_encoding_composition_type = options.morph_encoding_composition_type
        self.encoding_composition_alpha = options.encoding_composition_alpha
        self.pos_encoding_composition_type = options.pos_encoding_composition_type

        self.ldims = options.lstm_dims
        self.wdims = options.wembedding_dims
        self.mdims = options.membedding_dims
        self.tdims = options.tembedding_dims
        self.cdims = options.cembedding_dims
        self.layers = options.lstm_layers
        self.wordsCount = vocab
        self.vocab = {word: ind + 3 for word, ind in iter(w2i.items())}
        self.pos = {word: ind for ind, word in enumerate(pos)}
        self.id2pos = {ind: word for ind, word in enumerate(pos)}
        self.c2i = c2i
        self.m2i = m2i
        self.t2i = t2i
        self.i2t = {t2i[i]: i for i in self.t2i}
        self.morph_dict = morph_dict
        self.rels = {word: ind for ind, word in enumerate(rels)}
        self.irels = rels
        self.pdims = options.pembedding_dims
        self.tagging_attention_size = options.tagging_att_size

        self.vocab["*PAD*"] = 1
        self.vocab["*INITIAL*"] = 2
        self.wlookup = self.model.add_lookup_parameters((len(vocab) + 3, self.wdims))
        self.clookup = self.model.add_lookup_parameters((len(c2i), self.cdims))
        self.plookup = self.model.add_lookup_parameters((len(pos), self.pdims))
        self.ext_embeddings = None

        if options.external_embedding is not None:
            print("External embeddding is loading...")
            ext_embeddings, ext_emb_dim = load_embeddings_file(
                options.external_embedding,
                lower=self.lowerCase,
                type=options.external_embedding_type,
            )
            assert ext_emb_dim == self.wdims
            print("Initializing word embeddings by pre-trained vectors")
            count = 0
            for word in self.vocab:
                if word in ext_embeddings:
                    count += 1
                    self.wlookup.init_row(self.vocab[word], ext_embeddings[word])
            self.ext_embeddings = ext_embeddings
            print(
                "Vocab size: %d; #words having pretrained vectors: %d"
                % (len(self.vocab), count)
            )

        self.morph_dims = 2 * 2 * self.mdims if self.morphFlag else 0
        self.mtag_dims = 2 * self.tdims if self.morphTagFlag else 0

        self.pos_builders = [
            dynet.VanillaLSTMBuilder(
                1,
                self.wdims + self.cdims * 2 + self.morph_dims + self.mtag_dims,
                self.ldims,
                self.model,
            ),
            dynet.VanillaLSTMBuilder(
                1,
                self.wdims + self.cdims * 2 + self.morph_dims + self.mtag_dims,
                self.ldims,
                self.model,
            ),
        ]
        self.pos_bbuilders = [
            dynet.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
            dynet.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
        ]

        if self.pos_encoding_composition_type != "None":
            self.pos_encoding_vertical_composition_w = self.model.add_parameters(
                (self.pdims, 2 * self.pdims)
            )
            self.pos_encoding_vertical_composition_b = self.model.add_parameters(
                (self.pdims)
            )

        if self.bibiFlag:
            self.builders = [
                dynet.VanillaLSTMBuilder(
                    1,
                    self.wdims
                    + self.cdims * 2
                    + self.morph_dims
                    + self.mtag_dims
                    + self.pdims,
                    self.ldims,
                    self.model,
                ),
                dynet.VanillaLSTMBuilder(
                    1,
                    self.wdims
                    + self.cdims * 2
                    + self.morph_dims
                    + self.mtag_dims
                    + self.pdims,
                    self.ldims,
                    self.model,
                ),
            ]
            self.bbuilders = [
                dynet.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
                dynet.VanillaLSTMBuilder(1, self.ldims * 2, self.ldims, self.model),
            ]
        elif self.layers > 0:
            self.builders = [
                dynet.VanillaLSTMBuilder(
                    self.layers,
                    self.wdims
                    + self.cdims * 2
                    + self.morph_dims
                    + self.mtag_dims
                    + self.pdims,
                    self.ldims,
                    self.model,
                ),
                dynet.VanillaLSTMBuilder(
                    self.layers,
                    self.wdims
                    + self.cdims * 2
                    + self.morph_dims
                    + self.mtag_dims
                    + self.pdims,
                    self.ldims,
                    self.model,
                ),
            ]
        else:
            self.builders = [
                dynet.SimpleRNNBuilder(
                    1,
                    self.wdims + self.cdims * 2 + self.morph_dims + self.mtag_dims,
                    self.ldims,
                    self.model,
                ),
                dynet.SimpleRNNBuilder(
                    1,
                    self.wdims + self.cdims * 2 + self.morph_dims + self.mtag_dims,
                    self.ldims,
                    self.model,
                ),
            ]

        self.ffSeqPredictor = FFSequencePredictor(
            Layer(self.model, self.ldims * 2, len(self.pos), dynet.softmax)
        )

        self.hidden_units = options.hidden_units

        self.hidBias = self.model.add_parameters((self.ldims * 8))
        self.hidLayer = self.model.add_parameters((self.hidden_units, self.ldims * 8))
        self.hid2Bias = self.model.add_parameters((self.hidden_units))

        self.outLayer = self.model.add_parameters(
            (1, self.hidden_units if self.hidden_units > 0 else self.ldims * 8)
        )

        if self.labelsFlag:
            self.rhidBias = self.model.add_parameters((self.ldims * 8))
            self.rhidLayer = self.model.add_parameters(
                (self.hidden_units, self.ldims * 8)
            )
            self.rhid2Bias = self.model.add_parameters((self.hidden_units))
            self.routLayer = self.model.add_parameters(
                (
                    len(self.irels),
                    self.hidden_units if self.hidden_units > 0 else self.ldims * 8,
                )
            )
            self.routBias = self.model.add_parameters((len(self.irels)))
            self.ffRelPredictor = FFSequencePredictor(
                Layer(
                    self.model,
                    self.hidden_units if self.hidden_units > 0 else self.ldims * 8,
                    len(self.irels),
                    dynet.softmax,
                )
            )

        self.char_rnn = RNNSequencePredictor(
            dynet.LSTMBuilder(1, self.cdims, self.cdims, self.model)
        )

        if self.morphFlag:
            self.seg_lstm = [
                dynet.VanillaLSTMBuilder(1, self.cdims, self.cdims, self.model),
                dynet.VanillaLSTMBuilder(1, self.cdims, self.cdims, self.model),
            ]
            self.seg_hidLayer = self.model.add_parameters((1, self.cdims * 2))
            self.slookup = self.model.add_lookup_parameters((len(self.c2i), self.cdims))

            self.char_lstm = [
                dynet.VanillaLSTMBuilder(1, self.cdims, self.mdims, self.model),
                dynet.VanillaLSTMBuilder(1, self.cdims, self.mdims, self.model),
            ]
            self.char_hidLayer = self.model.add_parameters((self.mdims, self.mdims * 2))
            self.mclookup = self.model.add_lookup_parameters(
                (len(self.c2i), self.cdims)
            )

            self.morph_lstm = [
                dynet.VanillaLSTMBuilder(1, self.mdims * 2, self.wdims, self.model),
                dynet.VanillaLSTMBuilder(1, self.mdims * 2, self.wdims, self.model),
            ]
            self.morph_hidLayer = self.model.add_parameters(
                (self.wdims, self.wdims * 2)
            )
            self.mlookup = self.model.add_lookup_parameters((len(m2i), self.mdims))

            self.morph_rnn = RNNSequencePredictor(
                dynet.LSTMBuilder(1, self.mdims * 2, self.mdims * 2, self.model)
            )
            if self.morph_encoding_composition_type != "None":
                self.morph_encoding_vertical_composition_w = self.model.add_parameters(
                    (self.tdims, 8 * self.mdims)
                )
                self.morph_encoding_vertical_composition_b = self.model.add_parameters(
                    (self.tdims)
                )

        if self.morphTagFlag:
            # All weights for morpheme taging will be here. (CURSOR)

            # Decoder
            self.dec_lstm = dynet.VanillaLSTMBuilder(
                1, 2 * self.cdims + self.tdims + self.cdims * 2, self.cdims, self.model
            )

            # Attention
            self.attention_w1 = self.model.add_parameters(
                (self.tagging_attention_size, self.cdims * 2)
            )
            self.attention_w2 = self.model.add_parameters(
                (self.tagging_attention_size, self.cdims * 2)
            )
            self.attention_v = self.model.add_parameters(
                (1, self.tagging_attention_size)
            )

            # Attention Context
            self.attention_w1_context = self.model.add_parameters(
                (self.tagging_attention_size, self.cdims * 2)
            )
            self.attention_w2_context = self.model.add_parameters(
                (self.tagging_attention_size, self.cdims * 2)
            )
            self.attention_v_context = self.model.add_parameters(
                (1, self.tagging_attention_size)
            )

            # MLP - Softmax
            self.decoder_w = self.model.add_parameters((len(t2i), self.cdims))
            self.decoder_b = self.model.add_parameters((len(t2i)))

            self.mtag_rnn = RNNSequencePredictor(
                dynet.VanillaLSTMBuilder(1, self.tdims, self.tdims, self.model)
            )
            self.tlookup = self.model.add_lookup_parameters((len(t2i), self.tdims))
            if self.mtag_encoding_composition_type != "None":
                self.mtag_encoding_vertical_composition_w = self.model.add_parameters(
                    (2 * self.tdims, 4 * self.tdims)
                )
                self.mtag_encoding_vertical_composition_b = self.model.add_parameters(
                    (2 * self.tdims)
                )

    def initialize(self):
        if self.morphFlag and self.ext_embeddings:
            print("Initializing word embeddings by morph2vec")
            count = 0
            for word in self.vocab:
                if word not in self.ext_embeddings and word in self.morph_dict:
                    morph_seg = self.morph_dict[word]

                    count += 1
                    self.wlookup.init_row(
                        self.vocab[word], self.__getWordVector(morph_seg).vec_value()
                    )
            print(
                "Vocab size: %d; #missing words having generated vectors: %d"
                % (len(self.vocab), count)
            )
            dynet.renew_cg()

    def __getExpr(self, sentence, i, j):

        if sentence[i].headfov is None:
            sentence[i].headfov = dynet.concatenate(
                [sentence[i].lstms[0], sentence[i].lstms[1]]
            )
        if sentence[j].modfov is None:
            sentence[j].modfov = dynet.concatenate(
                [sentence[j].lstms[0], sentence[j].lstms[1]]
            )

        _inputVector = dynet.concatenate(
            [
                sentence[i].headfov,
                sentence[j].modfov,
                dynet.abs(sentence[i].headfov - sentence[j].modfov),
                dynet.cmult(sentence[i].headfov, sentence[j].modfov),
            ]
        )

        if self.hidden_units > 0:
            output = self.outLayer.expr() * self.activation(
                self.hid2Bias.expr()
                + self.hidLayer.expr()
                * self.activation(_inputVector + self.hidBias.expr())
            )
        else:
            output = self.outLayer.expr() * self.activation(
                _inputVector + self.hidBias.expr()
            )

        return output

    def __evaluate(self, sentence):
        exprs = [
            [self.__getExpr(sentence, i, j) for j in range(len(sentence))]
            for i in range(len(sentence))
        ]
        scores = np.array(
            [[output.scalar_value() for output in exprsRow] for exprsRow in exprs]
        )

        return scores, exprs

    def pick_neg_log(self, pred, gold):
        return -dynet.log(dynet.pick(pred, gold))

    def binary_crossentropy(self, pred, gold):
        return dynet.binary_log_loss(pred, gold)

    def cosine_proximity(self, pred, gold):
        def l2_normalize(x):
            square_sum = dynet.sqrt(
                dynet.bmax(
                    dynet.sum_elems(dynet.square(x)),
                    np.finfo(float).eps * dynet.ones((1))[0],
                )
            )
            return dynet.cdiv(x, square_sum)

        y_true = l2_normalize(pred)
        y_pred = l2_normalize(gold)

        return -dynet.sum_elems(dynet.cmult(y_true, y_pred))

    def __getRelVector(self, sentence, i, j):
        if sentence[i].rheadfov is None:
            sentence[i].rheadfov = dynet.concatenate(
                [sentence[i].lstms[0], sentence[i].lstms[1]]
            )
        if sentence[j].rmodfov is None:
            sentence[j].rmodfov = dynet.concatenate(
                [sentence[j].lstms[0], sentence[j].lstms[1]]
            )
        _outputVector = dynet.concatenate(
            [
                sentence[i].rheadfov,
                sentence[j].rmodfov,
                dynet.abs(sentence[i].rheadfov - sentence[j].rmodfov),
                dynet.cmult(sentence[i].rheadfov, sentence[j].rmodfov),
            ]
        )

        if self.hidden_units > 0:
            return self.rhid2Bias.expr() + self.rhidLayer.expr() * self.activation(
                _outputVector + self.rhidBias.expr()
            )
        else:
            return _outputVector

    def __getSegmentationVector(self, word):
        slstm_forward = self.seg_lstm[0].initial_state()
        slstm_backward = self.seg_lstm[1].initial_state()

        seg_lstm_forward = slstm_forward.transduce(
            [self.slookup[self.c2i[char] if char in self.c2i else 0] for char in word]
        )
        seg_lstm_backward = slstm_backward.transduce(
            [
                self.slookup[self.c2i[char] if char in self.c2i else 0]
                for char in reversed(word)
            ]
        )

        seg_vec = []
        for seg, rev_seg in zip(seg_lstm_forward, reversed(seg_lstm_backward)):
            seg_vec.append(
                dynet.logistic(
                    self.seg_hidLayer.expr() * dynet.concatenate([seg, rev_seg])
                )
            )

        seg_vec = dynet.concatenate(seg_vec)

        return seg_vec

    def __getMorphVector(self, morph):
        clstm_forward = self.char_lstm[0].initial_state()
        clstm_backward = self.char_lstm[1].initial_state()

        char_lstm_forward = clstm_forward.transduce(
            [self.mclookup[self.c2i[char] if char in self.c2i else 0] for char in morph]
            if len(morph) > 0
            else [self.mclookup[0]]
        )[-1]
        char_lstm_backward = clstm_backward.transduce(
            [
                self.mclookup[self.c2i[char] if char in self.c2i else 0]
                for char in reversed(morph)
            ]
            if len(morph) > 0
            else [self.mclookup[0]]
        )[-1]

        char_emb = self.char_hidLayer.expr() * dynet.concatenate(
            [char_lstm_forward, char_lstm_backward]
        )

        return dynet.concatenate(
            [self.mlookup[self.m2i[morph] if morph in self.m2i else 0], char_emb]
        )

    def __getWordVector(self, morph_seg):
        mlstm_forward = self.morph_lstm[0].initial_state()
        mlstm_backward = self.morph_lstm[1].initial_state()

        morph_lstm_forward = mlstm_forward.transduce(
            [self.__getMorphVector(morph) for morph in morph_seg]
        )[-1]
        morph_lstm_backward = mlstm_backward.transduce(
            [self.__getMorphVector(morph) for morph in reversed(morph_seg)]
        )[-1]

        morph_enc = dynet.concatenate([morph_lstm_forward, morph_lstm_backward])
        word_vec = self.morph_hidLayer.expr() * morph_enc

        return word_vec

    def attend(self, input_mat, state, w1dt):
        w2 = dynet.parameter(self.attention_w2)
        v = dynet.parameter(self.attention_v)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim,1)
        w2dt = w2 * dynet.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        # unnormalized: (seqlen,)
        unnormalized = dynet.transpose(v * dynet.tanh(dynet.colwise_add(w1dt, w2dt)))
        att_weights = dynet.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context

    def attend_context(self, input_mat, state, w1dt_context):
        w2_context = dynet.parameter(self.attention_w2_context)
        v_context = dynet.parameter(self.attention_v_context)

        # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
        # w1dt: (attdim x seqlen)
        # w2dt: (attdim,1)
        w2dt_context = w2_context * dynet.concatenate(list(state.s()))
        # att_weights: (seqlen,) row vector
        # unnormalized: (seqlen,)
        unnormalized = dynet.transpose(
            v_context * dynet.tanh(dynet.colwise_add(w1dt_context, w2dt_context))
        )
        att_weights = dynet.softmax(unnormalized)
        # context: (encoder_state)
        context = input_mat * att_weights
        return context

    def decode(self, vectors, decoder_seq, word_context):
        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        w1 = dynet.parameter(self.attention_w1)

        w1_context = dynet.parameter(self.attention_w1_context)
        input_mat = dynet.concatenate_cols(vectors)
        input_context = dynet.concatenate_cols(word_context)

        w1dt = None
        w1dt_context = None

        last_output_embeddings = self.tlookup[self.t2i["<s>"]]
        s = self.dec_lstm.initial_state().add_input(
            dynet.concatenate(
                [
                    dynet.vecInput(self.cdims * 2),
                    last_output_embeddings,
                    dynet.vecInput(self.cdims * 2),
                ]
            )
        )
        loss = []

        for char in decoder_seq:
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            w1dt_context = w1dt_context or w1_context * input_context
            vector = dynet.concatenate(
                [
                    self.attend(input_mat, s, w1dt),
                    last_output_embeddings,
                    self.attend_context(input_context, s, w1dt_context),
                ]
            )
            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dynet.softmax(out_vector)
            last_output_embeddings = self.tlookup[char]
            loss.append(-dynet.log(dynet.pick(probs, char)))
        loss = dynet.esum(loss)
        return loss

    def __getLossMorphTagging(self, all_encoded_states, decoder_gold, word_context):
        return self.decode(all_encoded_states, decoder_gold, word_context)

    def generate(self, encoded, word_context):
        w = dynet.parameter(self.decoder_w)
        b = dynet.parameter(self.decoder_b)
        w1 = dynet.parameter(self.attention_w1)

        w1_context = dynet.parameter(self.attention_w1_context)

        input_mat = dynet.concatenate_cols(encoded)
        input_context = dynet.concatenate_cols(word_context)

        w1dt = None
        w1dt_context = None

        last_output_embeddings = self.tlookup[self.t2i["<s>"]]
        s = self.dec_lstm.initial_state().add_input(
            dynet.concatenate(
                [
                    dynet.vecInput(self.cdims * 2),
                    last_output_embeddings,
                    dynet.vecInput(self.cdims * 2),
                ]
            )
        )

        out = []
        count_EOS = 0
        limit_features = 10
        for i in range(limit_features):
            if count_EOS == 2:
                break
            # w1dt can be computed and cached once for the entire decoding phase
            w1dt = w1dt or w1 * input_mat
            w1dt_context = w1dt_context or w1_context * input_context
            vector = dynet.concatenate(
                [
                    self.attend(input_mat, s, w1dt),
                    last_output_embeddings,
                    self.attend_context(input_context, s, w1dt_context),
                ]
            )

            s = s.add_input(vector)
            out_vector = w * s.output() + b
            probs = dynet.softmax(out_vector).vec_value()
            next_char = probs.index(max(probs))
            last_output_embeddings = self.tlookup[next_char]
            if next_char == self.t2i["<s>"]:
                count_EOS += 1
            out.append(next_char)
        return out

    def Save(self, filename):
        self.model.save(filename)

    def Load(self, filename):
        self.model.populate(filename)

    def predict_sentence(self, raw_sentence):
        dynet.renew_cg()
        conll_sentence = convert_raw_to_conll(
            raw_sentence, self.c2i, self.m2i, self.t2i, self.morph_dict
        )
        if self.morphTagFlag:
            sentence_context = []
            last_state_char = self.char_rnn.predict_sequence(
                [self.clookup[self.c2i["<start>"]]]
            )[-1]
            rev_last_state_char = self.char_rnn.predict_sequence(
                [self.clookup[self.c2i["<start>"]]]
            )[-1]
            sentence_context.append(
                dynet.concatenate([last_state_char, rev_last_state_char])
            )
            for entry in conll_sentence:
                last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in entry.idChars]
                )
                rev_last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in reversed(entry.idChars)]
                )
                entry.char_rnn_states = [
                    dynet.concatenate([f, b])
                    for f, b in zip(last_state_char, rev_last_state_char)
                ]
                sentence_context.append(entry.char_rnn_states[-1])

        prev_encoding_mtag = None
        prev_encoding_morph = None
        for idx, entry in enumerate(conll_sentence):
            wordvec = (
                self.wlookup[int(self.vocab.get(entry.norm, 0))]
                if self.wdims > 0
                else None
            )

            if self.morphTagFlag:
                entry.vec = dynet.concatenate([wordvec, entry.char_rnn_states[-1]])
            else:
                last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in entry.idChars]
                )[-1]
                rev_last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in reversed(entry.idChars)]
                )[-1]
                entry.vec = dynet.concatenate(
                    [wordvec, last_state_char, rev_last_state_char]
                )

            if self.morphFlag:
                if len(entry.norm) > 2:
                    if self.goldMorphFlag:
                        seg_vec = self.__getSegmentationVector(entry.norm)
                        seg_vec = dynet.vecInput(seg_vec.dim()[0][0])
                        seg_vec.set(entry.idMorphs)
                        morph_seg = generate_morphs(entry.norm, seg_vec.vec_value())
                        entry.pred_seg = morph_seg
                    else:
                        seg_vec = self.__getSegmentationVector(entry.norm)
                        morph_seg = generate_morphs(entry.norm, seg_vec.vec_value())
                        entry.pred_seg = seg_vec.vec_value()
                else:
                    morph_seg = [entry.norm]
                    entry.pred_seg = entry.idMorphs

                entry.seg = entry.idMorphs

                last_state_morph = self.morph_rnn.predict_sequence(
                    [self.__getMorphVector(morph) for morph in morph_seg]
                )[-1]
                rev_last_state_morph = self.morph_rnn.predict_sequence(
                    [self.__getMorphVector(morph) for morph in reversed(morph_seg)]
                )[-1]

                if self.morph_encoding_composition_type == "w_sum":
                    current_encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )
                    if prev_encoding_morph:
                        encoding_morph = (
                            prev_encoding_morph * self.encoding_composition_alpha
                            + current_encoding_morph
                            * (1 - self.encoding_composition_alpha)
                        )
                    else:
                        encoding_morph = current_encoding_morph
                    prev_encoding_morph = current_encoding_morph
                elif self.morph_encoding_composition_type == "cwise_mult":
                    current_encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )
                    if prev_encoding_morph:
                        encoding_mtag = dynet.cmult(
                            prev_encoding_morph, current_encoding_morph
                        )
                    else:
                        encoding_morph = current_encoding_morph
                elif self.morph_encoding_composition_type == "mlp":
                    current_encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )
                    if prev_encoding_morph:
                        morph_w = dynet.parameter(
                            self.morph_encoding_vertical_composition_w
                        )
                        morhp_b = dynet.parameter(
                            self.morph_encoding_vertical_composition_b
                        )
                        encoding_mtag = (
                            morph_w
                            * dynet.concatenate(
                                [current_encoding_morph, prev_encoding_morph]
                            )
                            + morhp_b
                        )
                    else:
                        encoding_morph = current_encoding_morph
                else:
                    encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )

                entry.vec = dynet.concatenate([entry.vec, encoding_morph])

            if self.morphTagFlag:
                if self.goldMorphTagFlag:
                    morph_tags = entry.idMorphTags
                    entry.pred_tags = entry.idMorphTags
                    entry.pred_tags_tokens = [
                        self.i2t[m_tag_id] for m_tag_id in entry.pred_tags
                    ]
                else:
                    word_context = [
                        c for i, c in enumerate(sentence_context) if i - 1 != idx
                    ]
                    entry.pred_tags = self.generate(entry.char_rnn_states, word_context)
                    morph_tags = entry.pred_tags
                    entry.tags = entry.idMorphTags
                    entry.pred_tags_tokens = [
                        self.i2t[m_tag_id] for m_tag_id in entry.pred_tags
                    ]

                last_state_mtag = self.mtag_rnn.predict_sequence(
                    [self.tlookup[t] for t in morph_tags]
                )[-1]
                rev_last_state_mtag = self.mtag_rnn.predict_sequence(
                    [self.tlookup[t] for t in reversed(morph_tags)]
                )[-1]

                if self.mtag_encoding_composition_type == "w_sum":
                    current_encoding_mtag = dynet.concatenate(
                        [last_state_mtag, rev_last_state_mtag]
                    )
                    if prev_encoding_mtag:
                        encoding_mtag = (
                            prev_encoding_mtag * self.encoding_composition_alpha
                            + current_encoding_mtag
                            * (1 - self.encoding_composition_alpha)
                        )
                    else:
                        encoding_mtag = current_encoding_mtag
                    prev_encoding_mtag = current_encoding_mtag
                elif self.mtag_encoding_composition_type == "cwise_mult":
                    current_encoding_mtag = dynet.concatenate(
                        [last_state_mtag, rev_last_state_mtag]
                    )
                    if prev_encoding_mtag:
                        encoding_mtag = dynet.cmult(
                            prev_encoding_mtag, current_encoding_mtag
                        )
                    else:
                        encoding_mtag = current_encoding_mtag
                    prev_encoding_mtag = current_encoding_mtag
                elif self.mtag_encoding_composition_type == "mlp":
                    current_encoding_mtag = dynet.concatenate(
                        [last_state_mtag, rev_last_state_mtag]
                    )
                    if prev_encoding_mtag:
                        mtag_w = dynet.parameter(
                            self.mtag_encoding_vertical_composition_w
                        )
                        mtag_b = dynet.parameter(
                            self.mtag_encoding_vertical_composition_b
                        )
                        encoding_mtag = (
                            mtag_w
                            * dynet.concatenate(
                                [current_encoding_mtag, prev_encoding_mtag]
                            )
                            + mtag_b
                        )
                    else:
                        encoding_mtag = current_encoding_mtag
                    prev_encoding_mtag = current_encoding_mtag
                else:
                    encoding_mtag = dynet.concatenate(
                        [last_state_mtag, rev_last_state_mtag]
                    )

                entry.vec = dynet.concatenate([entry.vec, encoding_mtag])

            entry.pos_lstms = [entry.vec, entry.vec]
            entry.headfov = None
            entry.modfov = None

            entry.rheadfov = None
            entry.rmodfov = None

        # Predicted pos tags
        lstm_forward = self.pos_builders[0].initial_state()
        lstm_backward = self.pos_builders[1].initial_state()
        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
            lstm_forward = lstm_forward.add_input(entry.vec)
            lstm_backward = lstm_backward.add_input(rentry.vec)

            entry.pos_lstms[1] = lstm_forward.output()
            rentry.pos_lstms[0] = lstm_backward.output()

        for entry in conll_sentence:
            entry.pos_vec = dynet.concatenate(entry.pos_lstms)

        blstm_forward = self.pos_bbuilders[0].initial_state()
        blstm_backward = self.pos_bbuilders[1].initial_state()

        for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
            blstm_forward = blstm_forward.add_input(entry.pos_vec)
            blstm_backward = blstm_backward.add_input(rentry.pos_vec)
            entry.pos_lstms[1] = blstm_forward.output()
            rentry.pos_lstms[0] = blstm_backward.output()

        concat_layer = [dynet.concatenate(entry.pos_lstms) for entry in conll_sentence]
        outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
        predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]
        predicted_postags = [self.id2pos[idx] for idx in predicted_pos_indices]

        # Add predicted pos tags
        prev_encoding_pos = None
        for entry, posid in zip(conll_sentence, predicted_pos_indices):
            if self.pos_encoding_composition_type == "w_sum":
                current_encoding_pos = self.plookup[posid]
                if prev_encoding_pos:
                    encoding_pos = (
                        prev_encoding_pos * self.encoding_composition_alpha
                        + current_encoding_pos * (1 - self.encoding_composition_alpha)
                    )
                else:
                    encoding_pos = current_encoding_pos
                prev_encoding_pos = current_encoding_pos
            elif self.pos_encoding_composition_type == "cwise_mult":
                current_encoding_pos = self.plookup[posid]
                if prev_encoding_pos:
                    encoding_mtag = dynet.cmult(prev_encoding_pos, current_encoding_pos)
                else:
                    encoding_pos = current_encoding_pos
                prev_encoding_pos = current_encoding_pos
            elif self.pos_encoding_composition_type == "mlp":
                current_encoding_pos = self.plookup[posid]
                if prev_encoding_pos:
                    pos_w = dynet.parameter(self.pos_encoding_vertical_composition_w)
                    pos_b = dynet.parameter(self.pos_encoding_vertical_composition_b)
                    encoding_pos = (
                        pos_w
                        * dynet.concatenate([current_encoding_pos, prev_encoding_pos])
                        + pos_b
                    )
                else:
                    encoding_pos = current_encoding_pos
                prev_encoding_pos = current_encoding_pos
            else:
                encoding_pos = self.plookup[posid]

            entry.vec = dynet.concatenate([entry.vec, encoding_pos])
            entry.lstms = [entry.vec, entry.vec]

        if self.blstmFlag:
            lstm_forward = self.builders[0].initial_state()
            lstm_backward = self.builders[1].initial_state()

            for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                lstm_forward = lstm_forward.add_input(entry.vec)
                lstm_backward = lstm_backward.add_input(rentry.vec)

                entry.lstms[1] = lstm_forward.output()
                rentry.lstms[0] = lstm_backward.output()

            if self.bibiFlag:
                for entry in conll_sentence:
                    entry.vec = dynet.concatenate(entry.lstms)

                blstm_forward = self.bbuilders[0].initial_state()
                blstm_backward = self.bbuilders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.vec)
                    blstm_backward = blstm_backward.add_input(rentry.vec)

                    entry.lstms[1] = blstm_forward.output()
                    rentry.lstms[0] = blstm_backward.output()

        scores, exprs = self.__evaluate(conll_sentence)
        heads = parse_proj(scores)

        # Multiple roots: heading to the previous "rooted" one
        rootCount = 0
        rootWid = -1
        for index, head in enumerate(heads):
            if head == 0:
                rootCount += 1
                if rootCount == 1:
                    rootWid = index
                if rootCount > 1:
                    heads[index] = rootWid
                    rootWid = index

        for entry, head, pos in zip(conll_sentence, heads, predicted_postags):
            entry.pred_parent_id = head
            entry.pred_relation = "_"
            entry.pred_pos = pos

        dump = False

        if self.labelsFlag:
            concat_layer = [
                self.__getRelVector(conll_sentence, head, modifier + 1)
                for modifier, head in enumerate(heads[1:])
            ]
            outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
            predicted_rel_indices = [np.argmax(o.value()) for o in outputFFlayer]
            predicted_rels = [self.irels[idx] for idx in predicted_rel_indices]
            for modifier, head in enumerate(heads[1:]):
                conll_sentence[modifier + 1].pred_relation = predicted_rels[modifier]

        dynet.renew_cg()
        if not dump:
            return conll_sentence

    def predict_morphemes(self, raw_sentence):
        dynet.renew_cg()
        conll_sentence = convert_raw_to_conll(
            raw_sentence, self.c2i, self.m2i, self.t2i, self.morph_dict
        )
        if self.morphTagFlag:
            sentence_context = []
            last_state_char = self.char_rnn.predict_sequence(
                [self.clookup[self.c2i["<start>"]]]
            )[-1]
            rev_last_state_char = self.char_rnn.predict_sequence(
                [self.clookup[self.c2i["<start>"]]]
            )[-1]
            sentence_context.append(
                dynet.concatenate([last_state_char, rev_last_state_char])
            )
            for entry in conll_sentence:
                last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in entry.idChars]
                )
                rev_last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in reversed(entry.idChars)]
                )
                entry.char_rnn_states = [
                    dynet.concatenate([f, b])
                    for f, b in zip(last_state_char, rev_last_state_char)
                ]
                sentence_context.append(entry.char_rnn_states[-1])

        prev_encoding_mtag = None
        prev_encoding_morph = None
        morph_seg = []
        tokens = []
        for idx, entry in enumerate(conll_sentence):
            wordvec = (
                self.wlookup[int(self.vocab.get(entry.norm, 0))]
                if self.wdims > 0
                else None
            )
            tokens.append(entry.form)
            if self.morphTagFlag:
                entry.vec = dynet.concatenate([wordvec, entry.char_rnn_states[-1]])
            else:
                last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in entry.idChars]
                )[-1]
                rev_last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in reversed(entry.idChars)]
                )[-1]
                entry.vec = dynet.concatenate(
                    [wordvec, last_state_char, rev_last_state_char]
                )

            if self.morphFlag:
                if len(entry.norm) > 2:
                    if self.goldMorphFlag:
                        seg_vec = self.__getSegmentationVector(entry.norm)
                        seg_vec = dynet.vecInput(seg_vec.dim()[0][0])
                        seg_vec.set(entry.idMorphs)
                        morph_seg.append(
                            generate_morphs(entry.norm, seg_vec.vec_value())
                        )
                        entry.pred_seg = morph_seg
                    else:
                        seg_vec = self.__getSegmentationVector(entry.norm)
                        morph_seg.append(
                            generate_morphs(entry.norm, seg_vec.vec_value())
                        )
                        entry.pred_seg = seg_vec.vec_value()
                else:
                    morph_seg.append([entry.norm])
                    entry.pred_seg = entry.idMorphs
        dynet.renew_cg()
        return tokens, morph_seg

    def predict_morpheme_tags(self, raw_sentence):
        dynet.renew_cg()
        conll_sentence = convert_raw_to_conll(
            raw_sentence, self.c2i, self.m2i, self.t2i, self.morph_dict
        )
        if self.morphTagFlag:
            sentence_context = []
            last_state_char = self.char_rnn.predict_sequence(
                [self.clookup[self.c2i["<start>"]]]
            )[-1]
            rev_last_state_char = self.char_rnn.predict_sequence(
                [self.clookup[self.c2i["<start>"]]]
            )[-1]
            sentence_context.append(
                dynet.concatenate([last_state_char, rev_last_state_char])
            )
            for entry in conll_sentence:
                last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in entry.idChars]
                )
                rev_last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in reversed(entry.idChars)]
                )
                entry.char_rnn_states = [
                    dynet.concatenate([f, b])
                    for f, b in zip(last_state_char, rev_last_state_char)
                ]
                sentence_context.append(entry.char_rnn_states[-1])

        prev_encoding_mtag = None
        prev_encoding_morph = None
        morph_tags_return = []
        tokens = []
        for idx, entry in enumerate(conll_sentence):
            wordvec = (
                self.wlookup[int(self.vocab.get(entry.norm, 0))]
                if self.wdims > 0
                else None
            )
            tokens.append(entry.form)
            if self.morphTagFlag:
                entry.vec = dynet.concatenate([wordvec, entry.char_rnn_states[-1]])
            else:
                last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in entry.idChars]
                )[-1]
                rev_last_state_char = self.char_rnn.predict_sequence(
                    [self.clookup[c] for c in reversed(entry.idChars)]
                )[-1]
                entry.vec = dynet.concatenate(
                    [wordvec, last_state_char, rev_last_state_char]
                )

            if self.morphFlag:
                if len(entry.norm) > 2:
                    if self.goldMorphFlag:
                        seg_vec = self.__getSegmentationVector(entry.norm)
                        seg_vec = dynet.vecInput(seg_vec.dim()[0][0])
                        seg_vec.set(entry.idMorphs)
                        morph_seg = generate_morphs(entry.norm, seg_vec.vec_value())
                        entry.pred_seg = morph_seg
                    else:
                        seg_vec = self.__getSegmentationVector(entry.norm)
                        morph_seg = generate_morphs(entry.norm, seg_vec.vec_value())
                        entry.pred_seg = seg_vec.vec_value()
                else:
                    morph_seg = [entry.norm]
                    entry.pred_seg = entry.idMorphs

                entry.seg = entry.idMorphs

                last_state_morph = self.morph_rnn.predict_sequence(
                    [self.__getMorphVector(morph) for morph in morph_seg]
                )[-1]
                rev_last_state_morph = self.morph_rnn.predict_sequence(
                    [self.__getMorphVector(morph) for morph in reversed(morph_seg)]
                )[-1]

                if self.morph_encoding_composition_type == "w_sum":
                    current_encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )
                    if prev_encoding_morph:
                        encoding_morph = (
                            prev_encoding_morph * self.encoding_composition_alpha
                            + current_encoding_morph
                            * (1 - self.encoding_composition_alpha)
                        )
                    else:
                        encoding_morph = current_encoding_morph
                    prev_encoding_morph = current_encoding_morph
                elif self.morph_encoding_composition_type == "cwise_mult":
                    current_encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )
                    if prev_encoding_morph:
                        encoding_mtag = dynet.cmult(
                            prev_encoding_morph, current_encoding_morph
                        )
                    else:
                        encoding_morph = current_encoding_morph
                elif self.morph_encoding_composition_type == "mlp":
                    current_encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )
                    if prev_encoding_morph:
                        morph_w = dynet.parameter(
                            self.morph_encoding_vertical_composition_w
                        )
                        morhp_b = dynet.parameter(
                            self.morph_encoding_vertical_composition_b
                        )
                        encoding_mtag = (
                            morph_w
                            * dynet.concatenate(
                                [current_encoding_morph, prev_encoding_morph]
                            )
                            + morhp_b
                        )
                    else:
                        encoding_morph = current_encoding_morph
                else:
                    encoding_morph = dynet.concatenate(
                        [last_state_morph, rev_last_state_morph]
                    )

                entry.vec = dynet.concatenate([entry.vec, encoding_morph])

            if self.morphTagFlag:
                if self.goldMorphTagFlag:
                    morph_tags = entry.idMorphTags
                    entry.pred_tags = entry.idMorphTags
                    entry.pred_tags_tokens = [
                        self.i2t[m_tag_id] for m_tag_id in entry.pred_tags
                    ]
                    morph_tags_return.append(entry.pred_tags_tokens)
                else:
                    word_context = [
                        c for i, c in enumerate(sentence_context) if i - 1 != idx
                    ]
                    entry.pred_tags = self.generate(entry.char_rnn_states, word_context)
                    morph_tags = entry.pred_tags
                    entry.tags = entry.idMorphTags
                    entry.pred_tags_tokens = [
                        self.i2t[m_tag_id] for m_tag_id in entry.pred_tags
                    ]
                    morph_tags_return.append(entry.pred_tags_tokens)
        dynet.renew_cg()
        return tokens, morph_tags_return

    def Predict(self, conll_path):
        print("Predicting...")
        with open(conll_path, "r") as conllFP:
            for iSentence, sentence in enumerate(
                read_conll(conllFP, self.c2i, self.m2i, self.t2i, self.morph_dict)
            ):
                conll_sentence = [
                    entry for entry in sentence if isinstance(entry, ConllEntry)
                ]

                if self.morphTagFlag:
                    sentence_context = []
                    last_state_char = self.char_rnn.predict_sequence(
                        [self.clookup[self.c2i["<start>"]]]
                    )[-1]
                    rev_last_state_char = self.char_rnn.predict_sequence(
                        [self.clookup[self.c2i["<start>"]]]
                    )[-1]
                    sentence_context.append(
                        dynet.concatenate([last_state_char, rev_last_state_char])
                    )
                    for entry in conll_sentence:
                        last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in entry.idChars]
                        )
                        rev_last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in reversed(entry.idChars)]
                        )
                        entry.char_rnn_states = [
                            dynet.concatenate([f, b])
                            for f, b in zip(last_state_char, rev_last_state_char)
                        ]
                        sentence_context.append(entry.char_rnn_states[-1])

                prev_encoding_mtag = None
                prev_encoding_morph = None
                for idx, entry in enumerate(conll_sentence):
                    wordvec = (
                        self.wlookup[int(self.vocab.get(entry.norm, 0))]
                        if self.wdims > 0
                        else None
                    )

                    if self.morphTagFlag:
                        entry.vec = dynet.concatenate(
                            [wordvec, entry.char_rnn_states[-1]]
                        )
                    else:
                        last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in entry.idChars]
                        )[-1]
                        rev_last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in reversed(entry.idChars)]
                        )[-1]
                        entry.vec = dynet.concatenate(
                            [wordvec, last_state_char, rev_last_state_char]
                        )

                    if self.morphFlag:
                        if len(entry.norm) > 2:
                            if self.goldMorphFlag:
                                seg_vec = self.__getSegmentationVector(entry.norm)
                                seg_vec = dynet.vecInput(seg_vec.dim()[0][0])
                                seg_vec.set(entry.idMorphs)
                                morph_seg = generate_morphs(
                                    entry.norm, seg_vec.vec_value()
                                )
                                entry.pred_seg = morph_seg
                            else:
                                seg_vec = self.__getSegmentationVector(entry.norm)
                                morph_seg = generate_morphs(
                                    entry.norm, seg_vec.vec_value()
                                )
                                entry.pred_seg = seg_vec.vec_value()
                        else:
                            morph_seg = [entry.norm]
                            entry.pred_seg = entry.idMorphs

                        entry.seg = entry.idMorphs

                        last_state_morph = self.morph_rnn.predict_sequence(
                            [self.__getMorphVector(morph) for morph in morph_seg]
                        )[-1]
                        rev_last_state_morph = self.morph_rnn.predict_sequence(
                            [
                                self.__getMorphVector(morph)
                                for morph in reversed(morph_seg)
                            ]
                        )[-1]

                        if self.morph_encoding_composition_type == "w_sum":
                            current_encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )
                            if prev_encoding_morph:
                                encoding_morph = (
                                    prev_encoding_morph
                                    * self.encoding_composition_alpha
                                    + current_encoding_morph
                                    * (1 - self.encoding_composition_alpha)
                                )
                            else:
                                encoding_morph = current_encoding_morph
                            prev_encoding_morph = current_encoding_morph
                        elif self.morph_encoding_composition_type == "cwise_mult":
                            current_encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )
                            if prev_encoding_morph:
                                encoding_mtag = dynet.cmult(
                                    prev_encoding_morph, current_encoding_morph
                                )
                            else:
                                encoding_morph = current_encoding_morph
                        elif self.morph_encoding_composition_type == "mlp":
                            current_encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )
                            if prev_encoding_morph:
                                morph_w = dynet.parameter(
                                    self.morph_encoding_vertical_composition_w
                                )
                                morhp_b = dynet.parameter(
                                    self.morph_encoding_vertical_composition_b
                                )
                                encoding_mtag = (
                                    morph_w
                                    * dynet.concatenate(
                                        [current_encoding_morph, prev_encoding_morph]
                                    )
                                    + morhp_b
                                )
                            else:
                                encoding_morph = current_encoding_morph
                        else:
                            encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )

                        entry.vec = dynet.concatenate([entry.vec, encoding_morph])

                    if self.morphTagFlag:
                        if self.goldMorphTagFlag:
                            morph_tags = entry.idMorphTags
                            entry.pred_tags = entry.idMorphTags
                            entry.pred_tags_tokens = [
                                self.i2t[m_tag_id] for m_tag_id in entry.pred_tags
                            ]
                        else:
                            word_context = [
                                c
                                for i, c in enumerate(sentence_context)
                                if i - 1 != idx
                            ]
                            entry.pred_tags = self.generate(
                                entry.char_rnn_states, word_context
                            )
                            morph_tags = entry.pred_tags
                            entry.tags = entry.idMorphTags
                            entry.pred_tags_tokens = [
                                self.i2t[m_tag_id] for m_tag_id in entry.pred_tags
                            ]

                        last_state_mtag = self.mtag_rnn.predict_sequence(
                            [self.tlookup[t] for t in morph_tags]
                        )[-1]
                        rev_last_state_mtag = self.mtag_rnn.predict_sequence(
                            [self.tlookup[t] for t in reversed(morph_tags)]
                        )[-1]

                        if self.mtag_encoding_composition_type == "w_sum":
                            current_encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )
                            if prev_encoding_mtag:
                                encoding_mtag = (
                                    prev_encoding_mtag * self.encoding_composition_alpha
                                    + current_encoding_mtag
                                    * (1 - self.encoding_composition_alpha)
                                )
                            else:
                                encoding_mtag = current_encoding_mtag
                            prev_encoding_mtag = current_encoding_mtag
                        elif self.mtag_encoding_composition_type == "cwise_mult":
                            current_encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )
                            if prev_encoding_mtag:
                                encoding_mtag = dynet.cmult(
                                    prev_encoding_mtag, current_encoding_mtag
                                )
                            else:
                                encoding_mtag = current_encoding_mtag
                            prev_encoding_mtag = current_encoding_mtag
                        elif self.mtag_encoding_composition_type == "mlp":
                            current_encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )
                            if prev_encoding_mtag:
                                mtag_w = dynet.parameter(
                                    self.mtag_encoding_vertical_composition_w
                                )
                                mtag_b = dynet.parameter(
                                    self.mtag_encoding_vertical_composition_b
                                )
                                encoding_mtag = (
                                    mtag_w
                                    * dynet.concatenate(
                                        [current_encoding_mtag, prev_encoding_mtag]
                                    )
                                    + mtag_b
                                )
                            else:
                                encoding_mtag = current_encoding_mtag
                            prev_encoding_mtag = current_encoding_mtag
                        else:
                            encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )

                        entry.vec = dynet.concatenate([entry.vec, encoding_mtag])

                    entry.pos_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                # Predicted pos tags
                lstm_forward = self.pos_builders[0].initial_state()
                lstm_backward = self.pos_builders[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.pos_lstms[1] = lstm_forward.output()
                    rentry.pos_lstms[0] = lstm_backward.output()

                for entry in conll_sentence:
                    entry.pos_vec = dynet.concatenate(entry.pos_lstms)

                blstm_forward = self.pos_bbuilders[0].initial_state()
                blstm_backward = self.pos_bbuilders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.pos_vec)
                    blstm_backward = blstm_backward.add_input(rentry.pos_vec)
                    entry.pos_lstms[1] = blstm_forward.output()
                    rentry.pos_lstms[0] = blstm_backward.output()

                concat_layer = [
                    dynet.concatenate(entry.pos_lstms) for entry in conll_sentence
                ]
                outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                predicted_pos_indices = [np.argmax(o.value()) for o in outputFFlayer]
                predicted_postags = [self.id2pos[idx] for idx in predicted_pos_indices]

                # Add predicted pos tags
                prev_encoding_pos = None
                for entry, posid in zip(conll_sentence, predicted_pos_indices):
                    if self.pos_encoding_composition_type == "w_sum":
                        current_encoding_pos = self.plookup[posid]
                        if prev_encoding_pos:
                            encoding_pos = (
                                prev_encoding_pos * self.encoding_composition_alpha
                                + current_encoding_pos
                                * (1 - self.encoding_composition_alpha)
                            )
                        else:
                            encoding_pos = current_encoding_pos
                        prev_encoding_pos = current_encoding_pos
                    elif self.pos_encoding_composition_type == "cwise_mult":
                        current_encoding_pos = self.plookup[posid]
                        if prev_encoding_pos:
                            encoding_mtag = dynet.cmult(
                                prev_encoding_pos, current_encoding_pos
                            )
                        else:
                            encoding_pos = current_encoding_pos
                        prev_encoding_pos = current_encoding_pos
                    elif self.pos_encoding_composition_type == "mlp":
                        current_encoding_pos = self.plookup[posid]
                        if prev_encoding_pos:
                            pos_w = dynet.parameter(
                                self.pos_encoding_vertical_composition_w
                            )
                            pos_b = dynet.parameter(
                                self.pos_encoding_vertical_composition_b
                            )
                            encoding_pos = (
                                pos_w
                                * dynet.concatenate(
                                    [current_encoding_pos, prev_encoding_pos]
                                )
                                + pos_b
                            )
                        else:
                            encoding_pos = current_encoding_pos
                        prev_encoding_pos = current_encoding_pos
                    else:
                        encoding_pos = self.plookup[posid]

                    entry.vec = dynet.concatenate([entry.vec, encoding_pos])
                    entry.lstms = [entry.vec, entry.vec]

                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = dynet.concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(
                            conll_sentence, reversed(conll_sentence)
                        ):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence)
                heads = parse_proj(scores)

                # Multiple roots: heading to the previous "rooted" one
                rootCount = 0
                rootWid = -1
                for index, head in enumerate(heads):
                    if head == 0:
                        rootCount += 1
                        if rootCount == 1:
                            rootWid = index
                        if rootCount > 1:
                            heads[index] = rootWid
                            rootWid = index

                for entry, head, pos in zip(conll_sentence, heads, predicted_postags):
                    entry.pred_parent_id = head
                    entry.pred_relation = "_"
                    entry.pred_pos = pos

                dump = False

                if self.labelsFlag:
                    concat_layer = [
                        self.__getRelVector(conll_sentence, head, modifier + 1)
                        for modifier, head in enumerate(heads[1:])
                    ]
                    outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
                    predicted_rel_indices = [
                        np.argmax(o.value()) for o in outputFFlayer
                    ]
                    predicted_rels = [self.irels[idx] for idx in predicted_rel_indices]
                    for modifier, head in enumerate(heads[1:]):
                        conll_sentence[modifier + 1].pred_relation = predicted_rels[
                            modifier
                        ]

                dynet.renew_cg()
                if not dump:
                    yield sentence

    def morph2word(self, morph_dict):
        word_emb = {}
        for word in morph_dict.keys():
            morph_seg = morph_dict[word]

            word_vec = self.__getWordVector(morph_seg)
            word_emb[word] = word_vec.vec_value()
        dynet.renew_cg()
        return word_emb

    def morph(self):
        morph_dict = {}
        for morph in self.m2i.keys():
            morph_dict[morph] = self.__getMorphVector(morph).vec_value()
        dynet.renew_cg()
        return morph_dict

    def Train_Morph(self):
        self.trainer.set_sparse_updates(False)
        start = time.time()
        for iWord, word in enumerate(list(self.morph_dict.keys())):
            if iWord % 2000 == 0 and iWord != 0:
                print(
                    "Processing word number: %d" % iWord,
                    ", Time: %.2f" % (time.time() - start),
                )
                start = time.time()

            morph_seg = self.morph_dict[word]
            morph_vec = self.__getWordVector(morph_seg)

            if self.ext_embeddings is None:
                vec_gold = self.wlookup[int(self.vocab.get(word, 0))].vec_value()
            elif word in self.ext_embeddings:
                vec_gold = self.ext_embeddings[word]
            else:
                vec_gold = None

            if vec_gold is not None:
                y_gold = dynet.vecInput(self.wdims)
                y_gold.set(vec_gold)
                mErrs = self.cosine_proximity(morph_vec, y_gold)
                mErrs.backward()
                self.trainer.update()
            dynet.renew_cg()

    def embed_word(self, word):
        return [self.input_lookup[char] for char in word]

    def run_lstm(self, init_state, input_vecs):
        s = init_state
        out_vectors = []
        for vector in input_vecs:
            s = s.add_input(vector)
            out_vector = s.output()
            out_vectors.append(out_vector)
        return out_vectors

    def encode_word(self, word):
        word_rev = list(reversed(word))
        fwd_vectors = self.run_lstm(self.enc_fwd_lstm.initial_state(), word)
        bwd_vectors = self.run_lstm(self.enc_bwd_lstm.initial_state(), word_rev)
        bwd_vectors = list(reversed(bwd_vectors))
        vectors = [dynet.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
        return vectors

    def Train(self, conll_path):
        print("Training...")
        self.trainer.set_sparse_updates(True)
        eloss = 0.0
        mloss = 0.0
        eerrors = 0
        etotal = 0
        start = time.time()

        with open(conll_path, "r") as conllFP:
            shuffledData = list(
                read_conll(conllFP, self.c2i, self.m2i, self.t2i, self.morph_dict)
            )
            random.shuffle(shuffledData)

            errs = []
            lerrs = []
            posErrs = []
            segErrs = []
            mTagErrs = []

            for iSentence, sentence in enumerate(shuffledData):
                if iSentence % 500 == 0 and iSentence != 0:
                    print(
                        "Processing sentence number: %d" % iSentence,
                        ", Loss: %.4f" % (eloss / etotal),
                        ", Time: %.2f" % (time.time() - start),
                    )
                    start = time.time()
                    eerrors = 0
                    eloss = 0.0
                    etotal = 0

                conll_sentence = [
                    entry for entry in sentence if isinstance(entry, ConllEntry)
                ]

                if self.morphTagFlag:
                    sentence_context = []
                    last_state_char = self.char_rnn.predict_sequence(
                        [self.clookup[self.c2i["<start>"]]]
                    )[-1]
                    rev_last_state_char = self.char_rnn.predict_sequence(
                        [self.clookup[self.c2i["<start>"]]]
                    )[-1]
                    sentence_context.append(
                        dynet.concatenate([last_state_char, rev_last_state_char])
                    )
                    for entry in conll_sentence:
                        last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in entry.idChars]
                        )
                        rev_last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in reversed(entry.idChars)]
                        )
                        entry.char_rnn_states = [
                            dynet.concatenate([f, b])
                            for f, b in zip(last_state_char, rev_last_state_char)
                        ]
                        sentence_context.append(entry.char_rnn_states[-1])

                prev_encoding_mtag = None
                prev_encoding_morph = None
                for idx, entry in enumerate(conll_sentence):
                    c = float(self.wordsCount.get(entry.norm, 0))
                    dropFlag = random.random() < (c / (0.25 + c))
                    wordvec = (
                        self.wlookup[
                            int(self.vocab.get(entry.norm, 0)) if dropFlag else 0
                        ]
                        if self.wdims > 0
                        else None
                    )
                    if self.morphTagFlag:
                        entry.vec = dynet.dropout(
                            dynet.concatenate([wordvec, entry.char_rnn_states[-1]]),
                            0.33,
                        )
                    else:
                        last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in entry.idChars]
                        )[-1]
                        rev_last_state_char = self.char_rnn.predict_sequence(
                            [self.clookup[c] for c in reversed(entry.idChars)]
                        )[-1]
                        entry.vec = dynet.dropout(
                            dynet.concatenate(
                                [wordvec, last_state_char, rev_last_state_char]
                            ),
                            0.33,
                        )

                    if self.morphFlag:
                        if len(entry.norm) > 2:
                            if self.goldMorphFlag:
                                seg_vec = self.__getSegmentationVector(entry.norm)
                                seg_vec = dynet.vecInput(seg_vec.dim()[0][0])
                                seg_vec.set(entry.idMorphs)
                                morph_seg = generate_morphs(
                                    entry.norm, seg_vec.vec_value()
                                )
                            else:
                                seg_vec = self.__getSegmentationVector(entry.norm)
                                morph_seg = generate_morphs(
                                    entry.norm, seg_vec.vec_value()
                                )
                                vec_gold = dynet.vecInput(seg_vec.dim()[0][0])
                                vec_gold.set(entry.idMorphs)
                                segErrs.append(
                                    self.binary_crossentropy(seg_vec, vec_gold)
                                )
                        else:
                            morph_seg = [entry.norm]

                        last_state_morph = self.morph_rnn.predict_sequence(
                            [self.__getMorphVector(morph) for morph in morph_seg]
                        )[-1]
                        rev_last_state_morph = self.morph_rnn.predict_sequence(
                            [
                                self.__getMorphVector(morph)
                                for morph in reversed(morph_seg)
                            ]
                        )[-1]

                        if self.morph_encoding_composition_type == "w_sum":
                            current_encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )
                            if prev_encoding_morph:
                                encoding_morph = (
                                    prev_encoding_morph
                                    * self.encoding_composition_alpha
                                    + current_encoding_morph
                                    * (1 - self.encoding_composition_alpha)
                                )
                            else:
                                encoding_morph = current_encoding_morph
                            prprev_encoding_morphe = current_encoding_morph
                        elif self.morph_encoding_composition_type == "cwise_mult":
                            current_encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )
                            if prev_encoding_morph:
                                encoding_mtag = dynet.cmult(
                                    prev_encoding_morph, current_encoding_morph
                                )
                            else:
                                encoding_morph = current_encoding_morph
                        elif self.morph_encoding_composition_type == "mlp":
                            current_encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )
                            if prev_encoding_morph:
                                morph_w = dynet.parameter(
                                    self.morph_encoding_vertical_composition_w
                                )
                                morhp_b = dynet.parameter(
                                    self.morph_encoding_vertical_composition_b
                                )
                                encoding_mtag = (
                                    morph_w
                                    * dynet.concatenate(
                                        [current_encoding_morph, prev_encoding_morph]
                                    )
                                    + morhp_b
                                )
                            else:
                                encoding_morph = current_encoding_morph
                            prev_encoding_morph = current_encoding_morph
                        else:
                            encoding_morph = dynet.concatenate(
                                [last_state_morph, rev_last_state_morph]
                            )

                        entry.vec = dynet.concatenate(
                            [entry.vec, dynet.dropout(encoding_morph, 0.33)]
                        )

                    if self.morphTagFlag:
                        if self.goldMorphTagFlag:
                            morph_tags = entry.idMorphTags
                        else:
                            word_context = [
                                c
                                for i, c in enumerate(sentence_context)
                                if i - 1 != idx
                            ]
                            mTagErrs.append(
                                self.__getLossMorphTagging(
                                    entry.char_rnn_states,
                                    entry.idMorphTags,
                                    word_context,
                                )
                            )
                            predicted_sequence = self.generate(
                                entry.char_rnn_states, word_context
                            )
                            morph_tags = predicted_sequence

                        last_state_mtag = self.mtag_rnn.predict_sequence(
                            [self.tlookup[t] for t in morph_tags]
                        )[-1]
                        rev_last_state_mtag = self.mtag_rnn.predict_sequence(
                            [self.tlookup[t] for t in reversed(morph_tags)]
                        )[-1]

                        if self.mtag_encoding_composition_type == "w_sum":
                            current_encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )
                            if prev_encoding_mtag:
                                encoding_mtag = (
                                    prev_encoding_mtag * self.encoding_composition_alpha
                                    + current_encoding_mtag
                                    * (1 - self.encoding_composition_alpha)
                                )
                            else:
                                encoding_mtag = current_encoding_mtag
                            prev_encoding_mtag = current_encoding_mtag
                        elif self.mtag_encoding_composition_type == "cwise_mult":
                            current_encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )
                            if prev_encoding_mtag:
                                encoding_mtag = dynet.cmult(
                                    prev_encoding_mtag, current_encoding_mtag
                                )
                            else:
                                encoding_mtag = current_encoding_mtag
                            prev_encoding_mtag = current_encoding_mtag
                        elif self.mtag_encoding_composition_type == "mlp":
                            current_encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )
                            if prev_encoding_mtag:
                                mtag_w = dynet.parameter(
                                    self.mtag_encoding_vertical_composition_w
                                )
                                mtag_b = dynet.parameter(
                                    self.mtag_encoding_vertical_composition_b
                                )
                                encoding_mtag = (
                                    mtag_w
                                    * dynet.concatenate(
                                        [current_encoding_mtag, prev_encoding_mtag]
                                    )
                                    + mtag_b
                                )
                            else:
                                encoding_mtag = current_encoding_mtag
                            prev_encoding_mtag = current_encoding_mtag
                        else:
                            encoding_mtag = dynet.concatenate(
                                [last_state_mtag, rev_last_state_mtag]
                            )

                        entry.vec = dynet.concatenate(
                            [entry.vec, dynet.dropout(encoding_mtag, 0.33)]
                        )

                    entry.pos_lstms = [entry.vec, entry.vec]
                    entry.headfov = None
                    entry.modfov = None

                    entry.rheadfov = None
                    entry.rmodfov = None

                # POS tagging loss
                lstm_forward = self.pos_builders[0].initial_state()
                lstm_backward = self.pos_builders[1].initial_state()
                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    lstm_forward = lstm_forward.add_input(entry.vec)
                    lstm_backward = lstm_backward.add_input(rentry.vec)

                    entry.pos_lstms[1] = lstm_forward.output()
                    rentry.pos_lstms[0] = lstm_backward.output()

                for entry in conll_sentence:
                    entry.pos_vec = dynet.concatenate(entry.pos_lstms)

                blstm_forward = self.pos_bbuilders[0].initial_state()
                blstm_backward = self.pos_bbuilders[1].initial_state()

                for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                    blstm_forward = blstm_forward.add_input(entry.pos_vec)
                    blstm_backward = blstm_backward.add_input(rentry.pos_vec)
                    entry.pos_lstms[1] = blstm_forward.output()
                    rentry.pos_lstms[0] = blstm_backward.output()

                concat_layer = [
                    dynet.dropout(dynet.concatenate(entry.pos_lstms), 0.33)
                    for entry in conll_sentence
                ]
                outputFFlayer = self.ffSeqPredictor.predict_sequence(concat_layer)
                posIDs = [self.pos.get(entry.pos) for entry in conll_sentence]
                for pred, gold in zip(outputFFlayer, posIDs):
                    posErrs.append(self.pick_neg_log(pred, gold))

                # Add predicted pos tags
                prev_encoding_pos = None
                for entry, poses in zip(conll_sentence, outputFFlayer):
                    if self.pos_encoding_composition_type == "w_sum":
                        current_encoding_pos = self.plookup[np.argmax(poses.value())]
                        if prev_encoding_pos:
                            encoding_pos = (
                                prev_encoding_pos * self.encoding_composition_alpha
                                + current_encoding_pos
                                * (1 - self.encoding_composition_alpha)
                            )
                        else:
                            encoding_pos = current_encoding_pos
                        prev_encoding_pos = current_encoding_pos
                    elif self.pos_encoding_composition_type == "cwise_mult":
                        current_encoding_pos = self.plookup[np.argmax(poses.value())]
                        if prev_encoding_pos:
                            encoding_mtag = dynet.cmult(
                                prev_encoding_pos, current_encoding_pos
                            )
                        else:
                            encoding_pos = current_encoding_pos
                        prev_encoding_pos = current_encoding_pos
                    elif self.pos_encoding_composition_type == "mlp":
                        current_encoding_pos = self.plookup[np.argmax(poses.value())]
                        if prev_encoding_pos:
                            pos_w = dynet.parameter(
                                self.pos_encoding_vertical_composition_w
                            )
                            pos_b = dynet.parameter(
                                self.pos_encoding_vertical_composition_b
                            )
                            encoding_pos = (
                                pos_w
                                * dynet.concatenate(
                                    [current_encoding_pos, prev_encoding_pos]
                                )
                                + pos_b
                            )
                        else:
                            encoding_pos = current_encoding_pos
                        prev_encoding_pos = current_encoding_pos
                    else:
                        encoding_pos = self.plookup[np.argmax(poses.value())]

                    entry.vec = dynet.concatenate(
                        [entry.vec, dynet.dropout(encoding_pos, 0.33)]
                    )
                    entry.lstms = [entry.vec, entry.vec]

                # Parsing losses
                if self.blstmFlag:
                    lstm_forward = self.builders[0].initial_state()
                    lstm_backward = self.builders[1].initial_state()

                    for entry, rentry in zip(conll_sentence, reversed(conll_sentence)):
                        lstm_forward = lstm_forward.add_input(entry.vec)
                        lstm_backward = lstm_backward.add_input(rentry.vec)

                        entry.lstms[1] = lstm_forward.output()
                        rentry.lstms[0] = lstm_backward.output()

                    if self.bibiFlag:
                        for entry in conll_sentence:
                            entry.vec = dynet.concatenate(entry.lstms)

                        blstm_forward = self.bbuilders[0].initial_state()
                        blstm_backward = self.bbuilders[1].initial_state()

                        for entry, rentry in zip(
                            conll_sentence, reversed(conll_sentence)
                        ):
                            blstm_forward = blstm_forward.add_input(entry.vec)
                            blstm_backward = blstm_backward.add_input(rentry.vec)

                            entry.lstms[1] = blstm_forward.output()
                            rentry.lstms[0] = blstm_backward.output()

                scores, exprs = self.__evaluate(conll_sentence)
                gold = [entry.parent_id for entry in conll_sentence]
                heads = parse_proj(scores, gold if self.costaugFlag else None)

                if self.labelsFlag:
                    concat_layer = [
                        dynet.dropout(
                            self.__getRelVector(conll_sentence, head, modifier + 1),
                            0.33,
                        )
                        for modifier, head in enumerate(gold[1:])
                    ]
                    outputFFlayer = self.ffRelPredictor.predict_sequence(concat_layer)
                    relIDs = [
                        self.rels[conll_sentence[modifier + 1].relation]
                        for modifier, _ in enumerate(gold[1:])
                    ]
                    for pred, goldid in zip(outputFFlayer, relIDs):
                        lerrs.append(self.pick_neg_log(pred, goldid))

                e = sum([1 for h, g in zip(heads[1:], gold[1:]) if h != g])
                eerrors += e
                if e > 0:
                    loss = [
                        (exprs[h][i] - exprs[g][i])
                        for i, (h, g) in enumerate(zip(heads, gold))
                        if h != g
                    ]  # * (1.0/float(e))
                    eloss += e
                    mloss += e
                    errs.extend(loss)

                etotal += len(conll_sentence)

                if iSentence % 1 == 0:
                    if (
                        len(errs) > 0
                        or len(lerrs) > 0
                        or len(posErrs) > 0
                        or len(segErrs) > 0
                        or len(mTagErrs) > 0
                    ):
                        eerrs = dynet.esum(errs + lerrs + posErrs + segErrs + mTagErrs)
                        eerrs.scalar_value()
                        eerrs.backward()
                        self.trainer.update()
                        errs = []
                        lerrs = []
                        posErrs = []
                        segErrs = []
                        mTagErrs = []

                    dynet.renew_cg()

        print("Loss: %.4f" % (mloss / iSentence))
