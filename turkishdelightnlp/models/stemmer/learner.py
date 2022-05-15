# coding=utf-8
import dynet as dy
import dynet_config

# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)


def embed_sentence(sentence, input_lookup):
    sentence = [EOS] + list(sentence) + [EOS]
    sentence = [char2int[c] for c in sentence]
    return [input_lookup[char] for char in sentence]


def run_lstm(init_state, input_vecs):
    s = init_state
    out_vectors = []
    for vector in input_vecs:
        s = s.add_input(vector)
        out_vector = s.output()
        out_vectors.append(out_vector)
    return out_vectors


def encode_sentence(enc_fwd_lstm, enc_bwd_lstm, sentence):
    sentence_rev = list(reversed(sentence))
    fwd_vectors = run_lstm(enc_fwd_lstm.initial_state(), sentence)
    bwd_vectors = run_lstm(enc_bwd_lstm.initial_state(), sentence_rev)
    bwd_vectors = list(reversed(bwd_vectors))
    vectors = [dy.concatenate(list(p)) for p in zip(fwd_vectors, bwd_vectors)]
    return vectors


def generate(
    in_seq,
    enc_fwd_lstm,
    enc_bwd_lstm,
    dec_lstm,
    decoder_w,
    decoder_b,
    attention_w1,
    output_lookup,
    input_lookup,
    attention_w2,
    attention_v,
):
    embedded = embed_sentence(in_seq, input_lookup)
    encoded = encode_sentence(enc_fwd_lstm, enc_bwd_lstm, embedded)

    w = dy.parameter(decoder_w)
    b = dy.parameter(decoder_b)
    w1 = dy.parameter(attention_w1)
    input_mat = dy.concatenate_cols(encoded)
    w1dt = None

    last_output_embeddings = output_lookup[char2int[EOS]]
    s = dec_lstm.initial_state().add_input(
        dy.concatenate([dy.vecInput(STATE_SIZE * 2), last_output_embeddings])
    )

    out = ""
    count_EOS = 0
    for i in range(len(in_seq) * 2):
        if count_EOS == 2:
            break
        # w1dt can be computed and cached once for the entire decoding phase
        w1dt = w1dt or w1 * input_mat
        vector = dy.concatenate(
            [
                attend(input_mat, s, w1dt, attention_w2, attention_v),
                last_output_embeddings,
            ]
        )
        s = s.add_input(vector)
        out_vector = w * s.output() + b
        probs = dy.softmax(out_vector).vec_value()
        next_char = probs.index(max(probs))
        last_output_embeddings = output_lookup[next_char]
        if int2char[next_char] == EOS:
            count_EOS += 1
            continue

        out += int2char[next_char]
    return out


def attend(input_mat, state, w1dt, attention_w2, attention_v):
    w2 = dy.parameter(attention_w2)
    v = dy.parameter(attention_v)

    # input_mat: (encoder_state x seqlen) => input vecs concatenated as cols
    # w1dt: (attdim x seqlen)
    # w2dt: (attdim x attdim)
    w2dt = w2 * dy.concatenate(list(state.s()))
    # att_weights: (seqlen,) row vector
    unnormalized = dy.transpose(v * dy.tanh(dy.colwise_add(w1dt, w2dt)))
    att_weights = dy.softmax(unnormalized)
    # context: (encoder_state)
    context = input_mat * att_weights
    return context


EOS = "<EOS>"
characters = list("aâbcçdefgğhıijklmnoöpqrsştuüvwxyz.,;_'\"?:!()-0123456789 ")
characters.append(EOS)

int2char = list(characters)
char2int = {c: i for i, c in enumerate(characters)}

VOCAB_SIZE = len(characters)

LSTM_NUM_OF_LAYERS = 1
EMBEDDINGS_SIZE = 128
STATE_SIZE = 128
ATTENTION_SIZE = 32


class Stemmer:
    def __init__(self):
        self.model = dy.Model()

        self.enc_fwd_lstm = dy.LSTMBuilder(
            LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model
        )
        self.enc_bwd_lstm = dy.LSTMBuilder(
            LSTM_NUM_OF_LAYERS, EMBEDDINGS_SIZE, STATE_SIZE, self.model
        )
        self.dec_lstm = dy.LSTMBuilder(
            LSTM_NUM_OF_LAYERS, STATE_SIZE * 2 + EMBEDDINGS_SIZE, STATE_SIZE, self.model
        )

        self.input_lookup = self.model.add_lookup_parameters(
            (VOCAB_SIZE, EMBEDDINGS_SIZE)
        )
        self.attention_w1 = self.model.add_parameters((ATTENTION_SIZE, STATE_SIZE * 2))
        self.attention_w2 = self.model.add_parameters(
            (ATTENTION_SIZE, STATE_SIZE * LSTM_NUM_OF_LAYERS * 2)
        )
        self.attention_v = self.model.add_parameters((1, ATTENTION_SIZE))
        self.decoder_w = self.model.add_parameters((VOCAB_SIZE, STATE_SIZE))
        self.decoder_b = self.model.add_parameters((VOCAB_SIZE))
        self.output_lookup = self.model.add_lookup_parameters(
            (VOCAB_SIZE, EMBEDDINGS_SIZE)
        )

    def load(self, model_path):
        self.model.populate(model_path)

    def predict_stem(self, word):
        predicted_stem = generate(
            word,
            self.enc_fwd_lstm,
            self.enc_bwd_lstm,
            self.dec_lstm,
            self.decoder_w,
            self.decoder_b,
            self.attention_w1,
            self.output_lookup,
            self.input_lookup,
            self.attention_w2,
            self.attention_v,
        )
        return predicted_stem
