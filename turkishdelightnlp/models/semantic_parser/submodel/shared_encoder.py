import math
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pack_sequence,
    pad_sequence,
    pad_packed_sequence,
)

from module import CharLSTM, EncoderLayer, PositionEncoding


class LSTM_Encoder(nn.Module):
    def __init__(
        self,
        vocab,
        ext_emb,
        word_dim,
        pos_dim,
        dep_dim,
        ent_dim,
        ent_iob_dim,
        lstm_dim,
        lstm_layer,
        emb_drop=0.5,
        lstm_drop=0.4,
    ):
        super(LSTM_Encoder, self).__init__()
        self.vocab = vocab
        self.ext_word_embedding = nn.Embedding.from_pretrained(ext_emb)
        self.word_embedding = nn.Embedding(vocab.num_train_word, word_dim, padding_idx=0)

        self.pos_embedding = nn.Embedding(vocab.num_pos, pos_dim)
        self.dep_embedding = nn.Embedding(vocab.num_dep, dep_dim)
#        self.ent_embedding = nn.Embedding(vocab.num_ent, ent_dim)
#        self.ent_iob_embedding = nn.Embedding(vocab.num_ent_iob, ent_iob_dim)

        self.lstm = nn.LSTM(
#            input_size=word_dim + pos_dim + dep_dim + ent_dim + ent_iob_dim,
            input_size=word_dim + pos_dim + dep_dim,
            hidden_size=lstm_dim // 2,
            bidirectional=True,
            num_layers=lstm_layer,
            dropout=lstm_drop,
        )
        self.emb_drop = nn.Dropout(emb_drop)
        self.lstm_dim = lstm_dim
        self.reset_parameters()

    def reset_parameters(self):
        self.word_embedding.weight.data.zero_()

    def forward(self, word_idxs, pos_idxs, dep_idxs):
        mask = word_idxs.ne(self.vocab.PAD_index)
        sen_lens = mask.sum(1)
        sorted_lens, sorted_idx = torch.sort(sen_lens, dim=0, descending=True)
        reverse_idx = torch.sort(sorted_idx, dim=0)[1]
        max_len = sorted_lens[0]

        word_idxs = word_idxs[:, :max_len]
        pos_idxs = pos_idxs[:, :max_len]
        dep_idxs = dep_idxs[:, :max_len]
#        ent_idxs = ent_idxs[:, :max_len]
#        ent_iob_idxs = ent_iob_idxs[:, :max_len]
        mask = mask[:, :max_len]

        word_emb = self.ext_word_embedding(word_idxs)
        word_emb += self.word_embedding(word_idxs.masked_fill_(word_idxs.ge(self.word_embedding.num_embeddings),
                               self.vocab.UNK_index))
        pos_emb = self.pos_embedding(pos_idxs)
        dep_emb = self.dep_embedding(dep_idxs)
#        ent_emb = self.ent_embedding(ent_idxs)
#        ent_iob_emb = self.ent_iob_embedding(ent_iob_idxs)

#        emb = torch.cat((word_emb, pos_emb, dep_emb, ent_emb, ent_iob_emb), -1)
        emb = torch.cat((word_emb, pos_emb, dep_emb), -1)
        emb = self.emb_drop(emb)

        emb = emb[sorted_idx]
        lstm_input = pack_padded_sequence(emb, sorted_lens, batch_first=True)

        r_out, _ = self.lstm(lstm_input, None)
        lstm_out, _ = pad_packed_sequence(r_out, batch_first=True)

        # get all span vectors
        x = lstm_out[reverse_idx].transpose(0, 1)
        x = x.unsqueeze(1) - x
        x_forward, x_backward = x.chunk(2, dim=-1)

        mask = (mask & word_idxs.ne(self.vocab.STOP_index))[:, :-1]
        mask = mask.unsqueeze(1) & mask.new_ones(max_len - 1, max_len - 1).triu(1)
        lens = mask.sum((1, 2))
        x_forward = x_forward[:-1, :-1].permute(2, 1, 0, 3)
        x_backward = x_backward[1:, 1:].permute(2, 0, 1, 3)
        x_span = torch.cat([x_forward[mask], x_backward[mask]], -1)
        x_span = pad_sequence(torch.split(x_span, lens.tolist()), True)
        return x_span, (sen_lens - 2).tolist()


class Transformer_Encoder(nn.Module):

    def __init__(
        self,
        vocab,
        ext_emb,
        word_dim,
        pos_dim,
        dep_dim,
        ent_dim,
        ent_iob_dim,
        lstm_dim,
        lstm_layer,
        emb_drop=0.5,
        lstm_drop=0.4,
    ):
        super(Transformer_Encoder, self).__init__()
        self.vocab = vocab
        self.ext_word_embedding = nn.Embedding.from_pretrained(ext_emb)
        self.word_embedding = nn.Embedding(vocab.num_train_word, word_dim, padding_idx=0)

        self.pos_embedding = nn.Embedding(vocab.num_pos, pos_dim)
        self.dep_embedding = nn.Embedding(vocab.num_dep, dep_dim)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
            mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):

        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
#        a = output.clone()
        output = output.mean(dim=1)
#        output = output.view(src.size(0),-1)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
            -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)