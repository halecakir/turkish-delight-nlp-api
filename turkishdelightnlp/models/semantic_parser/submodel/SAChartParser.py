import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

use_cuda = False
if use_cuda:
    torch_t = torch.cuda
    def from_numpy(ndarray):
        return torch.from_numpy(ndarray).pin_memory().cuda(async=True)
else:
    print("Not using CUDA!")
    torch_t = torch
    from torch import from_numpy
    
class SAChartParser(nn.Module):
    
    def __init__(self, vocab, hparams, restore=False):
        super().__init__()
        self.vocab = vocab
        self.d_model = hparams.d_model
        self.partitioned = hparams.partitioned
        self.d_content = (self.d_model // 2) if self.partitioned else self.d_model
        self.d_positionel = (hparams.d_model // 2) if self.partitioned else None
        
        num_embeddings_map = {
            'tags': vocab.num_pos,
            'words': vocab.num_train_word,
            'dep': vocab.num_dep,
            'ent': vocab.num_ent,
            'ent_iob': vocab.num_ent_iob,
        }
        
        emb_dropouts_map = {
              'tags': hparams.tag_emb_dropout,
              'words': hparams.word_emb_dropout, 
              'dep': hparams.dep_emb_dropout,
              'ent': hparams.ent_emb_dropout,
              'ent_iob': hparams.ent_iob_emb_dropout,  
              }
        self.emb_types = ['tags', 'words', 'dep', 'ent', 'ent_iob']
        if use_encoder:
            self.embedding = MultiLevelEmbedding(
                [num_embeddings_map[emb_type] for emb_type in self.emb_types],
                hparams.d_model,
                d_positional=self.d_positional,
                dropout=hparams.embedding_dropout,
                timing_dropout=hparams.timing_dropout,
                emb_dropouts_list=[emb_dropouts_map[emb_type] for emb_type in self.emb_types],
                extra_content_dropout=self.morpho_emb_dropout,
                max_len=hparams.sentence_max_len,
            )
            
            self.encoder = Encoder(
                self.embedding,
                num_layers=hparams.num_layers,
                num_heads=hparams.num_heads,
                d_kv=hparams.d_kv,
                d_ff=hparams.d_ff,
                d_positional=self.d_positional,
                num_layers_position_only=hparams.num_layers_position_only,
                relu_dropout=hparams.relu_dropout,
                residual_dropout=hparams.residual_dropout,
                attention_dropout=hparams.attention_dropout,)
        
        self.span_nagram_attentions = SpanNgramAttentions(ngram_size=self.ngram_vocab.size,
                                                          d_emb=hparams.d_model,
                                                          n_channels=self.ngram_channels)
        self.channel_ids = from_numpy(np.array([i for i in range(self.ngram_channels)]))
        
        self.f_label = nn.Sequential(
            nn.Linear(hparams.d_model * (self.ngram_channels + 1), hparams.d_label_hidden),
            # nn.Linear(hparams.d_model, hparams.d_label_hidden),
            LayerNormalization(hparams.d_label_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_label_hidden, vocab.num_parse_label),
            )
            
        
        self.f_split = nn.Sequential(
            nn.Linear(hparams.d_model * (self.ngram_channels + 1), hparams.d_split_hidden),
            # nn.Linear(hparams.d_model, hparams.d_label_hidden),
            LayerNormalization(hparams.d_split_hidden),
            nn.ReLU(),
            nn.Linear(hparams.d_split_hidden, 1),
            )
        
    def forward(self, span):
        label_score = self.f_label(span)
        split_scpre = self.f_split(span)
        
        
    def helper(self, label_scores, split_scores, sen_len, left, right, gold=None):
        position = get_position(sen_len, left, right)
        label_score = label_scores[position]
        if self.training:
            oracle_label = gold.oracle_label(left, right)
            oracle_label_index = self.vocab.parse_label2id(oracle_label)
            label_score = self.augment(label_score, oracle_label_index)
            oracle_label_score = label_score[oracle_label_index]

        if right - left == sen_len:
            label_score[0] = float("-inf")

        argmax_label_score, argmax_label_index = torch.max(label_score, dim=0)
        argmax_label = self.vocab.id2parse_label(int(argmax_label_index))

        if self.training:
            label = oracle_label
            label_loss = argmax_label_score - oracle_label_score
        else:
            label = argmax_label
            label_loss = label_score[argmax_label_index]

        if right - left == 1:
            tree = LeafParseNode(left, "pos", "word")
            if label:
                tree = InternalParseNode(label, [tree])
            return [tree], label_loss

        left_positions = get_position(sen_len, left, range(left + 1, right))
        right_positions = get_position(sen_len, range(left + 1, right), right)
        splits = split_scores[left_positions] + split_scores[right_positions]

        if self.training:
            oracle_splits = gold.oracle_splits(left, right)
            oracle_split = min(oracle_splits)
            oracle_split_index = oracle_split - (left + 1)
            splits = self.augment(splits, oracle_split_index)
            oracle_split_score = splits[oracle_split_index]

        argmax_split_score, argmax_split_index = torch.max(splits, dim=0)
        argmax_split = argmax_split_index + (left + 1)

        if self.training:
            split = oracle_split
            split_loss = argmax_split_score - oracle_split_score
        else:
            split = argmax_split
            split_loss = splits[argmax_split_index]

        left_trees, left_loss = self.helper(
            label_scores, split_scores, sen_len, left, int(split), gold
        )
        right_trees, right_loss = self.helper(
            label_scores, split_scores, sen_len, int(split), right, gold
        )

        children = left_trees + right_trees
        loss = label_loss + split_loss + left_loss + right_loss
        if label:
            children = [InternalParseNode(label, children)]
        return children, loss

    def get_loss(self, spans, sen_lens, trees):
        batch_loss = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score, split_score = self.forward(spans[i][:span_num])
            _, loss = self.helper(label_score, split_score, length, 0, length, trees[i])
            batch_loss.append(loss)
        return batch_loss

    def predict(self, spans, sen_lens):
        pred_trees = []
        for i, length in enumerate(sen_lens):
            span_num = (1 + length) * length // 2
            label_score, split_score = self.forward(spans[i][:span_num])
            pred_tree, _ = self.helper(label_score, split_score, length, 0, length)
            pred_trees.append(pred_tree[0].convert())
        return pred_trees

    def augment(self, scores, oracle_index):
        increment = torch.ones_like(scores)
        increment[oracle_index] = 0
        return scores + increment