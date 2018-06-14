"""NN layers"""

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.modules.seq2vec_encoders.cnn_encoder import CnnEncoder
from allennlp.commands.elmo import ElmoEmbedder

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers,
                 dropout=0, residual=False):
        super(TokenLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
        )
        self.dropout = dropout

    def forward(self, x, x_mask=None):
        """A more thorough implementation would take care of padding."""
        x = x.transpose(0, 1)

        enc = self.lstm(x)[0]

        return enc.transpose(0, 1)


class CharCNN(nn.Module):
    def __init__(self, input_dim, num_filters, ngrams):
        super(CharCNN, self).__init__()
        self.cnn = CnnEncoder(
            input_dim,
            num_filters,
            ngram_filter_sizes=ngrams
        )

    def forward(self, x):
        batch_size, seq_len, char_len, hdim = x.size()

        # Flatten and encode
        flat = x.view(-1, char_len, hdim)
        enc = self.cnn(flat, None).view(batch_size, seq_len, -1)

        return enc


class CRF(nn.Module):

    def __init__(self, input_dim, tag_dict):
        super(CRF, self).__init__()
        constraints = allowed_transitions('BIO', tag_dict)
        self.crf = ConditionalRandomField(len(tag_dict), constraints)
        self.projection = nn.Linear(input_dim, len(tag_dict))

    def forward(self, x, x_mask, tags=None):
        batch_size, seq_len, hdim = x.size()

        # Flatten and project class scores
        flat = x.view(-1, seq_len, hdim)
        scores = self.projection(flat).view(batch_size, seq_len, -1)

        # Compute CRF
        predicted_tags = self.crf.viterbi_tags(scores, x_mask)

        # Compute likelihood if given tags
        if tags is not None:
            log_likelihood = self.crf(scores, tags, x_mask)
        else:
            log_likelihood = None

        return predicted_tags, log_likelihood


class ELMO(nn.Module):
    def __init__(self, projection_dim=None, dropout=0.5):
        super(ELMO, self).__init__()
        self.dropout = dropout
        # funky... should be able to not force this
        device = 0 if torch.cuda.is_available() else -1
        self.elmo = ElmoEmbedder(cuda_device=device)
        self.layer_weights = nn.Parameter(torch.ones(3))
        if projection_dim:
            self.projection = nn.Linear(1024, projection_dim)
        else:
            self.projection = None

    def forward(self, batch_sentences):
        # Embed words
        embeddings = self.elmo.batch_to_embeddings(batch_sentences)[0]

        # Apply learned weights to combine 3 layers
        norm_weights = F.softmax(self.layer_weights, dim=-1).view(1, 3, 1, 1)
        norm_weights = norm_weights.expand_as(embeddings)
        embeddings = (norm_weights * embeddings).sum(1)

        if self.dropout:
            embeddings = F.dropout(embeddings, self.dropout, self.training)

        if self.projection is not None:
            flat = embeddings.view(-1, embeddings.size(-1))
            projected = F.relu(self.projection(flat))
            embeddings = projected.view(
                embeddings.size(0), embeddings.size(1), -1
            )
        return embeddings


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, trainable=False):
        super(Embedding, self).__init__()
        self.embedding = nn.Embedding(num_embeddings,
                                      embedding_dim,
                                      padding_idx=0)
        if not trainable:
            for p in self.embedding.parameters():
                p.requires_grad = False

    def forward(self, x):
        embs = self.embedding(x)
        return embs
