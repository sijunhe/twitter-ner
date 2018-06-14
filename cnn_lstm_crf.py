from model import AbstractModel
from modules import Embedding, CharCNN, TokenLSTM, ELMO, CRF

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class CnnLstmCrf(AbstractModel):

    @staticmethod
    def add_cmdline_args(parser):
        AbstractModel.add_cmdline_args(parser)

        model = parser.add_argument_group('CNN-LSTM-CRF')
        model.add_argument('--glove-dim', type=int, default=50)
        model.add_argument('--use-glove', type='bool', default=True)
        model.add_argument('--glove-file', type=str,
                           default='data/embeddings/glove.6B.50d.txt')
        model.add_argument('--use-twitter', type='bool', default=True)
        model.add_argument('--twitter-dim', type=int, default=50)
        model.add_argument('--twitter-file', type=str,
                           default='data/embeddings/glove.twitter.27B.50d.txt')
        model.add_argument('--use-pos', type='bool', default=True)
        model.add_argument('--pos-dim', type=int, default=8)
        model.add_argument('--use-elmo', type='bool', default=False)
        model.add_argument('--elmo-dim', type=int, default=150)
        model.add_argument('--elmo-dropout', type=float, default=0.5)
        model.add_argument('--char-dim', type=int, default=16)
        model.add_argument('--char-cnn-filters', type=int, default=50)
        model.add_argument('--char-cnn-ngrams', type=int, nargs='+', default=[3, 5])
        model.add_argument('--lstm-dim', type=int, default=100)
        model.add_argument('--lstm-layers', type=int, default=2)
        model.add_argument('--lstm-dropout', type=float, default=0.5)
        model.add_argument('--dropout-emb', type=float, default=0.5)
        model.add_argument('--emb-lambda', type=float, default=0.01)
        model.add_argument('--translate-emb', type='bool', default=False)

    def __init__(self, args, word_dict, tag_dict,
                 pos_dict=None, binary_dict=None):
        super(CnnLstmCrf, self).__init__()

        # Save dictionaries
        self.args = args
        self.word_dict = word_dict
        self.tag_dict = tag_dict
        self.pos_dict = pos_dict
        self.binary_dict = binary_dict

        # Pretrained Glove Wikipedia embeddings
        self.glove_emb = Embedding(len(word_dict), args.glove_dim)
        if args.use_glove:
            self.load_embeddings(self.glove_emb, args.glove_file, word_dict)

        # Pretrained Glove Twitter embeddings
        self.twitter_emb = Embedding(len(word_dict), args.twitter_dim)
        if args.use_twitter:
            self.load_embeddings(self.twitter_emb, args.twitter_file, word_dict)

        # POS embeddings
        if args.use_pos:
            self.pos_emb = Embedding(len(pos_dict), args.pos_dim)

        # ELMo pretrained BiLM embeddings
        if args.use_elmo:
            self.elmo = ELMO(args.elmo_dim, args.elmo_dropout)

        # Character byte encoder
        self.char_emb = Embedding(262, args.char_dim)
        self.char_cnn = CharCNN(args.char_dim, args.char_cnn_filters,
                                args.char_cnn_ngrams)

        # Main LSTM
        input_dim = (args.glove_dim
                     + args.twitter_dim
                     + (args.pos_dim if args.use_pos else 0)
                     + (args.elmo_dim if args.use_elmo else 0)
                     + (len(binary_dict) if binary_dict else 0)
                     + self.char_cnn.cnn.get_output_dim())
        self.lstm = TokenLSTM(input_dim, args.lstm_dim, args.lstm_layers,
                              args.lstm_dropout)

        # CRF layer
        self.crf = CRF(args.lstm_dim * 2, tag_dict[1])

    def forward(self, x_tokens, x_chars, x_mask, x_char_mask,
                x_pos=None, x_raw=None, x_bin=None, x_tags=None):
        batch_size, seq_len, char_len = x_chars.size()

        # Assemble and concatenate all the inputs
        inputs = []

        # First snag the token embeddings
        inputs.append(self.glove_emb(x_tokens))
        inputs.append(self.twitter_emb(x_tokens))

        # Then we get the character level embed/CNN
        char_emb = self.char_emb(x_chars.view(-1, char_len))
        char_emb = char_emb.view(batch_size, seq_len, char_len, -1)
        inputs.append(self.char_cnn(char_emb))

        # Elmo!
        if self.args.use_elmo:
            inputs.append(self.elmo(x_raw))

        # POS!
        if self.args.use_pos:
            inputs.append(self.pos_emb(x_pos))

        # Binary features!
        if x_bin is not None:
            inputs.append(x_bin)

        embeddings = torch.cat(inputs, dim=-1)

        if self.args.dropout_emb:
            embeddings = F.dropout(embeddings, self.args.dropout_emb,
                                   self.training)

        # Encode with LSTM
        hiddens = self.lstm(embeddings).contiguous()

        # Predict with CRF
        tags, likelihood = self.crf(hiddens, x_mask, x_tags)

        return tags, likelihood

    def vectorize(self, words, pos, tags):
        tok_vecs = self.vectorize_words(words, self.word_dict)
        pos_vecs = self.vectorize_words(pos, self.pos_dict)
        char_vecs = self.vectorize_chars(words)
        tags = self.vectorize_words(tags, self.tag_dict[0])
        return tok_vecs, pos_vecs, char_vecs, words, tags

    def batchify(self, batch):
        tok_batch, tok_mask = self.batch_words([b[0] for b in batch])
        pos_batch = self.batch_words([b[1] for b in batch])[0]
        char_batch, char_mask = self.batch_chars([b[2] for b in batch])
        raw_batch = [b[3] for b in batch]
        tag_batch = self.batch_words([b[4] for b in batch])[0]

        inputs = {'x_tokens': tok_batch,
                  'x_pos': pos_batch,
                  'x_chars': char_batch,
                  'x_raw': raw_batch,
                  'x_mask': tok_mask,
                  'x_char_mask': char_mask,
                  'x_tags': tag_batch}
        targets = tag_batch

        return inputs, targets

    def regularize(self):
        loss = None
        if self.args.translate_emb:
            emb_loss = self.glove_emb.weight.weight.abs().sum()
            emb_loss += self.twitter_emb.weight.weight.abs().sum()
            emb_loss *= self.args.emb_lambda
            if loss is None:
                loss = emb_loss
            else:
                loss += emb_loss
        return loss

    def save(self, filename):
        state = {
            'state': self.state_dict(),
            'word_dict': self.word_dict,
            'tag_dict': self.tag_dict,
            'pos_dict': self.pos_dict,
            'args': self.args,
            'binary_dict': self.binary_dict,
        }
        torch.save(state, filename)

    @classmethod
    def load(cls, filename):
        saved = torch.load(filename)
        state = saved.pop('state')
        module = cls(**saved)
        module.load_state_dict(state)
        return module
