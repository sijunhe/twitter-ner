"""Dictionary classes."""

import logging
import nltk
import regex as re
from collections import defaultdict
from sklearn.utils import murmurhash3_32
from torch.utils.data import Dataset

logger = logging.getLogger()


class Dictionary(object):
    NUMBER = re.compile(r'^\d+$')
    URL = re.compile(r'^http[s]?://', flag=re.IGNORECASE)
    HANDLE = re.compile(r'@([A-Za-z0-9_]+)')
    FIRST_CAP = re.compile(r'^[A-Z][a-z]+')
    APOS = re.compile(r'(\w+)\'s')

    def __init__(self, num_unk=100, special_tokens=True, cased=False):
        self.num_unk = num_unk
        self.special_tokens = special_tokens
        self.cased = cased
        self._ind2tok = {0: '<NULL>', 1: '<NUM>', 2: '<HANDLE>',
                         3: '<URL>', 4: '<FCAP>', 5: '<APOS>'}
        self._tok2ind = {'<NULL>': 0, '<NUM>': 1, '<HANDLE>': 2,
                         '<URL>': 3, '<FCAP>': 4, '<APOS>': 5}
        self._tok2freq = defaultdict(int)
        for i in range(self.num_unk):
            unk = '<UNK_%d>' % i
            self._tok2ind[unk] = len(self._ind2tok)
            self._ind2tok[len(self._ind2tok)] = unk
        self._start_idx = len(self._tok2ind)

    def __len__(self):
        return len(self._ind2tok)

    def __contains__(self, key):
        if type(key) == int:
            return key in self._ind2tok
        elif type(key) == str:
            return self._resolve(key) in self._tok2ind
        raise KeyError('Expecting string or int, got %s' % type(key))

    def __getitem__(self, key):
        if type(key) == int:
            return self._ind2tok[key]
        elif type(key) == str:
            key = self._resolve(key)
            if key not in self._tok2ind:
                key = '<UNK_%d>' % self._hash(key)
            return self._tok2ind[key]
        raise KeyError('Expecting string or int, got %s' % type(key))

    def add(self, token):
        token = self._resolve(token)
        self._tok2freq[token] += 1
        if token not in self._tok2ind:
            self._tok2ind[token] = len(self._ind2tok)
            self._ind2tok[len(self._ind2tok)] = token

    def add_list(self, tokens, vocab_file=None):
        if vocab_file:
            vocab = set()
            with open(vocab_file) as f:
                for line in f:
                    vocab.add(line.split()[0])
        for token in tokens:
            token = self._resolve(token)
            if vocab_file and token not in vocab:
                continue
            self._tok2freq[token] += 1
            if token not in self._tok2ind:
                self._tok2ind[token] = len(self._ind2tok)
                self._ind2tok[len(self._ind2tok)] = token

    def prune(self, min_freq):
        """Sort and remove all tokens that occur less than min_freq."""
        curr_size = len(self)
        sorted_pairs = sorted(self._tok2freq.items(), key=lambda x: (-x[1], x[0]))
        tok2ind = {}
        ind2tok = {}

        for i in range(self._start_idx):
            key = self._ind2tok[i]
            ind2tok[i] = key
            tok2ind[key] = i

        for key, freq in sorted_pairs:
            if self._tok2ind[key] >= self._start_idx:
                if freq > min_freq:
                    index = len(tok2ind)
                    tok2ind[key] = index
                    ind2tok[index] = key
                else:
                    del self._tok2freq[key]

        self._tok2ind = tok2ind
        self._ind2tok = ind2tok
        logger.info('Pruned dictionary from %d to %d' % (curr_size, len(self)))

    def _hash(self, token):
        return murmurhash3_32(token, positive=True) % self.num_unk

    def _resolve(self, token):
        if not self.cased:
            token = token.lower()
        if self.special_tokens:
            if self.NUMBER.match(token):
                token = '<NUM>'
            elif self.URL.match(token):
                token = '<URL>'
            elif self.HANDLE.match(token):
                token = '<HANDLE>'
            elif self.FIRST_CAP.match(token):
                token = '<FCAP>'
            elif self.APOS.match(token):
                token = '<APOS>'
        return token


class SequenceTaggingDataset(Dataset):

    def __init__(self, examples, vectorize_fn):
        self.vectorize_fn = vectorize_fn
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        tokens = self.examples[index]['tokens']
        pos = self.examples[index]['pos']
        tags = self.examples[index]['tags']
        return self.vectorize_fn(tokens, pos, tags)


def load_data(filename):
    tweets = []
    with open(filename) as fp:
        tweet = {'tokens': [], 'tags': []}
        for line in fp:
            line = line.strip()
            if line:
                fields = line.split('\t')
                word = ' '.join(fields[0:-1]).strip()
                if not word:
                    word = '<UNK>'
                label = fields[-1] if len(fields) > 1 else 'O'
                tweet['tokens'].append(word)
                tweet['tags'].append(label)
            else:
                tweet['pos'] = [p[1] for p in nltk.pos_tag(tweet['tokens'])]
                tweets.append(tweet)
                tweet = {'tokens': [], 'tags': []}
    return tweets

def load_test_data(filename):
    tweets = []
    with open(filename) as fp:
        tweet = {'tokens': [], 'tags': []}
        for line in fp:
            line = line.strip()
            if line:
                tweet['tokens'].append(line)
                tweet['tags'].append('O')
            else:
                tweet['pos'] = [p[1] for p in nltk.pos_tag(tweet['tokens'])]
                tweets.append(tweet)
                tweet = {'tokens': [], 'tags': []}
    return tweets
