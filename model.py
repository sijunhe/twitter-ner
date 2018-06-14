import torch
import torch.nn as nn


class AbstractModel(nn.Module):
    """Abstract model wrapper."""

    @staticmethod
    def add_cmdline_args(parser):
        basic = parser.add_argument_group('Basic Model')
        basic.add_argument('--max-chars', type=int, default=20)

    @staticmethod
    def load_embeddings(module, filename, tok2ind):
        weights = module.embedding.weight.data
        with open(filename) as f:
            for line in f:
                line = line.rstrip().split(' ')
                tok, dims = line[0], line[1:]
                if len(dims) != weights.size(1):
                    import pdb
                    pdb.set_trace()
                    raise RuntimeError('Incompatible dimension.')
                if tok in tok2ind:
                    idx = tok2ind[tok]
                    weights[idx] = torch.Tensor([float(d) for d in dims])

    def init_optimizer(self, optimizer_fn):
        """Initialize an optimizer for the free parameters of the network."""
        parameters = [p for p in self.parameters() if p.requires_grad]
        self.optimizer = optimizer_fn(parameters)

    def vectorize_words(self, words, dictionary):
        """Helper method for vectorizing words."""
        indexed = torch.LongTensor([dictionary[t] for t in words])
        return indexed

    def vectorize_chars(self, words, min_length=2):
        """Helper method for vectorizing characters."""
        # Convert to bytes
        words = [[259] + list(t.encode('utf-8', 'ignore')) + [260]
                 for t in words]

        # Count max characters in words
        max_chars = max([len(t) for t in words])

        # Truncate if too long
        if self.args.max_chars > 0:
            max_chars = min(max_chars, self.args.max_chars)
            max_chars = max(max_chars, min_length)

        indexed = torch.LongTensor(len(words), max_chars).zero_()
        for i, w in enumerate(words):
            for j, c in enumerate(w[:max_chars]):
                indexed[i, j] = c + 1

        return indexed

    def batch_words(self, batch):
        """Helper method for batching words."""
        # Count max words in batch
        max_words = max([len(ex) for ex in batch])
        word_batch = torch.LongTensor(len(batch), max_words).zero_()

        # Fill
        for i, ex in enumerate(batch):
            word_batch[i, :len(ex)] = ex

        # Mask
        word_mask = 1 - word_batch.eq(0).long()

        return word_batch, word_mask

    def batch_chars(self, batch):
        """Helper method for batching characters."""
        # Count max words and chars in batch
        max_words = max([ex.size(0) for ex in batch])
        max_chars = max([ex.size(1) for ex in batch])
        char_batch = torch.LongTensor(len(batch), max_words, max_chars).zero_()

        # Fill
        for i, ex in enumerate(batch):
            char_batch[i, :ex.size(0), :ex.size(1)] = ex

        # Mask
        char_mask = 1 - char_batch.eq(0).long()

        return char_batch, char_mask

    def vectorize(self, ex):
        """Convert input to torch formats."""
        raise NotImplementedError

    def batchify(self, batch):
        """Convert vectorized inputs to a batch."""
        raise NotImplementedError
