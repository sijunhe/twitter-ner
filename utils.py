"""Utilities"""
import torch
from torch.autograd import Variable
import time

# ------------------------------------------------------------------------------
# Convenience functions
# ------------------------------------------------------------------------------


def wrap_variables(inputs, cuda=False, volatile=False):
    """Walk input, converting to Variable"""
    if isinstance(inputs, list) or isinstance(inputs, tuple):
        converted = []
        for i in iter(inputs):
            converted.append(wrap_variables(i, cuda, volatile))
        return converted
    elif isinstance(inputs, dict):
        converted = {}
        for k, v in inputs.items():
            converted[k] = wrap_variables(v, cuda, volatile)
        return converted
    else:
        if not torch.is_tensor(inputs):
            return inputs
        if cuda:
            try:
                inputs = inputs.cuda(async=True)
            except AttributeError:
                pass
        return Variable(inputs, volatile=volatile)


# ------------------------------------------------------------------------------
# Metric keeping
# ------------------------------------------------------------------------------


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.new = True

    def update(self, val, n=1):
        self.new = False
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        return "{:0.2f}".format(self.avg) if not self.new else "N/A"


# ------------------------------------------------------------------------------
# Metric keeping
# ------------------------------------------------------------------------------


def get_indices(filename):
    with open(filename) as fp:
        idx = []
        i = 0
        for line in fp:
            line = line.strip()
            i+=1
            if line:
                pass
            else:
                idx.append(i)
    return idx


def generate_outfile(outputs, outfile):

    with open(outfile, 'w') as fp:
        for tweet in outputs:
            for tag in tweet:
                fp.write(tag)
                fp.write('\n')
            fp.write('\n')


def time_elapsed(start_time, msg=''):
    time_now = time.time()
    print('[ time elapsed: {} ] '.format(time_now - start_time) + msg)
    return time_now
