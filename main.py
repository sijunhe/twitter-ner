"""Main training script for Twitter NER"""
import logging
import pathlib
import argparse
import json
import sys
import os
import numpy as np
from tqdm import tqdm
import time

import dataset
import utils
import tageval
from cnn_lstm_crf import CnnLstmCrf

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader




# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

logger = logging.getLogger()


def get_cmdline_args():
    # Helper to get boolean types
    def _str2bool(v):
        return v.lower() in ('yes', 'true', 't', '1', 'y')

    parser = argparse.ArgumentParser(
        description='NER',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.register('type', 'bool', _str2bool)

    CnnLstmCrf.add_cmdline_args(parser)

    model_args = parser.add_argument_group('Architecture')
    model_args.add_argument('--load-model', type='bool', default=False)
    model_args.add_argument('--pretrained', type=str, default=None)

    optim_args = parser.add_argument_group('Optimization')
    optim_args.add_argument('--optimizer', type=str, default='adam',
                            choices={'sgd', 'adam', 'adagrad', 'adamax'})
    optim_args.add_argument('--learning-rate', type=float, default=.001)
    optim_args.add_argument('--num-epochs', type=int, default=10)
    optim_args.add_argument('--momentum', type=float, default=0)
    optim_args.add_argument('--weight-decay', type=float, default=0)

    data_args = parser.add_argument_group('Data Provider')
    data_args.add_argument('--train', type=str, default='data/train/train.txt')
    data_args.add_argument('--dev', type=str, default='data/dev/dev.txt')
    data_args.add_argument('--test', type=str, default='data/test/test.nolabels.txt')

    data_args.add_argument('--min-word-freq', type=int, default=0)
    data_args.add_argument('--min-pos-freq', type=int, default=0)
    data_args.add_argument('--batch-size', type=int, default=16)
    data_args.add_argument('--data-workers', type=int, default=4)

    runtime_args = parser.add_argument_group('Runtime Environment')
    runtime_args.add_argument('--cuda', type='bool', default=True)
    runtime_args.add_argument('--random-seed', type=int, default=1013)

    file_args = parser.add_argument_group('Filesystem')
    file_args.add_argument('--model-name', type=str, default='test')
    file_args.add_argument('--save', type='bool', default=True)
    file_args.add_argument('--write-test', type='bool', default=False)


    return parser.parse_args()


# ------------------------------------------------------------------------------
# Train and validation helpers
# ------------------------------------------------------------------------------


def train(args, model, data_loader, optimizer):
    """Run one epoch of supervised training."""
    # Keep track of average accuracy and loss
    avg_loss = utils.AverageMeter()

    # Set training mode
    model.train()

    # for inputs, target in tqdm(data_loader):
    for inputs, targets in tqdm(data_loader):
        # Prep
        inputs = utils.wrap_variables(inputs, cuda=args.cuda)
        targets = utils.wrap_variables(targets, cuda=args.cuda)

        # Run forward
        predictions, log_likelihood = model(**inputs)

        # Loss = -NLL
        loss = -log_likelihood
        avg_loss.update(loss.data[0], len(inputs['x_tags']))

        # Run backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return {'loss': avg_loss}


def validate(args, gold, model, data_loader, test=False):
    """Make and score predictions over a validation set."""
    # Keep track of average accuracy and loss
    avg_loss = utils.AverageMeter()

    # Set eval mode
    model.eval()

    tags = []
    # for inputs, target in tqdm(data_loader):
    for inputs, targets in tqdm(data_loader):
        # Prep
        inputs = utils.wrap_variables(inputs, cuda=args.cuda)
        targets = utils.wrap_variables(targets, cuda=args.cuda)

        # Run forward
        predictions, log_likelihood = model(**inputs)

        # Loss = -NLL
        loss = -log_likelihood
        avg_loss.update(loss.data[0], len(inputs['x_tags']))

        for ex in predictions:
            tags.append([model.tag_dict[1][i] for i in ex])

    if test:
        f1 = 0
    else:
        f1 = tageval.evaluate_tagging(args.dev, tags)

    return {'f1': f1, 'loss': avg_loss, 'tags': tags}


def initialize_optimizer(model, args):
    """Set up optimizer with appropriate hyper parameters."""
    parameters = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(parameters, args.learning_rate,
                              momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(parameters, args.learning_rate,
                               weight_decay=args.weight_decay)
    elif args.optimizer == 'adamax':
        optimizer = optim.Adamax(parameters, args.learning_rate,
                                 weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(parameters, args.learning_rate,
                                  weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    return optimizer, scheduler


# ------------------------------------------------------------------------------
# Main training loop
# ------------------------------------------------------------------------------


def main(args):
    # --------------------------------------------------------------------------
    # Load Data
    # --------------------------------------------------------------------------
    logger.info('-' * 50)
    logger.info('Loading data...')
    train_tweets = dataset.load_data(args.train)
    dev_tweets = dataset.load_data(args.dev)
    # test_tweets = dataset.load_test_data(args.test)

    word_dict = dataset.Dictionary()
    pos_dict = dataset.Dictionary(num_unk=1, special_tokens=False, cased=True)
    tag_dict = ({'O': 0, 'B': 1, 'I': 2}, {0: 'O', 1: 'B', 2: 'I'})
    for tweet in train_tweets + dev_tweets:
        for w in tweet['tokens']:
            word_dict.add(w)
        for p in tweet['pos']:
            pos_dict.add(p)
    word_dict.prune(args.min_word_freq)
    pos_dict.prune(args.min_pos_freq)
    test_tweets = dataset.load_data(args.test)

    # --------------------------------------------------------------------------
    # Initialize Model
    # --------------------------------------------------------------------------
    logger.info('-' * 50)
    logger.info('Initializing model...')

    if args.pretrained:
        model = CnnLstmCrf.load(args.pretrained)
    elif args.load_model:
        model = CnnLstmCrf.load(args.model_file)
    else:
        word_dict = dataset.Dictionary()
        pos_dict = dataset.Dictionary(num_unk=1, special_tokens=False,
                                      cased=True)
        tag_dict = ({'O': 0, 'B': 1, 'I': 2}, {0: 'O', 1: 'B', 2: 'I'})
        for tweet in train_tweets + dev_tweets:
            for w in tweet['tokens']:
                word_dict.add(w)
            for p in tweet['pos']:
                pos_dict.add(p)
        word_dict.prune(args.min_word_freq)
        pos_dict.prune(args.min_pos_freq)
        model = CnnLstmCrf(args, word_dict, tag_dict, pos_dict)

    logger.info('\n%s' % model)

    # --------------------------------------------------------------------------
    # Transfer to cuda
    # --------------------------------------------------------------------------
    if args.cuda:
        logger.info('Setting up CUDA...')
        model = model.cuda()
    else:
        logger.info('Running on CPU only.')

    # --------------------------------------------------------------------------
    # Get data loaders
    # --------------------------------------------------------------------------
    train_loader = DataLoader(
        dataset.SequenceTaggingDataset(train_tweets, model.vectorize),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.cuda,
        collate_fn=model.batchify,
        num_workers=args.data_workers,
    )
    dev_loader = DataLoader(
        dataset.SequenceTaggingDataset(dev_tweets, model.vectorize),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.cuda,
        collate_fn=model.batchify,
        num_workers=args.data_workers,
    )
    test_loader = DataLoader(
        dataset.SequenceTaggingDataset(test_tweets, model.vectorize),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.cuda,
        collate_fn=model.batchify,
        num_workers=args.data_workers,
    )

    # --------------------------------------------------------------------------
    # Begin training
    # --------------------------------------------------------------------------
    logger.info('-' * 50)
    logger.info('Initializing optimizer...')
    optimizer, scheduler = initialize_optimizer(model, args)

    logger.info('-' * 50)
    logger.info('Starting training...')
    best_f1 = 0

    start_time = time.time()

    for epoch in range(1, args.num_epochs + 1):
        logger.info('-' * 50)

        # Run one epoch
        metrics = train(args, model, train_loader, optimizer)
        logger.info('Train | Epoch = %d | loss = %s' % (epoch, metrics['loss']))

        metrics = validate(args, args.dev, model, dev_loader)
        logger.info('Dev | Epoch = %d | loss = %s | f1 = %.4f' %
                    (epoch, metrics['loss'], metrics['f1']))

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            logger.info('Best validation so far: %2.4f vs %2.4f' %
                        (metrics['f1'], best_f1))
            utils.generate_outfile(metrics['tags'], 'dev_predictions.out')

            if args.model_file:
                logger.info('Saving best model to %s' % args.model_file)
                model.save(args.model_file)

            # generate new test file
            if args.write_test:
                metrics_tst = validate(args, args.test, model, test_loader, test=True)
                logger.info('New Best. Writing Test to File...')
                utils.generate_outfile(metrics_tst['tags'], 'test_predictions.out')

        scheduler.step(metrics['loss'].avg)

    utils.time_elapsed(start_time, "TOTAL TRAINING TIME")
    logger.info('BEST F1: {}'.format(best_f1))

# ------------------------------------------------------------------------------
# Launch script!
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Parse commandline arguments
    args = get_cmdline_args()

    # See if cuda is actually available...
    args.cuda = args.cuda and torch.cuda.is_available()

    # Set random state
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if args.cuda:
        torch.cuda.manual_seed(args.random_seed)

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s: [ %(message)s ]',
                            '%m/%d/%Y %I:%M:%S %p')
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)

    # Set up directory and file names
    args.model_file = None
    if args.model_name and (args.save or args.load_model):
        basedir = os.path.dirname(args.model_name)
        pathlib.Path(basedir).mkdir(parents=True, exist_ok=True)
        if args.save or args.load_model:
            args.model_file = args.model_name + '.mdl'

    # Logger.Info config
    logger.info('-' * 100)
    logger.info('COMMAND: %s' % ' '.join(sys.argv))
    logger.info('CONFIG:\n%s' %
                json.dumps(vars(args), indent=4, sort_keys=True))

    main(args)
