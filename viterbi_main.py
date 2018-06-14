import pickle
from data_extractor import get_data, get_processed_data
from viterbi_utils import generate_outfile
from viterbi_utils import evaluate_dev, read_predictions
from decoder import Trellis, TrellisMEMM
import numpy as np
import tageval
from tqdm import tqdm
# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# Load Data
# --------------------------------------------------------------------------
# (trainX, trainY), train_tweets = get_data('data/train/train.txt', 'processed_data/train_data.pkl')
# (devX, devY), dev_tweets = get_data('data/dev/dev.txt', 'processed_data/dev_data.pkl')
# test_data = get_data('data/test/test.nolabels.txt', 'processed_data/test_data.pkl', isTest=True)

# (trainX, trainY), train_tweets = get_processed_data('processed_data/train_data.pkl', None, load=True)
# (devX, devY), dev_tweets = get_processed_data('processed_data/dev_data.pkl', None, load=True)

# --------------------------------------------------------------------------
# Train
# --------------------------------------------------------------------------
model = None


# --------------------------------------------------------------------------
# Decode
# --------------------------------------------------------------------------
dev_tweets = pickle.load(open('data/pos_pickle_file/dev_data_pos.p', 'rb'))
clf = pickle.load(open('processed_data/mlp_viterbi.pkl', 'rb'))
trel = TrellisMEMM(dev_tweets)
dev_tags = []
for tweet in tqdm(dev_tweets):
    tweet_pos_tokens = [tok[0:2] for tok in tweet]
    predicted_tags = trel.viterbi_memm(tweet_pos_tokens, clf)
    print(predicted_tags)
    dev_tags.append(predicted_tags)

f1 = tageval.evaluate_tagging('data/dev/dev.out', dev_tags)
print('F1: {}'.format(f1))


# --------------------------------------------------------------------------
# Evaluate
# --------------------------------------------------------------------------
# tag2idx = {'B': 0, 'I': 1, 'O': 2}
# dev_predictions = read_predictions('exp/2/dev_predictions.out')
# evaluate_dev(dev_tweets, dev_predictions)