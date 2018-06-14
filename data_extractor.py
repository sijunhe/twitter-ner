import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import pickle
import time
import nltk
from collections import defaultdict

from viterbi_utils import time_elapsed
from tqdm import tqdm

unique_tags = ['O', '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
               'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
               'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
postag2idx = {tag:idx for idx, tag in enumerate(unique_tags)}
nertag2idx = {'B': 0, 'I': 1, 'O': 2, '<SOS>': 3, '<EOS>': 4}

# --------------------------------------------------------------------------
# Regular Expression Class
# --------------------------------------------------------------------------
class RegExChecker:

    @staticmethod
    def isTwitterHandle(str):
        prog = re.compile(r'@([A-Za-z0-9_]+)')
        return prog.match(str)

    @staticmethod
    def isAllPunctuation(str):
        prog = re.compile(r'^[\.\,!\?"\':;_\-]{2,}$')
        return prog.match(str)

    @staticmethod
    def isAllUpperCase(str):
        prog = re.compile(r'^[A-Z][A-Z]+$')
        return prog.match(str)

    @staticmethod
    def isURL(str):
        prog = re.compile(r'^http[s]?://')
        return prog.match(str)

    @staticmethod
    def isNumber(str):
        prog = re.compile(r'^\d+$')
        return prog.match(str)

    @staticmethod
    def isFirstLetterCapital(str):
        prog = re.compile(r'^[A-Z][a-z]+')
        return prog.match(str)

    @staticmethod
    def endsWithApostropheS(str):
        prog = re.compile(r'(\w+)\'s')
        return prog.match(str)


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

# function that extracts features from a token
def preprocess(str):
    word = str
    if RegExChecker.isURL(str):
        word = '<url>'
    elif RegExChecker.isTwitterHandle(str):
        word = '<user>'
    elif RegExChecker.isNumber(str):
        word = '<num>'
    elif str == 'RT':
        word = 'rt'
    elif RegExChecker.isAllUpperCase(str):
         word = '<allcaps>'      
    return word


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

# function that reads from input file and creates in the format
# [[(w1,l1),(w2,l2)],[(w1,l1),(w2,l2),(w3,l3)]]
def load_data(filename, isTest = False):
    tweets = []
    with open(filename) as fp:
        tweet = []
        for line in fp:
            line = line.strip()
            if line:
                tokens = line.split('\t')
                word = preprocess(tokens[0])
                if isTest:
                    tweet.append((word, 'O'))
                else:
                    tweet.append((word, tokens[1]))
            else:
                tweets.append(tweet)
                tweet = []
    return tweets

# function that reads from input file and creates in the format
# [[(w1 w2 w3 w4),(l1 l2 l3 l4)],[(w1 w2),(l1 l2)]]
def load_data_sentences(filename,isTest=False):
    tweets = []
    tags = []
    with open(filename) as fp:
        tweet = []
        tag = []
        for line in fp:
            line = line.strip()
            if line:
                tokens = line.split('\t')
                word = tokens[0]
                if isTest:
                    tweet.append(word)
                    tag.append('O')
                else:    
                    tweet.append(word)
                    tag.append(tokens[1])
            else:
                tweets.append((' '.join(tweet),' '.join(tag)))
                tweet = []
                tag = []
    return tweets

# Function to get POS tags for each word 
# [[(w1 w2 w3 w4),(t1 t2 t3 t4),(l1 l2 l3 l4)],[(w1 w2),(t1 t2),(l1 l2)]]
def get_pos_bytweet(data):
    res = []
    for i, tweet in enumerate(data):
        res_tweet = []
        tweet_words = tweet[0].split(' ')
        ner_tags = nltk.word_tokenize(tweet[1])
        pos_tags = nltk.pos_tag(tweet_words)
        pos_tags = [p[1] for p in pos_tags]
        tweet_words = [preprocess(w) for w in tweet_words]
        tup = [tweet_words, pos_tags, ner_tags]
        res.append(list(zip(*tup)))
        tup = []
        if i % 200 == 0:
            print(i)
    return res

glove_data_file = 'data/glove/glove.twitter.27B.25d.txt'
EMB_DIM = 25

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    model['<sos>'] = np.random.randn(EMB_DIM)
    model['<eos>']= np.random.randn(EMB_DIM)
    print( "Done.")
    return model

words = loadGloveModel(glove_data_file)

COUNTRIES = pd.read_table('data/lexicon/location.country.txt',header=None)
MOVIES = pd.read_table('data/lexicon/movies.txt', header=None)
MUSIC_ARTISTS = pd.read_table('data/lexicon/music_artists.txt', header=None)


# --------------------------------------------------------------------------
# Feature Building
# --------------------------------------------------------------------------

def isCountry(str):
    if str in COUNTRIES.values:
        return True

def isMovie(str):
    if str in MOVIES.values:
        return True

def isArtist(str):
    if str in MUSIC_ARTISTS.values:
        return True

# function that extracts features from a token
def extract(tweet):

    features = np.zeros(6)
    word = tweet[0].lower()
    try:
        emb = words[word]
    except:
        emb = np.random.randn(EMB_DIM)
        words[word] = emb
    word = tweet[0]
    if isCountry(word):
        features[0] = 1
    if isMovie(word):
        features[1] = 1
    if isArtist(word):
        features[2] = 1
    if RegExChecker.isFirstLetterCapital(word):
        features[3] = 1
    if RegExChecker.endsWithApostropheS(word):
        features[4] = 1
    features[5] = postag2idx[tweet[1]]
    # Append the other features to the embeddings
    feat_vect = np.concatenate((emb, features), axis=0)
    return feat_vect

# function that builds features from tweet tokens
def build_features(tweet, i):
    features = []
    for j in range(i-1,i+2):
        if (j < 0):
            token = ('<sos>', 'O', 'O')
        elif (j >= len(tweet)):
            token = ('<eos>', 'O', 'O')
        else:
            token = tweet[j][0:2]
        j_features = extract(token)
        features.extend(j_features)
    return features

# function that builds features from tweet tokens for memm viterbi algo
def build_features_viterbi(tweet, i):
    features = []
    prev_ner_tag = None
    for j in range(i-1, i+1):
        if (j < 0):
            token = ('<sos>','O','O')
        elif (j >= len(tweet)):
            token = ('<eos>','O','O')
        else:
            token = tweet[j]
        if not prev_ner_tag:
            prev_ner_tag = nertag2idx[token[2]]
        j_features = extract(token)
        features.extend(j_features)

    features.append(prev_ner_tag)

    return features

def extract_features(data, viterbi=False):
    X = []
    y = []
    for idx , tweet in enumerate(tqdm(data)):
        for i, pair in enumerate(tweet):
            if viterbi:
                X.append(build_features_viterbi(tweet, i))
            else:
                X.append(build_features(tweet, i))
            y.append(pair[2])

    return X, y


# --------------------------------------------------------------------------
# Public Interface for getting data
# --------------------------------------------------------------------------

def get_processed_data(filename, save_filename, viterbi = False):

    # load pickle file
    data = pickle.load(open(filename, 'rb'))
    # features = pickle.load(open('data/pos_pickle_file/dev_features_pos.p', 'rb'))

    # extract features
    features, y = extract_features(data, viterbi)

    # save data
    data = (features, y)
    pickle.dump(data, open(save_filename, 'wb'))

    return features, y


def get_data(from_filename, to_filename, isTest=False, load=False, viterbi=False):

    start_time = time.time()
    if load:
        (X, y), tweets = pickle.load(open(from_filename, 'rb'))
        start_time = time_elapsed(start_time, 'Loaded Data From File')

    else:

        # load and process the data
        tweets = load_data(from_filename, isTest)
        start_time = time_elapsed(start_time, 'Loaded Data')

        # build features
        X, y = extract_features(tweets, viterbi)
        start_time = time_elapsed(start_time, 'Extracted Features')

        # save data
        data = ((X, y), tweets)
        pickle.dump(data, open(to_filename, 'wb'))
        start_time = time_elapsed(start_time, 'Saved Data')

    return (X, y), tweets





