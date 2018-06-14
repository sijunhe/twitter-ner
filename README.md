
# Named Entity Recognition on Twitter data

## Overview
The goal of this assignment is to build a named-entity recognizer for Twitter text - given a tweet, identify
sub-spans of words that represent named entities. The training, development and test sets were comprised of
2394, 959, and 2377 tweets respectively. The train and dev dataset consisted of Begin-Inside-Outside (BIO)
encoding for each word of the tweets and our task was to predict the BIO encoding for the test dataset.

## Dependencies
* Python (3.6.4+)
* Natural Language ToolKit (nltk)
* PyTorch (0.3.1)
* allennlp==0.4.1
* tqdm
* regex
* sklearn
* numpy

## How to run
```
usage: main.py [-h] [--max-chars MAX_CHARS] [--glove-dim GLOVE_DIM]
               [--use-glove USE_GLOVE] [--glove-file GLOVE_FILE]
               [--use-twitter USE_TWITTER] [--twitter-dim TWITTER_DIM]
               [--twitter-file TWITTER_FILE] [--use-pos USE_POS]
               [--pos-dim POS_DIM] [--use-elmo USE_ELMO] [--elmo-dim ELMO_DIM]
               [--elmo-dropout ELMO_DROPOUT] [--char-dim CHAR_DIM]
               [--char-cnn-filters CHAR_CNN_FILTERS]
               [--char-cnn-ngrams CHAR_CNN_NGRAMS [CHAR_CNN_NGRAMS ...]]
               [--lstm-dim LSTM_DIM] [--lstm-layers LSTM_LAYERS]
               [--lstm-dropout LSTM_DROPOUT] [--dropout-emb DROPOUT_EMB]
               [--emb-lambda EMB_LAMBDA] [--translate-emb TRANSLATE_EMB]
               [--load-model LOAD_MODEL] [--pretrained PRETRAINED]
               [--optimizer {adagrad,adamax,sgd,adam}]
               [--learning-rate LEARNING_RATE] [--num-epochs NUM_EPOCHS]
               [--momentum MOMENTUM] [--weight-decay WEIGHT_DECAY]
               [--train TRAIN] [--dev DEV] [--test TEST]
               [--min-word-freq MIN_WORD_FREQ] [--min-pos-freq MIN_POS_FREQ]
               [--batch-size BATCH_SIZE] [--data-workers DATA_WORKERS]
               [--cuda CUDA] [--random-seed RANDOM_SEED]
               [--model-name MODEL_NAME] [--save SAVE]
               [--write-test WRITE_TEST]


         
