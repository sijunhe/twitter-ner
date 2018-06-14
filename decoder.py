import numpy as np
import nltk
from collections import Counter
import random

from data_extractor import build_features_viterbi, extract

epsilon = 10 ** -50

unique_tags = ['O', '#', '$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS',
               'LS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM',
               'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB', '``']
postag2idx = {tag:idx for idx, tag in enumerate(unique_tags)}
num_states = 5  # 3 ner tags + SOS
tag2idx = {'B': 0, 'I': 1, 'O': 2, '<SOS>': 3, '<EOS>': 4}
idx2tag = {0: 'B', 1 : 'I', 2: 'O', 3: '<SOS>', 4: '<EOS>'}


class HMMDataGenerator:

    def __init__(self, train_data):
        self.train_data = train_data
        self.vocab = self.generate_vocab()
        self.vocab2idx = {word:i for i, word in enumerate(self.vocab)}
        self.tag_counter = self.count_ner_tags()

    def count_ner_tags(self):
        tags = []
        for tweet in self.train_data:
            for tok in tweet:
                tags.append(tok[1])
        return Counter(tags)

    def generate_vocab(self):

        # get count of words
        word_cnt = {}
        for tweet in self.train_data:
            for token in tweet:
                tok = token[0]
                if tok not in word_cnt:
                    word_cnt[tok] = 0
                    word_cnt[tok] += 1

        # filter words by frequency
        filtered_vocab = self.filter_words_by_frequency(word_cnt, list(word_cnt.keys()), frequency_threshold=1)

        return filtered_vocab

    def filter_words_by_frequency(self, word_counter, words, frequency_threshold=1):
        previous_word, transformed_list = None, []
        for word in words:
            if word_counter[word] > frequency_threshold:
                transformed_list.append(word)
            else:
                if random.random() < 0.05:
                    transformed_list.append(find_pattern(previous_word, word))
                else:
                    transformed_list.append(word)

            previous_word = word
        return transformed_list


    def get_emission_prob(self, k_smoothing_value):

        tag_word_pairs = []
        for tweet in self.train_data:
            tag_word_pairs.extend([('<SOS>', '<SOS>')])
            tag_word_pairs.extend(pair for pair in tweet)
            tag_word_pairs.extend([('<EOS>', '<EOS>')])

        tag_word_counter = Counter(tag_word_pairs)
        unique_words = self.vocab + ['<SOS>', '<EOS>']
        word_smoothing_normalization_constant = k_smoothing_value * len(unique_words)

        # emission_probabilities = np.zeros((len(unique_words),num_states))
        # for tweet in self.train_data:
        #     prev_token = '<SOS>'
        #     emission_probabilities[tag2idx['<SOS>'], tag2idx['<SOS>']] += 1
        #     for i, token in enumerate(tweet):
        #         tag_idx = tag2idx[token[1]]
        #         tok = token[0]
        #         if tok in self.vocab:
        #             vocab_idx = self.vocab2idx[tok]
        #         else:
        #             vocab_idx = self.vocab2idx[find_pattern(prev_token, tok)]
        #         emission_probabilities[vocab_idx, tag_idx] += 1
        #         prev_token = tok
        #     emission_probabilities[tag2idx['<EOS>'], tag2idx['<EOS>']] += 1

        emission_probabilities = np.array([[(tag_word_counter[(word, tag)] + k_smoothing_value) /
                                            (self.tag_counter[tag] + word_smoothing_normalization_constant) for tag in
                                            ['B', 'I', 'O', '<SOS>', '<EOS>']] for word in unique_words])

        emission_probabilities = np.log(emission_probabilities + epsilon).T

        return emission_probabilities

    def get_transition_prob(self):
        transition_prob = np.zeros((num_states, num_states))
        for tweet in self.train_data:
            for i, token in enumerate(tweet):
                j = i - 1
                if j < 0:
                    prev_state = tag2idx['<SOS>']
                else:
                    prev_state = tag2idx[tweet[j][1]]
                curr_state = tag2idx[tweet[i][1]]
                transition_prob[prev_state, curr_state] += 1
            transition_prob[curr_state, tag2idx['<EOS>']] += 1

        for r in range(len(transition_prob)):
            transition_prob[r, :] = (transition_prob[r, :]) / sum(transition_prob[r, :] + epsilon)

        transition_prob = np.log(transition_prob + epsilon)

        return transition_prob


class Trellis:

    def __init__(self, train_data):
        self.tag2idx = {'B': 0, 'I': 1, 'O': 2, '<SOS>': 3, '<EOS>': 4}
        self.transition_from_map = {
            0: np.log(np.array([0, 0, 1, 1, 0]) + epsilon),
            1: np.log(np.array([1, 0, 0, 0, 0]) + epsilon),
            2: np.log(np.array([1, 1, 1, 1, 0]) + epsilon),
            3: np.log(np.array([0, 0, 0, 0, 0]) + epsilon),
            4: np.log(np.array([1, 1, 1, 0, 0]) + epsilon)
        }
        self.train_data = train_data
        self.data_generator = HMMDataGenerator(self.train_data)
        self.vocab = self.data_generator.vocab
        self.vocab2idx = self.data_generator.vocab2idx
        self.emis = self.data_generator.get_emission_prob(k_smoothing_value=1)
        self.trans = self.data_generator.get_transition_prob()


    def viterbi_generative(self, sentence, gram = 2):
        sentence = sentence + ['<EOS>']
        self.bp = np.zeros((num_states, len(sentence)+1))
        self.pi = np.zeros((num_states, len(sentence)+1)) + epsilon

        self.pi[tag2idx['<SOS>']][0] = 1 + epsilon
        self.pi = np.log(self.pi)
        self.bp[:, 0] = None

        prev_word = None
        for word_num, word in enumerate(sentence):
            for tag in range(0, self.pi.shape[0]):

                from_map = self.transition_from_map[tag]
                if word in self.vocab:
                    emission_tag_idx = self.vocab2idx[word]
                else:
                    emission_tag_idx = self.vocab2idx[find_pattern(prev_word, word)]
                scores = (self.pi[:, word_num] + self.trans[tag, :] + self.emis[tag, emission_tag_idx]) + from_map

                max_idx = np.argmax(scores)
                self.bp[tag, word_num + 1] = max_idx
                self.pi[tag, word_num + 1] = scores[max_idx]

            prev_word = word

        likelihood = np.log(1 + epsilon)
        tags = []
        for i in range(1, self.bp.shape[1]):
            likelihood += max(self.pi[:, i])
            if gram == 2:
                tags.append(idx2tag[int(self.bp[np.argmax(self.pi[:, i]), i])])
            else:
                tags.append(self.tags_map[self.bp[np.argmax(self.pi[:, i]), i]][-1])

        return tags[1:]





class TrellisMEMM:

    def __init__(self, dev_data):
        self.dev_data = dev_data
        self.tag2idx = {'B': 0, 'I': 1, 'O': 2, '<SOS>': 3, '<EOS>': 4}
        self.transition_from_map = {
            0: np.log(np.array([0, 0, 1, 1, 0]) + epsilon),
            1: np.log(np.array([1, 0, 0, 0, 0]) + epsilon),
            2: np.log(np.array([1, 1, 1, 1, 0]) + epsilon),
            3: np.log(np.array([0, 0, 0, 0, 0]) + epsilon),
            4: np.log(np.array([1, 1, 1, 0, 0]) + epsilon)
        }
        self.transition_to_map = {
            0: np.log(np.array([0, 1, 1, 0, 1]) + epsilon),
            1: np.log(np.array([0, 1, 1, 0, 1]) + epsilon),
            2: np.log(np.array([1, 1, 1, 0, 1]) + epsilon),
            3: np.log(np.array([1, 0, 1, 0, 0]) + epsilon),
            4: np.log(np.array([0, 0, 0, 0, 0]) + epsilon)
        }

    def extract_features(self, token):
        return extract(token)

    def build_features(self, sentence, i, prev_ner_tag):

        feature = []

        # get previous token features
        # each "token" needs to be (token, POS)
        j = i - 1
        if j < 0:
            prev_token = ('<sos>', 'O')
        else:
            prev_token = sentence[i-1]
        feature.extend(self.extract_features(prev_token))

        # get current token features with previously generated ner tag
        feature.extend(self.extract_features(sentence[i]))

        # manually append previous tag
        feature.append(prev_ner_tag)

        return np.array(feature)

    def viterbi_memm(self, sentence, classifier, gram = 2):
        # sentence = sentence + [('<EOS>','<EOS>')]
        self.bp = np.zeros((num_states, len(sentence) + 1))
        self.pi = np.zeros((num_states, len(sentence) + 1)) + epsilon

        self.pi[tag2idx['<SOS>']][0] = 1 + epsilon
        self.pi = np.log(self.pi)
        self.bp[:, 0] = None

        prev_word = None
        for i, token in enumerate(sentence):
            scores = np.zeros((num_states, num_states))
            for tag in range(0, self.pi.shape[0]):

                to_map = self.transition_to_map[tag]
                features = np.array(self.build_features(sentence, i, tag))
                scores_ = np.append(classifier.predict_log_proba(features.reshape(1, -1)), np.log([epsilon, epsilon]))
                scores_ = scores_ + self.pi[:, i] + to_map
                scores[:, tag] = scores_

            max_idxs = np.argmax(scores, axis=0)
            for tag in range(0, self.pi.shape[0]):
                self.bp[tag, i + 1] = max_idxs[tag]
                self.pi[tag, i + 1] = scores[tag][max_idxs[tag]]


        likelihood = np.log(1 + epsilon)
        tags = []
        state = np.argmax(self.pi[:, -1])
        for i in range(self.bp.shape[1]-1, 0, -1):
            tags.append(idx2tag[state])
            state = int(self.bp[state, i])

        # for i in range(1, self.bp.shape[1]):
        #     likelihood += max(self.pi[:, i])
        #     if gram == 2:
        #         tags.append(idx2tag[int(self.bp[np.argmax(self.pi[:, i]), i])])
        #     else:
        #         tags.append(self.tags_map[self.bp[np.argmax(self.pi[:, i]), i]][-1])

        return tags



