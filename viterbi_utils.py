
import numpy as np
import time
import re
from sklearn.metrics import f1_score

np.set_printoptions(suppress=True)

tag2idx = {'B': 0, 'I': 1, 'O': 2}
idx2tag = {v: k for k, v in tag2idx.items()}

patterns_to_check = [("twoDigitNum", re.compile("^(\d{2})$")), ("fourDigitNum", re.compile("^(\d{4})$")),
                     ("containsDigitAndAlpha", re.compile("^(?=.*[a-zA-Z])(?=.*\d).+$")),
                     ("containsDigitAndDash", re.compile("^(?=.*[-])(?=.*\d).+$")),
                     ("containsDigitAndSlash", re.compile("^(?=.*[/])(?=.*\d).+$")),
                     ("containsDigitAndComma", re.compile("^(?=.*[,])(?=.*\d).+$")),
                     ("containsDigitAndPeriod", re.compile("^(?=.*[.])(?=.*\d).+$")),
                     ("containsDigitAndColon", re.compile("^(?=.*[:])(?=.*\d).+$")),
                     ("othernum", re.compile("^(\d+)$")), ("allCaps", re.compile("^([A-Z]+)$")),
                     ("capPeriod", re.compile("^([A-Z]+)\.$"))]

firstCharCaps = re.compile("^([A-Z][a-zA-Z]*)$")

more_patterns_to_check = [("firstCharCapsDash", re.compile("^[A-Z](?=.*[a-z])(?=.*-).+$")),
                          ("containsCapsAndPeriod", re.compile("^(?=.*[.])(?=.*[A-Z]).+$")),
                          ("lowercase", re.compile("^([a-z]*)$")),
                          ("lowercaseDash", re.compile("^(?=.*[a-z])(?=.*-).+$")),
                          ("capitalizedDash", re.compile("^(?=.*[A-Z])(?=.*-).+$")),
                          ("lowercaseSlash", re.compile("^(?=.*[a-z])(?=.*/).+$")),
                          ("uppercaseApostropheuppercase", re.compile("^([A-Z][\'][A-Z][a-z]*).+$")),
                          ("upperANDupper", re.compile("^[A-Z][&][A-Z].*$"))]


def get_indices(filename):
    with open(filename) as fp:
        idx = []
        i = 0
        for line in fp:
            line = line.strip()
            i += 1
            if line:
                pass
            else:
                idx.append(i)
    return idx


def generate_outfile(outputs, mode='dev'):
    if mode == 'dev':
        indices = get_indices('data/dev/dev.nolabels.txt')
        output_file = 'dev_predictions.out'
    else:
        indices = get_indices('data/test/test.nolabels.txt')
        output_file = 'test_predictions.out'
    curr_idx = 0
    curr = 1
    with open(output_file, 'w') as fp:
        for pred in outputs:
            if curr == indices[curr_idx]:
                fp.write('\n')
                curr += 1
                curr_idx += 1
            fp.write(pred)
            fp.write('\n')
            curr += 1
        fp.close()


def evaluate_dev(dev_data, predictions):
    # dev_data: list of lists of (token, label)
    # predictions: list of lists of predicted tags for each token
    pidx = 0
    cmatrix = np.zeros((3, 3))
    dev_labels = []
    for tidx, tweet in enumerate(dev_data):
        preds = predictions[pidx:pidx + len(tweet)]
        sentence = [tok[0] for tok in tweet]
        ner_tags = [tok[1] for tok in tweet]
        dev_labels.extend([tag2idx[t] for t in ner_tags])

        all_right = True
        for i in range(len(preds)):
            ner_tag_idx, p = tag2idx[ner_tags[i]], int(preds[i])
            if p != ner_tag_idx:
                all_right = False
            cmatrix[p][ner_tag_idx] += 1

        if not all_right:
            print(sentence)
            print(ner_tags)
            print([idx2tag[int(pred)] for pred in preds])
            print('\n')
        pidx += len(tweet)

    print("Confusion Matrix: ")
    print(cmatrix)

    # dev_score = f1_score(dev_labels, [int(p) for p in predictions])
    # print('Dev F1 Score... ' + str(dev_score))


def read_predictions(filename):
    predictions = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue
            label = tag2idx[line]
            predictions.append(label)
    return predictions


def time_elapsed(start_time, msg=''):
    time_now = time.time()
    print('[ time elapsed: {} ] '.format(time_now - start_time) + msg)
    return time_now


def find_pattern(previous_word, word):
    for tag, pattern in patterns_to_check:
        if pattern.match(word):
            return "UNK_" + tag

    # firstCharCaps needs special treatment
    if firstCharCaps.match(word):
        if previous_word is not None and previous_word in [".", "-DOCSTART-"]:
            return "UNK_firstCharCaps"
        return "UNK_initCap"

    for tag, pattern in more_patterns_to_check:
        if pattern.match(word):
            return "UNK_" + tag

    return "UNK_unmatched"
