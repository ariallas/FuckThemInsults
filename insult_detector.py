# -*- coding: utf-8 -*-
import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time
import pymorphy2
import pickle

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
from sklearn.svm.classes import LinearSVC
from sklearn.svm.classes import SVC
from sklearn.preprocessing import StandardScaler

import random

__author__ = 'tpc 2015'

word_regexp = re.compile(u'(?u)\w+|:\)+|;\)+|:\-\)+|;\-\)+|%%\)|=\)+|\(\(+|\)\)+|!+|\?|\+[0-9]+|\++')

def my_tokenizer(text, stop_words=None):
    tokens = word_regexp.findall(text.lower())
    filtered_tokens = []
    for token in tokens:
        ch = token[0]
        if stop_words is not None and token in stop_words:
            continue
        if ch == ':' or ch == ';' or ch == '=' or ch == '%%':
            token = ':)'
        elif ch == '(':
            token = '('
        elif ch == ')':
            token = ')'
        elif ch == '?':
            token = '?'
        elif ch == '!':
            token = '!'
        elif ch == '+':
            token = '+'
        elif '0' <= ch <= '9':
            continue
        filtered_tokens.append(token)
    return filtered_tokens


class InsultFeatures(TransformerMixin):
    def __init__(self, insult_words_regex, address_words_regex, weak_insult_words_regex):
        self.insult_words_regex = insult_words_regex
        self.address_words_regex = address_words_regex
        self.weak_insult_words_regex = weak_insult_words_regex

        self.insult_dict = dict()
        self.normal_dict = dict()

        self.mostly_insult_dict = dict()
        self.mostly_normal_dict = dict()

    def transform(self, texts):
        features = []

        # Some advanced level text processing!
        for text in texts:
            this_features = []

            insult_range = 0
            address_range = 0
            weak_insult_range = 0

            directed_insults = 0
            total_insults = 0
            token_count = 0
            total_addreses = 0

            total_adj = 0

            positive_weight = 0
            negative_weight = 0

            for parsed_token in text:
                token = parsed_token[0]
                tags = parsed_token[1]

                is_address = False
                is_insult = False
                is_weak_insult = False

                if 'ADJF' in tags or 'ADJS' in tags:
                    total_adj += 1

                if self.address_words_regex.match(token) or 'excl' in tags:
                    total_addreses += 1
                    is_address = True
                elif self.insult_words_regex.match(token):
                    is_insult = True
                elif self.weak_insult_words_regex.match(token):
                    is_weak_insult = True

                # Just insults, not super accurate..
                if is_insult or (address_range > 0 or insult_range > 0) and is_weak_insult:
                    total_insults += 1

                # More direct insults
                if \
                        insult_range > 0 and (is_insult or is_address or is_weak_insult) \
                        or address_range > 0 and (is_insult or is_weak_insult) \
                        or weak_insult_range > 0 and (is_insult or is_address):
                    directed_insults += 1

                insult_range -= 1
                address_range -= 1
                weak_insult_range -= 1

                if is_insult:
                    insult_range = 3
                elif is_address:
                    address_range = 3
                elif is_weak_insult:
                    weak_insult_range = 2

                if token in self.mostly_normal_dict:
                    positive_weight += self.mostly_normal_dict[token]
                if token in self.mostly_insult_dict:
                    negative_weight += self.mostly_insult_dict[token]

                token_count += 1

            if len(text) == 0:
                insults_ratio = 0
                address_ratio = 0
                directed_insult_ratio = 0
                adj_ratio = 0
            else:
                insults_ratio = total_insults / len(text)
                address_ratio = total_addreses / len(text)
                directed_insult_ratio = directed_insults / len(text)
                adj_ratio = total_adj / len(text)

            if directed_insults > 2:
                directed_insults = 1
            else:
                directed_insults /= 2

            # this_features.append(directed_insults)
            # this_features.append(adj_ratio)
            this_features.append(directed_insult_ratio)
            this_features.append(len(text))
            # this_features.append(total_insults)
            this_features.append(insults_ratio)
            this_features.append(positive_weight - negative_weight)
            this_features.append(positive_weight)
            this_features.append(negative_weight)
            # this_features.append(total_addreses)

            features.append(this_features)

        return features

    def fit(self, texts, y=None):
        total_insults = 0
        total_normal = 0

        for i in range(len(y)):
            if y[i]:
                total_insults += 1
                for token in texts[i]:
                    self.insult_dict.setdefault(token[0], 0)
                    self.insult_dict[token[0]] += 1
            else:
                total_normal += 1
                for token in texts[i]:
                    self.normal_dict.setdefault(token[0], 0)
                    self.normal_dict[token[0]] += 1

        for key, value in list(self.insult_dict.items()):
            if value > 0:
                self.insult_dict[key] /= total_insults
            else:
                self.insult_dict.pop(key, None)
        for key, value in list(self.normal_dict.items()):
            if value > 1:
                self.normal_dict[key] /= total_normal
            else:
                self.normal_dict.pop(key, None)

        max_value = 20
        min_threshold = 5

        for key in self.insult_dict:
            if key in self.normal_dict and self.insult_dict[key] / self.normal_dict[key] > min_threshold:
                self.mostly_insult_dict[key] = max(self.insult_dict[key] / self.normal_dict[key], max_value)
            elif key not in self.normal_dict:
                self.mostly_insult_dict[key] = max(self.insult_dict[key] * total_insults, max_value)

        for key in self.normal_dict:
            if key in self.insult_dict and self.normal_dict[key] / self.insult_dict[key] > min_threshold:
                self.mostly_normal_dict[key] = max(self.normal_dict[key] / self.insult_dict[key], max_value)
            elif key not in self.insult_dict:
                self.mostly_normal_dict[key] = max(self.normal_dict[key] * total_normal, max_value)

        return self


class InsultDetector:
    def __init__(self):
        with open('insult_words.txt', mode='r', encoding='utf-8') as file:
            insult_words = file.read().splitlines()
        with open('address_words.txt', mode='r', encoding='utf-8') as file:
            address_words = file.read().splitlines()
        with open('weak_insults.txt', mode='r', encoding='utf-8') as file:
            weak_insult_words = file.read().splitlines()
        with open('stop_words.txt', mode='r', encoding='utf-8') as f:
            self.stop_words = f.read().splitlines()

        self.text_clf = None
        self.insult_words_regex = self.create_regex(insult_words)
        self.address_words_regex = self.create_regex(address_words)
        self.weak_insult_words_regex = self.create_regex(weak_insult_words)

    def _json_to_dataset(self, json_data):
        dataset = dict(data=1, target=2)
        dataset['data'] = []
        dataset['target'] = []

        def _iterate(json_data):
            if 'text' in json_data and 'insult' in json_data and json_data['text']:
                dataset['data'].append(json_data['text'])
                dataset['target'].append(json_data['insult'])
            if 'children' in json_data:
                for child in json_data['children']:
                    _iterate(child)

        for root in json_data:
            _iterate(root["root"])

        return dataset

    @staticmethod
    def create_regex(expression_list):
        regex_str = '^('
        for exp in expression_list:
            regex_str += exp + '|'
        regex_str = regex_str[:-1] + ')$'
        regex = re.compile(regex_str)
        return regex

    @staticmethod
    def _reduce_dataset(dataset):
        set_length = len(dataset['data'])
        num_insults = 0

        reduced_dataset = dict(data=1, target=2)
        reduced_dataset['data'] = []
        reduced_dataset['target'] = []

        for i in range(set_length):
            if dataset['target'][i]:
                num_insults += 1
                reduced_dataset['data'].append(dataset['data'][i])
                reduced_dataset['target'].append(True)
        step = set_length / num_insults / 4

        for i in range(0, set_length,  round(step)):
            if not dataset['target'][i]:
                reduced_dataset['data'].append(dataset['data'][i])
                reduced_dataset['target'].append(False)

        print(len(reduced_dataset['data']), num_insults)
        return reduced_dataset

    def train(self, labeled_discussions):
        if type(labeled_discussions) is list:  # for cross validation
            dataset = self._json_to_dataset(labeled_discussions)
        else:
            dataset = labeled_discussions
        # dataset = self._reduce_dataset(dataset)

        text_clf = Pipeline([
            ('insults', InsultFeatures(self.insult_words_regex,
                                       self.address_words_regex,
                                       self.weak_insult_words_regex)),
            ('scaler', StandardScaler()),
            ('clf', SVC(verbose=True, class_weight='auto', C=100, max_iter=100000))
            # ('clf', LinearSVC(verbose=True, class_weight='auto', C=0.04))
        ])

        self.text_clf = text_clf.fit(dataset['data'], dataset['target'])

    def classify(self, unlabeled_discussions):
        def _iterate(discussion):
            if 'text' in discussion and not discussion['text']:
                discussion['insult'] = False
            elif 'text' in discussion:
                discussion['insult'] = self.text_clf.predict([discussion['text']])[0]
            if 'children' in discussion:
                for child in discussion['children']:
                    _iterate(child)

        if type(unlabeled_discussions[0]) is dict:  # for easier cross validation
            for root in unlabeled_discussions:
                _iterate(root["root"])
        else:
            return self.text_clf.predict(unlabeled_discussions)
        return unlabeled_discussions

    def _get_parsed_data(self, data):
        parsed_dataset = []
        analyzer = pymorphy2.MorphAnalyzer()
        analyzer.parse("абв")
        with open('dd', 'rb') as input:
            parsed_dataset = pickle.load(input)
        return parsed_dataset

        total = len(data)
        cnt = 0

        for text in data:
            parsed_text = []
            tokens = my_tokenizer(text, stop_words=self.stop_words)
            for token in tokens:
                parsed_token = analyzer.parse(token)[0]
                parsed_text.append((parsed_token.normal_form, parsed_token.tag))
            parsed_dataset.append(parsed_text)

            cnt += 1
            if cnt % 1000 == 0:
                print('%r out of %r done (%.1f%%)' % (cnt, total, cnt / total * 100.0))

        return parsed_dataset

    def test(self):
        json_file = open('discussions.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json', encoding='utf-8', errors='replace')

        json_data = json.load(json_file)
        dataset = self._json_to_dataset(json_data)

        # a = self._get_parsed_data(dataset['data'])

    def _test_split(self):
        start_time = time.time()
        json_file = open('discussions.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json', encoding='utf-8', errors='replace')
        json_data = json.load(json_file)

        dataset = self._json_to_dataset(json_data)
        dataset['data'] = self._get_parsed_data(dataset['data'])

        dataset['data'], data_test, dataset['target'], target_test \
            = cross_validation.train_test_split(dataset['data'], dataset['target'], test_size=0.2, random_state=1)

        self.train(dataset)
        print('Training done')
        print(f1_score(target_test, self.text_clf.predict(data_test), pos_label=True))
        print("--- %.1f ---" % ((time.time() - start_time) / 60))

    def _test_if_i_broke_something(self):
        json_file = open('test_discussions/learn.json')
        json_data = json.load(json_file)
        self.train(json_data)
        json_test = open('test_discussions/test.json')
        json_test_data = json.load(json_test)
        print(self.classify(json_test_data))

if __name__ == '__main__':
    d = InsultDetector()
    # d.test()
    d._test_split()
#     d._test_if_i_broke_something()
#     m = pymorphy2.MorphAnalyzer()
#     print(m.parse("аффтара"))
