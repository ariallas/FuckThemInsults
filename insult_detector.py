# -*- coding: utf-8 -*-
__author__ = 'tpc 2015'

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
# import nltk
# import pymorphy2
import time

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.base import TransformerMixin
from sklearn.metrics import f1_score
from sklearn.svm.classes import SVC

import random

word_regexp = re.compile(u"(?u)\w+|:\)+|;\)+|:\-\)+|;\-\)+|=\)+|\(\(+|\)\)+|!+|\?|\++[0-9]+")

# rustem = nltk.stem.snowball.RussianStemmer()
# rustem = pymorphy2.MorphAnalyzer()

def my_tokenizer(str):
    tokens = word_regexp.findall(str.lower())
    filtered_tokens = []
    for token in tokens:
        ch = token[0]
        if ch == ':' or ch == ';' or ch == '=' or ch == '%':
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
    def __init__(self, bad_words_part, bad_words_begin, address_words):
        self.bad_words_part = bad_words_part
        self.bad_words_begin = bad_words_begin
        self.address_words = address_words

    def _count_addresses(self, tokens):
        cnt = 0
        for token in tokens:
            if token in self.address_words:
                cnt += 1
        return cnt

    def transform(self, texts):
        features = []

        for text in texts:
            this_features = []
            tokens = my_tokenizer(text)

            ads = self._count_addresses(tokens)
            if len(tokens) > 0:
                this_features.append(ads)
            else:
                this_features.append(0)
            was_insult = 0

            is_preious_insult = False
            is_previous_address = False
            near_ins = 0
            for token in my_tokenizer(text):
                is_address = False
                is_insult = False
                if token:
                    is_address = True
                for ins in self.bad_words_part:
                    if ins in token:
                        is_insult = True
                        was_insult = 1
                        break
                for ins in self.bad_words_begin:
                    if token.startswith(ins):
                        is_insult = True
                        break
                if is_preious_insult and is_insult or is_previous_address and is_insult:
                    near_ins += 1
                is_preious_insult = is_insult
                is_previous_address = is_address
            if near_ins > 0:
                near_ins += 5
            # this_features.append(near_ins)
            if was_insult == 0:
                print(text)

            features.append(this_features)
        return sparse.csr_matrix(features)

    def fit(self, texts, y=None):
        return self

    def get_params(self, deep=True):
        return {}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            pass
        return self


class InsultDetector:
    def __init__(self):
        with open('insult_words_part.txt', mode='r', encoding='utf-8') as f:
            self.bad_words_part = f.read().splitlines()
        with open('insult_words_begin.txt', mode='r', encoding='utf-8') as f:
            self.bad_words_begin = f.read().splitlines()
        with open('stop_words.txt', mode='r', encoding='utf-8') as f:
            self.stop_words = f.read().splitlines()
        with open('address_words.txt', mode='r', encoding='utf-8') as f:
            self.address_words = f.read().splitlines()
        self.text_clf = None

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
        dataset = self._reduce_dataset(dataset)

        # text_clf = Pipeline([
        #     ('vect', FeatureUnion([
        #         ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
        #                                   tokenizer=my_tokenizer)),
        #             ('addreses', AddressTransformet(var=1))
        #         ])),
        #     ('clf', SVC(#class_weight='auto',
        #                 C=0.1,
        #                 kernel='linear',
        #                 verbose=True))
        # ])

        text_clf = Pipeline([
            ('vect', FeatureUnion(
                transformer_list=[
                    ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
                                              tokenizer=my_tokenizer,
                                              stop_words=self.stop_words)),
                    ('addreses', InsultFeatures(bad_words_part=self.bad_words_part,
                                                bad_words_begin=self.bad_words_begin))
                ],
                transformer_weights={
                    'tfidf': 0.5,
                    'addreses': 1.0
                })),
            # ('clf',   SGDClassifier(class_weight='auto',
            #                         n_jobs=-1,
            #                         penalty='l2',
            #                         alpha=5e-7,
            #                         loss='hinge',
            #                         n_iter=10))
            ('clf', SVC(C=1,
                        kernel='linear',
                        verbose=True))
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

        if type(unlabeled_discussions[0]) is dict: # for easier cross validation
            for root in unlabeled_discussions:
                _iterate(root["root"])
        else:
            return self.text_clf.predict(unlabeled_discussions)
        return unlabeled_discussions

    def _grid_search(self, json_data):
        dataset = self._json_to_dataset(json_data)
        # text_clf = Pipeline([('vect',  TfidfVectorizer()),
        #                      ('clf',   SGDClassifier(class_weight='auto', n_jobs=-1))])
        # text_clf = Pipeline([('vect',  TfidfVectorizer(max_df=0.75, ngram_range=(1, 2))),
        #                      ('clf',   SGDClassifier(class_weight='auto',
        #                                              n_jobs=-1,
        #                                              loss='squared_hinge',
        #                                              n_iter=5))])
        # text_clf = Pipeline([
        #     ('vect', FeatureUnion([
        #         ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
        #                                   tokenizer=my_tokenizer)),
        #         ('addreses', AddressTransformet())
        #     ])),
        #     ('clf',   SGDClassifier(class_weight='auto',
        #                             n_jobs=-1
        #                             ))
        # ])
        # dataset = self._reduce_dataset(dataset)
        text_clf = Pipeline([
            ('vect', FeatureUnion([
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
                                          tokenizer=my_tokenizer)),
                ('addreses', AddressTransformet(var=1))
            ])),
            ('clf', SVC())
        ])
        parameters = {'clf__C': (0.5, 1, 5),
                      'clf__kernel': ('linear', 'poly'),
                      'clf__class_weight': ('auto', None)
                      # 'clf__loss': ('hinge', 'squared_hinge', 'squared_loss'),
                      # 'clf__penalty': ('l2', 'elasticnet', 'l1'),
                      # 'vect__tfidf__max_df': (0.75, 0.9, 1.0),
                      # 'vect__tfidf__use_idf': (True, False),
                      # 'vect__addreses__var': (0, 1)
                      }
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, scoring='f1', verbose=5)
        gs_clf = gs_clf.fit(dataset['data'], dataset['target'])
        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %r" % (param_name, best_parameters[param_name]))
        print(score)

    def _cross_validate(self, json_data):
        dataset = self._json_to_dataset(json_data)
        # dataset = self._reduce_dataset(dataset)

        # text_clf = Pipeline([
        #     ('vect', FeatureUnion([
        #         ('tfidf', TfidfVectorizer(max_df=1.0,
        #                                   ngram_range=(1, 2),
        #                                   tokenizer=my_tokenizer)),
        #         ('addreses', AddressTransformet(var=1))
        #     ])),
        #     ('clf',   SGDClassifier(class_weight='auto',
        #                             n_jobs=-1,
        #                             penalty='elasticnet',
        #                             alpha=1e-6,
        #                             loss='hinge',
        #                             n_iter=10))
        #     # ('clf', SVC(kernel='linear'))
        # ])
        text_clf = Pipeline([
            ('vect', FeatureUnion(
                [
                ('tfidf', TfidfVectorizer(ngram_range=(1, 2),
                                          tokenizer=my_tokenizer,
                                          stop_words=self.stop_words)),
                ('inaults', InsultFeatures(bad_words_part=self.bad_words_part,
                                           bad_words_begin=self.bad_words_begin))
                ]
            )),
            ('clf',   SGDClassifier(class_weight='auto',
                                    n_jobs=-1,
                                    penalty='elasticnet',
                                    alpha=9e-7,
                                    loss='hinge',
                                    n_iter=10))
        ])
        score = cross_validation.cross_val_score(text_clf,
                                                 dataset['data'],
                                                 dataset['target'],
                                                 cv=5,
                                                 scoring='f1',
                                                 n_jobs=-1,
                                                 verbose=5)
        print(score)

    def test_tokenizer(self, json_data):
        dataset = self._json_to_dataset(json_data)

        tok = TfidfVectorizer().build_tokenizer()
        for text in dataset['data'][:20]:
            try:
                print(text)
                print(my_tokenizer(text))
            except:
                pass
        exit()

    def plot_some_graphs(self, json_data):
        dataset = self._json_to_dataset(json_data)
        at = InsultFeatures(self.bad_words_part, self.bad_words_begin, self.address_words)

        ins = []
        not_ins = []
        for i in range(len(dataset['target'])):
            if dataset['target'][i]:
                ins.append(dataset['data'][i])
            else:
                not_ins.append(dataset['data'][i])

        ins_array = at.transform(ins).toarray()
        # not_ins_array = at.transform(not_ins).toarray()

        rand_arr_ins = [random.random() * 10. for i in range(len(ins_array))]
        rand_arr_not_ins = [random.random() * 10. for i in range(len(not_ins_array))]

        plt.plot(ins_array[:, 0], rand_arr_ins[:], 'r.')
        plt.plot(not_ins_array, rand_arr_not_ins, 'b.')

        plt.show()

    def test(self):
        json_file = open('discussions.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json', encoding='utf-8', errors='replace')

        json_data = json.load(json_file)

        # self._cross_validate(json_data)
        # self._grid_search(json_data)
        # self.test_tokenizer(json_data)
        # self.train(json_data)
        self.plot_some_graphs(json_data)

        # fun = FeatureUnion([
        #     ('tfidf', TfidfVectorizer(ngram_range=(1, 2))),
        #     ('strlen', StrLengthTransformer())
        # ])
        # st = StrLengthTransformer()
        # vt = CountVectorizer()

        # dataset = self._json_to_dataset(json_data)
        # dataset = self._reduce_dataset(dataset)
        # res = fun.fit_transform(dataset['data'])
        # print(res)
        # print(st.transform(dataset['data']))
        # print(vt.fit_transform(dataset['data']))

        # print(res)
        # print(fun.vocabulary_)

        # dataset = self._json_to_dataset(json_data)
        # at = AddressTransformet()
        # ins = []
        # for i in range(len(dataset['data'])):
        #     if (dataset['target'][i]):
        #         ins.append(dataset['data'][i])
        # d = dict()
        # for i in dataset['data']:
        #     for tok in my_tokenizer(i):
        #         for w in bad_words_part:
        #             if w in tok:
        #                 d.setdefault(tok, 0)
        #                 d[tok] += 1
        #         for w in bad_words_begin:
        #             if tok.startswith(w):
        #                 d.setdefault(tok, 0)
        #                 d[tok] += 1
        #
        # import operator
        # for i in sorted(d.items(), key=operator.itemgetter(1)):
        #     if (i[1] > 10):
        #         print(i)

        # print(at.transform(ins))
    def _test_split(self):
        start_time = time.time()
        json_file = open('discussions.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json', encoding='utf-8', errors='replace')
        json_data = json.load(json_file)

        dataset = self._json_to_dataset(json_data)
        dataset['data'], data_test, dataset['target'], target_test \
            = cross_validation.train_test_split(dataset['data'], dataset['target'], test_size=0.2, random_state=1)

        self.train(dataset)
        print('Training done')
        print(f1_score(target_test, self.text_clf.predict(data_test), pos_label=True))
        print("--- %.1f ---" % ((time.time() - start_time) / 60))

    def _test_if_i_broke_something(self):
        # json_file = open('discussions.json', encoding='utf-8', errors='replace')
        json_file = open('test_discussions/learn.json')
        json_data = json.load(json_file)
        self.train(json_data)
        json_test = open('test_discussions/test.json')
        json_test_data = json.load(json_test)
        print(self.classify(json_test_data))

if __name__ == '__main__':
    d = InsultDetector()
    d.test()
    # d._test_split()
#     d._test_if_i_broke_something()
