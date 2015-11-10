from nltk.chat.rude import rude_chat

__author__ = 'tpc 2015'

import json
import re
import time
import nltk

from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score

word_regexp = re.compile(u"(?u)\w+"
                         u"|:\)+"
                         u"|;\)+"
                         u"|:\-\)+"
                         u"|;\-\)+"
                         u"|\(\(+"
                         u"|\)\)+"
                         u"|!+"
                         u"|\?+"
                         u"|\+[0-9]+"
                         u"|\++")

rustem = nltk.stem.snowball.RussianStemmer()
insult_words_regex = None
address_words_regex = None
weak_insult_words_regex = None


def my_tokenizer(text):
    tokens = word_regexp.findall(text.lower())
    filtered_tokens = []
    for token in tokens:
        ch = token[0]
        if ch == ':' or ch == ';':
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
        else:
            token = rustem.stem(token)
        filtered_tokens.append(token)
    return filtered_tokens

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

        global insult_words_regex, address_words_regex, weak_insult_words_regex
        insult_words_regex = self.insult_words_regex
        address_words_regex = self.address_words_regex
        weak_insult_words_regex = self.weak_insult_words_regex

    @staticmethod
    def _json_to_dataset(json_data):
        dataset = dict(data=1, target=2)
        dataset['data'] = []
        dataset['target'] = []
        cnt = [0, 0]

        def _iterate(json_data):
            if 'text' in json_data and json_data['text']:
                if 'insult' in json_data and json_data['insult']:
                    dataset['data'].append(json_data['text'])
                    dataset['target'].append(True)
                    cnt[0] += 1
                # else:
                elif 'insult' in json_data:
                    dataset['data'].append(json_data['text'])
                    dataset['target'].append(False)
                    cnt[1] += 1
            if 'children' in json_data:
                for child in json_data['children']:
                    _iterate(child)

        for root in json_data:
            _iterate(root["root"])
        print("DONE CONVERTING", ' ', cnt, ' ', cnt[0] + cnt[1])

        return dataset

    @staticmethod
    def create_regex(expression_list):
        regex_str = '^('
        for exp in expression_list:
            regex_str += exp + '|'
        regex_str = regex_str[:-1] + ')$'
        regex = re.compile(regex_str)
        return regex

    def train(self, labeled_discussions):
        if (type(labeled_discussions) is list): # for cross validation
            dataset = self._json_to_dataset(labeled_discussions)
        else:
            dataset = labeled_discussions

        text_clf = Pipeline([('vect',  TfidfVectorizer(ngram_range=(1, 3),
                                                       tokenizer=my_tokenizer)),
                             ('clf',   SGDClassifier(class_weight='auto',
                                                     n_jobs=-1,
                                                     alpha=5e-07,
                                                     penalty='l2',
                                                     loss='hinge',
                                                     n_iter=50))])

        self.text_clf = text_clf.fit(dataset['data'], dataset['target'])

    def classify(self, unlabeled_discussions):
        def _iterate(discussion):
            if 'text' in discussion and not discussion['text'] or 'text' not in discussion:
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

    def _grid_search(self, json_data):
        dataset = self._json_to_dataset(json_data)
        # text_clf = Pipeline([('vect',  TfidfVectorizer()),
        #                      ('clf',   SGDClassifier(class_weight='auto', n_jobs=-1))])
        text_clf = Pipeline([('vect',  TfidfVectorizer(max_df=0.9, ngram_range=(1, 2))),
                             ('clf',   SGDClassifier(class_weight='auto',
                                                     n_jobs=-1,
                                                     loss='squared_hinge',
                                                     n_iter=5))])
        parameters = {
                        'clf__alpha': (678e-8, 679e-8, 680e-8),
                      # 'clf__loss': ('hinge', 'squared_hinge', 'modified_huber', 'squared_loss'),
                      # 'clf__n_iter': (5, 10, 20),
                      # 'clf__penalty': ('l2', 'elasticnet'),
                      # 'vect__max_df': (0.5, 0.75, 1.0),
                      # 'vect__ngram_range': [(1, 1), (1, 2)]
                      }
        gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, scoring='f1')
        gs_clf = gs_clf.fit(dataset['data'], dataset['target'])
        best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
        for param_name in sorted(parameters.keys()):
            print("%s: %.10f" % (param_name, best_parameters[param_name]))
        print(score)

    def _cross_validate(self, json_data):
        dataset = self._json_to_dataset(json_data)

        text_clf = Pipeline([('vect',  TfidfVectorizer(ngram_range=(1, 4),
                                                       tokenizer=my_tokenizer)),
                             ('clf',   SGDClassifier(class_weight='auto',
                                                     n_jobs=-1,
                                                     alpha=1e-06,
                                                     penalty='l2',
                                                     loss='hinge',
                                                     n_iter=50))])

        score = cross_validation.cross_val_score(text_clf,
                                                 dataset['data'],
                                                 dataset['target'],
                                                 cv=4,
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

    def _test_split(self):
        start_time = time.time()
        # json_file = open('discussions.json', encoding='utf-8', errors='replace')
        json_file = open('discussions_new.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json', encoding='utf-8', errors='replace')
        json_data = json.load(json_file)

        dataset = self._json_to_dataset(json_data)
        dataset['data'], data_test, dataset['target'], target_test \
            = cross_validation.train_test_split(dataset['data'], dataset['target'], test_size=0.2, random_state=1)

        self.train(dataset)
        print('Training done')
        print(f1_score(target_test, self.text_clf.predict(data_test), pos_label=True))
        print("--- %.1f ---" % ((time.time() - start_time) / 60))

    def test(self):
        json_file = open('discussions.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json')

        json_data = json.load(json_file)

        self._cross_validate(json_data)
        # self._grid_search(json_data)
        # self.test_tokenizer(json_data)
        # self.train(json_data)

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
    # d.test()
    d._test_split()
    # d._test_if_i_broke_something()
