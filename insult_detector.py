

__author__ = 'tpc 2015'

import nltk
import json
import numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import cross_validation
from sklearn.metrics import f1_score

class InsultDetector:
    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        pass

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

        # print(json_data)
        for root in json_data:
            _iterate(root["root"])

        return dataset

    def train(self, labeled_discussions):
        """
        This method train the model.
        :param discussions: the list of discussions. See description of the discussion in the manual.
        :return: None
        """
        if (type(labeled_discussions) is list): # for cross validation
            dataset = self._json_to_dataset(labeled_discussions)
        else:
            dataset = labeled_discussions

        # dataset['data'] = ["abc abc abc bcd", "abc", "bcd", "qwe", "abc"]
        # dataset['target'] = [True, True, False, False, True]

        text_clf = Pipeline([('vect',  TfidfVectorizer(stop_words=["вы"], max_df=0.85)),
                             ('clf',   SGDClassifier(class_weight='auto'))])

        self.text_clf = text_clf.fit(dataset['data'], dataset['target'])

    def classify(self, unlabeled_discussions):
        """
        This method take the list of discussions as input. You should predict for every message in every
        discussion (except root) if the message insult or not. Than you should replace the value of field "insult"
        for True if the method is insult and False otherwise.
        :param discussion: list of discussion. The field insult would be replaced by False.
        :return: None
        """

        def _iterate(discussion):
            if 'text' in discussion and not discussion['text']:
                discussion['insult'] = False
            elif 'text' in discussion:
                discussion['insult'] = self.text_clf.predict([discussion['text']])[0]
            if 'children' in discussion:
                for child in discussion['children']:
                    _iterate(child)

        # predicted = self.text_clf.predict(test_data)
        # print(predicted)
        # print(numpy.mean(predicted == test_anwser))

        if type(unlabeled_discussions[0]) is dict: # for easier cross validation
            for root in unlabeled_discussions:
                _iterate(root["root"])
        else:
            return self.text_clf.predict(unlabeled_discussions)
        return unlabeled_discussions

    def test(self):
        json_file = open('discussions.json', encoding='utf-8', errors='replace')
        # json_file = open('test_discussions/learn.json')
        # json_data = json.load(json_file)
        # print(json_data[0]["root"]["text"].encode('cp1251', errors='replace')[:50])
        # d.train(json_data)

        # json_test = open('test_discussions/test.json')
        # json_test_data = json.load(json_test)
        # print(d.classify(json_test_data))

        # json_file = open('discussions1.json', encoding='utf-8-sig')
        json_data = json.load(json_file)
        dataset = self._json_to_dataset(json_data)

        train_dataset = dict(data=1, target=2)
        train_dataset['data'], test_discussions, train_dataset['target'], test_anwsers = \
            cross_validation.train_test_split(dataset['data'], dataset['target'], test_size=0.3, random_state=1)

        d.train(train_dataset)
        predicted = d.classify(test_discussions)
        for i in range(len(predicted)):
            if predicted[i] == True:
                try:
                    print(test_discussions[i])
                except:
                    pass
        print(f1_score(test_anwsers, predicted, pos_label=True))

d = InsultDetector()
d.test()