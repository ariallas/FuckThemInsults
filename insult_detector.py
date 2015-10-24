

__author__ = 'tpc 2015'

import nltk
import json
import pickle
import numpy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

class InsultDetector:
    def __init__(self):
        """
        it is constructor. Place the initialization here. Do not place train of the model here.
        :return: None
        """
        pass

    def train(self, labeled_discussions):
        """
        This method train the model.
        :param discussions: the list of discussions. See description of the discussion in the manual.
        :return: None
        """

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

        # data = ["abc abc abc bcd", "abc", "bcd", "qwe", "abc"]
        # target = [True, True, False, False, True]
        # target = [1, 1, 0, 0, 1]

        dataset = _json_to_dataset(self, labeled_discussions)
        text_clf = Pipeline([('vect',  CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf',   SGDClassifier())])
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
                print(discussion['insult'])
            if 'children' in discussion:
                for child in discussion['children']:
                    _iterate(child)

        # test_data = ["abc", "bcd", "oop", "abc bcd"]
        # test_anwser = [True, False, False, True]

        # predicted = self.text_clf.predict(test_data)
        
        # print(predicted)
        # print(numpy.mean(predicted == test_anwser))

        for root in unlabeled_discussions:
            _iterate(root["root"])
        return unlabeled_discussions

    def test(self):
        json_file = open('test_discussions/learn.json')
        json_data = json.load(json_file)
        d.train(json_data)

        json_test = open('test_discussions/test.json')
        json_test_data = json.load(json_test)
        print(d.classify(json_test_data))

d = InsultDetector()
d.test()

#data_string = open('dis.json', 'rb')
#print(data_string.encode('utf-8'))
#data = json.loads(data_string)

#print(data[0]["root"]["text"].encode('utf-8'))