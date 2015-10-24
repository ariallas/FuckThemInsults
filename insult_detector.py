

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

        data = ["abc abc abc bcd", "abc", "bcd", "qwe", "abc"]
        target = [True, True, False, False, True]
        # target = [1, 1, 0, 0, 1]
        test_data = ["abc", "bcd", "oop", "abc bcd"]
        test_anwser = [True, False, False, True]

        text_clf = Pipeline([('vect',  CountVectorizer()),
                             ('tfidf', TfidfTransformer()),
                             ('clf',   SGDClassifier())])
        text_clf = text_clf.fit(data, target)

        predicted = text_clf.predict(test_data)
        
        print(predicted)
        print(numpy.mean(predicted == test_anwser))

    def classify(self, unlabeled_discussions):
        """
        This method take the list of discussions as input. You should predict for every message in every
        discussion (except root) if the message insult or not. Than you should replace the value of field "insult"
        for True if the method is insult and False otherwise.
        :param discussion: list of discussion. The field insult would be replaced by False.
        :return: None
        """
        # TODO put your code here
        return unlabeled_discussions

d = InsultDetector()

#data_string = open('dis.json', 'rb').read().decode('utf-8')
#print(data_string.encode('utf-8'))
#data = json.loads(data_string)

#print(data[0]["root"]["text"].encode('utf-8'))
d.train(1)