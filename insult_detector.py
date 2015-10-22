__author__ = 'tpc 2015'


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
        # TODO put your code here
        pass

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

