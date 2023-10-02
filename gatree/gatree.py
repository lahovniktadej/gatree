import numpy as np


class GATree():
    def __init__(self, MAX_DEPTH=None, random=None):
        """Genetic Algorithm Tree Classifier

        :param MAX_DEPTH: maximum depth of the tree
        :param random: random number generator
        """
        self.MAX_DEPTH = MAX_DEPTH
        self.random = random if random is not None else np.random

    def fit(self, X, y):
        """Fit a tree to a training set

        :param X: training data
        :param y: target values
        """
        ATT_INDEX = np.arange(X.shape[1])
        ATT_VALUES = {i: [(min_val + max_val) / 2 for min_val, max_val in zip(sorted(
            X.iloc[:, i].unique())[:-1], sorted(X.iloc[:, i].unique())[1:])] for i in range(X.shape[1])}
        ATT_VALUES[-1] = sorted(y.unique())
        self.CLASS_COUNT = len(ATT_VALUES[-1])

    def predict(self, X):
        pass
