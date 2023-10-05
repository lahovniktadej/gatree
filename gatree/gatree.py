import pandas as pd
import numpy as np
from tree.node import Node


class GATree():
    def __init__(self, max_depth=None, random=None):
        """Genetic Algorithm Tree Classifier

        :param max_depth: maximum depth of the tree
        :param random: random number generator
        """
        self.max_depth = max_depth
        self.random = random if random is not None else np.random

    def fit(self, X, y):
        """Fit a tree to a training set

        :param X: training data
        :param y: target values
        """
        self.X = X
        self.y = y
        self.att_indexes = np.arange(X.shape[1])
        self.att_values = {i: [(min_val + max_val) / 2 for min_val, max_val in zip(sorted(
            X.iloc[:, i].unique())[:-1], sorted(X.iloc[:, i].unique())[1:])] for i in range(X.shape[1])}
        self.att_values[-1] = sorted(y.unique())
        self.class_count = len(self.att_values[-1])

        node = Node()
        tree = node.make_node(max_depth=self.max_depth, random=self.random,
                              att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count)
        return tree

    def predict(self, X):
        pass


if __name__ == '__main__':
    gatree = GATree(max_depth=8)

    df = pd.DataFrame({'Višina': [160, 180, 170, 180, 160, 190], 'Teža': [
                      60, 50, 70, 50, 50, 120], 'BMI': ['normalen', 'nizek', 'normalen', 'nizek', 'nizek', 'visok']})
    X = df[['Višina', 'Teža']]
    y = df['BMI']
    tree = gatree.fit(X, y)
