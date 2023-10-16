import pandas as pd
import numpy as np
from tree.node import Node
from ga.crossover import Crossover
from ga.mutation import Mutation


class GATree():
    def __init__(self, max_depth=None, random=None):
        """ Genetic Algorithm Tree Classifier

        :param max_depth: maximum depth of the tree
        :param random: random number generator
        """
        self.max_depth = max_depth
        self.random = random if random is not None else np.random

    def fit(self, X, y):
        """ Fit a tree to a training set

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
        tree1 = node.make_node(max_depth=self.max_depth, random=self.random,
                               att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count)
        tree2 = node.make_node(max_depth=self.max_depth, random=self.random,
                               att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count)

        tree = Crossover.crossover(
            tree1=tree1, tree2=tree2, random=self.random)

        tree = Mutation.mutation(root=tree, att_indexes=self.att_indexes,
                                 att_values=self.att_values, class_count=self.class_count, random=self.random)

        return tree

    def predict(self, X):
        pass

    def plot(self, node=None, prefix=''):
        """ Plot the decision tree with nodes and leaves

        :param node: current node to plot
        :param prefix: prefix for the current node
        """
        if node is not None:
            if node.att_index != -1:
                print(prefix + '├── {} > {}'.format(self.X.columns.tolist()
                      [node.att_index], node.att_value))
            else:
                print(
                    prefix + '└── Class: {}'.format(self.att_values[-1][node.att_value]))

            if node.left is not None or node.right is not None:
                self.plot(node.left, prefix + '    ')
                self.plot(node.right, prefix + '    ')


if __name__ == '__main__':
    gatree = GATree(max_depth=5)

    df = pd.DataFrame({'Height': [160, 150, 170, 180, 165, 190], 'Weight': [
                      60, 50, 70, 50, 50, 120], 'Age': [15, 13, 17, 23, 18, 22], 'BMI': ['normal', 'low', 'normal', 'low', 'low', 'high']})
    X = df[['Height', 'Weight', 'Age']]
    y = df['BMI']
    tree = gatree.fit(X, y)
    gatree.plot(tree)
