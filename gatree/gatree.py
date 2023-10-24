import pandas as pd
import numpy as np
from tree.node import Node
from ga.crossover import Crossover
from ga.mutation import Mutation
from sklearn.metrics import accuracy_score


class GATree():
    """
    Genetic Algorithm Tree Classifier.

    Args:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random, optional): Random number generator.
        fitness_function (function, optional): Fitness function for the genetic algorithm.

    Attributes:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random): Random number generator.
        X (pandas.DataFrame): Training data.
        y (pandas.Series): Target values.
        att_indexes (numpy.ndarray): Array of attribute indexes.
        att_values (dict): Dictionary of attribute values.
        class_count (int): Number of classes.
        fitness_function (function): Fitness function for the genetic algorithm.
    """

    def __init__(self, max_depth=None, random=None, fitness_function=None):
        """
        Initialize the Genetic Algorithm Tree Classifier.

        Args:
            max_depth (int, optional): Maximum depth of the tree.
            random (Random, optional): Random number generator.
            fitness_function (function, optional): Fitness function for the genetic algorithm.
        """
        self.max_depth = max_depth
        self.random = random if random is not None else np.random
        self.fitness_function = fitness_function if fitness_function is not None else self.default_fitness_function

    def default_fitness_function(self, root):
        """ 
        Default fitness function for the genetic algorithm.

        Args:
            root (Node): Root node of the tree.

        Returns:
            float: The fitness value.
        """
        acc = accuracy_score(root.y_true, root.y_pred)
        return (1 - acc + 0.002 * root.size())

    def fit(self, X, y):
        """
        Fit a tree to a training set.

        Args:
            X (pandas.DataFrame): Training data.
            y (pandas.Series): Target values.

        Returns:
            Node: The fitted tree.
        """
        self.X = X
        self.y = y
        self.att_indexes = np.arange(X.shape[1])
        self.att_values = {i: [(min_val + max_val) / 2 for min_val, max_val in zip(sorted(
            X.iloc[:, i].unique())[:-1], sorted(X.iloc[:, i].unique())[1:])] for i in range(X.shape[1])}
        self.att_values[-1] = sorted(y.unique())
        self.class_count = len(self.att_values[-1])

        # Random tree generation
        node = Node()
        tree1 = node.make_node(max_depth=self.max_depth, random=self.random,
                               att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count)
        tree2 = node.make_node(max_depth=self.max_depth, random=self.random,
                               att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count)

        # Crossover between two randomly generated trees
        tree = Crossover.crossover(
            tree1=tree1, tree2=tree2, random=self.random)

        # Mutation of the tree
        tree = Mutation.mutation(root=tree, att_indexes=self.att_indexes,
                                 att_values=self.att_values, class_count=self.class_count, random=self.random)

        # Evaluation of the tree
        for i in range(X.shape[0]):
            tree.predict_one(X.iloc[i], y.iloc[i])
        tree.fitness = self.fitness_function(tree)

        return tree

    def predict(self, X):
        pass

    def plot(self, node=None, prefix=''):
        """
        Plot the decision tree with nodes and leaves.

        Args:
            node (Node, optional): Current node to plot.
            prefix (str, optional): Prefix for the current node.
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
    def fitness_function(root):
        """ 
        Fitness function for the genetic algorithm.

        Returns:
            float: The fitness value.
        """
        acc = accuracy_score(root.y_true, root.y_pred)
        return (acc + root.size())

    gatree = GATree(max_depth=5, fitness_function=fitness_function)

    from sklearn import datasets
    iris = datasets.load_iris()

    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.Series(iris.target)

    tree = gatree.fit(X, y)
    gatree.plot(tree)
