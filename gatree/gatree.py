import numpy as np
from sklearn.base import BaseEstimator

# GATree
from gatree.tree.node import Node


class GATree(BaseEstimator):
    """
    Evolutionary decision tree classifier. The GATree classifier is a decision tree classifier that is trained using a genetic algorithm. The genetic algorithm is used to evolve a population of trees over multiple generations. The fitness of each tree is evaluated using a fitness function, which is used to select the best trees for crossover and mutation.

    Args:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random, optional): Random number generator.
        fitness_function (function, optional): Fitness function for the genetic algorithm.
        n_jobs (int, optional): Number of jobs to run in parallel.
        random_state (int, optional): Seed for reproducibility.

    Attributes:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random): Random number generator.
        X (pandas.DataFrame): Training data.
        y (pandas.Series): Target values.
        att_indexes (numpy.ndarray): Array of attribute indexes.
        att_values (dict): Dictionary of attribute values.
        class_count (int): Number of classes.
        fitness_function (function): Fitness function for the genetic algorithm.
        n_jobs (int): Number of jobs to run in parallel.
        random_state (int): Seed for reproducibility.
        _tree (Node): The fitted tree.
        _best_fitness (list): List of best fitness values for each iteration.
        _avg_fitness (list): List of average fitness values for each iteration.
    """

    def __init__(self, max_depth=None, random=None, fitness_function=None, n_jobs=1, random_state=None):
        """
        Initialise the Genetic Algorithm Tree Classifier. Maximum depth, random number generator, fitness function, number of jobs, and random state can be specified.

        Args:
            max_depth (int, optional): Maximum depth of the tree.
            random (Random, optional): Random number generator.
            fitness_function (function, optional): Fitness function for the genetic algorithm.
            random_state (int, optional): Seed reproducibility.
        """
        self.max_depth = max_depth
        if random is None and random_state is not None:
            np.random.seed(random_state)
        self.random = random if random is not None else np.random
        self.fitness_function = fitness_function if fitness_function is not None else self.default_fitness_function
        self.n_jobs = n_jobs
        self._tree = None
        self._best_fitness = []
        self._avg_fitness = []

    @staticmethod
    def default_fitness_function(root):
        """ 
        Default fitness function for the genetic algorithm.

        Args:
            root (Node): Root node of the tree.

        Returns:
            float: The fitness value.
        """
        pass

    @staticmethod
    def _predict_and_evaluate(tree, X, y, fitness_function, is_training=False):
        """
        Evaluate a tree on a training set (in parallel).

        Args:
            tree (Node): Tree to evaluate.
            X (pandas.DataFrame): Training data.
            y (pandas.Series): Target values.
            fitness_function (function): Fitness function for the genetic algorithm.
            is_training (bool): If the instances are used for training or predicting.

        Returns:
            Node: The evaluated tree.
        """
        for j in range(X.shape[0]):
            # Predict class for current instance
            tree.predict_one(X.iloc[j], y.iloc[j], is_training)
        tree.fitness = fitness_function(tree)
        return tree

    def fit(self, X, y, population_size=150, max_iter=2000, mutation_probability=0.1, elite_size=1, selection_tournament_size=2):
        """
        Fit a tree to a training set. The population size, maximum iterations, mutation probability, elite size, and selection tournament size can be specified.

        Args:
            X (pandas.DataFrame): Training data.
            y (pandas.Series): Target values.
            population_size (int, optional): Size of the population.
            max_iter (int, optional): Maximum number of iterations.
            mutation_probability (float, optional): Probability of mutation.
            elite_size (int, optional): Number of elite trees.
            selection_tournament_size (int, optional): Number of trees in tournament.

        Returns:
            Node: The fitted tree.
        """
        pass

    def predict(self, X):
        """
        Predict classes for the given data.

        Args:
            X (pandas.DataFrame): Data to predict.

        Returns:
            list: Predicted classes.
        """
        y_pred = []
        for i in range(X.shape[0]):
            index = self._tree.predict_one(X.iloc[i])
            y_pred.append(self.att_values[-1][index])
        return y_pred

    def plot(self, node=None, prefix=''):
        """
        Plot the decision tree with nodes and leaves.

        Args:
            node (Node, optional): Current node to plot.
            prefix (str, optional): Prefix for the current node.
        """
        if node is None:
            node = self._tree

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
