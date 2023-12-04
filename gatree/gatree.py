import time
from multiprocessing.pool import Pool
import numpy as np
from gatree.tree.node import Node
from gatree.ga.selection import Selection
from gatree.ga.crossover import Crossover
from gatree.ga.mutation import Mutation
from sklearn.metrics import accuracy_score


class GATree():
    """
    Genetic Algorithm Tree Classifier.

    Args:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random, optional): Random number generator.
        fitness_function (function, optional): Fitness function for the genetic algorithm.
        _tree (Node, optional): The fitted tree.
        _best_fitness (list, optional): List of best fitness values for each iteration.
        _avg_fitness (list, optional): List of average fitness values for each iteration.

    Attributes:
        max_depth (int, optional): Maximum depth of the tree.
        random (Random): Random number generator.
        X (pandas.DataFrame): Training data.
        y (pandas.Series): Target values.
        att_indexes (numpy.ndarray): Array of attribute indexes.
        att_values (dict): Dictionary of attribute values.
        class_count (int): Number of classes.
        fitness_function (function): Fitness function for the genetic algorithm.
        random_state (int): Seed for reproducibility.
        _tree (Node): The fitted tree.
        _best_fitness (list): List of best fitness values for each iteration.
        _avg_fitness (list): List of average fitness values for each iteration.
    """

    def __init__(self, max_depth=None, random=None, fitness_function=None, random_state=None):
        """
        Initialize the Genetic Algorithm Tree Classifier.

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
        acc = accuracy_score(root.y_true, root.y_pred)
        return (1 - acc + 0.002 * root.size())

    @staticmethod
    def _predict_and_evaluate(tree, X, y, fitness_function):
        for j in range(X.shape[0]):
            tree.predict_one(X.iloc[j], y.iloc[j])
        tree.fitness = fitness_function(tree)
        return tree

    def fit(self, X, y, population_size=150, max_iter=2000, mutation_probability=0.1, elite_size=1, selection_tournament_size=2):
        """
        Fit a tree to a training set.

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
        start = time.time()
        self.X = X
        self.y = y
        self.att_indexes = np.arange(X.shape[1])
        self.att_values = {i: [(min_val + max_val) / 2 for min_val, max_val in zip(sorted(
            X.iloc[:, i].unique())[:-1], sorted(X.iloc[:, i].unique())[1:])] for i in range(X.shape[1])}
        self.att_values[-1] = sorted(y.unique())
        self.class_count = len(self.att_values[-1])

        # Generation of initial population
        node = Node()
        population = []
        for _ in range(population_size):
            population.append(node.make_node(max_depth=self.max_depth, random=self.random,
                              att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count))

        times = {'evaluation': 0, 'selection': 0,
                 'crossover': 0, 'mutation': 0}
        for i in range(max_iter+1):
            start_time = time.time()

            # Evaluation of population
            args = [(tree, X, y, self.fitness_function) for tree in population]
            with Pool(processes=5) as pool:
                population = pool.starmap(GATree._predict_and_evaluate, args)

            """ for tree in population:
                for j in range(X.shape[0]):
                    tree.predict_one(X.iloc[j], y.iloc[j])
                tree.fitness = self.fitness_function(tree) """

            end_time = time.time()
            times['evaluation'] += end_time - start_time

            # Sort population by fitness
            population.sort(key=lambda x: x.fitness, reverse=True)

            # Log best and average fitness
            self._best_fitness.append(population[0].fitness)
            self._avg_fitness.append(
                sum([tree.fitness for tree in population]) / len(population))

            if i != max_iter:
                # Elites
                elites = population[:elite_size]

                # Descendant generation
                descendant = []
                for j in range(0, len(population), 2):
                    start_time = time.time()
                    # Tournament selection
                    tree1, tree2 = Selection.selection(
                        population=population, selection_tournament_size=selection_tournament_size, random=self.random)
                    end_time = time.time()
                    times['selection'] += end_time - start_time

                    start_time = time.time()
                    # Crossover between selected trees
                    crossover1 = Crossover.crossover(
                        tree1=tree1, tree2=tree2, random=self.random)
                    crossover2 = Crossover.crossover(
                        tree1=tree2, tree2=tree1, random=self.random)
                    end_time = time.time()
                    times['crossover'] += end_time - start_time

                    start_time = time.time()
                    # Mutation of new trees
                    mutation1 = crossover1
                    mutation2 = crossover2
                    if self.random.random() < mutation_probability:
                        mutation1 = Mutation.mutation(root=crossover1, att_indexes=self.att_indexes,
                                                      att_values=self.att_values, class_count=self.class_count, random=self.random)
                    if self.random.random() < mutation_probability:
                        mutation2 = Mutation.mutation(root=crossover2, att_indexes=self.att_indexes,
                                                      att_values=self.att_values, class_count=self.class_count, random=self.random)
                    end_time = time.time()
                    times['mutation'] += end_time - start_time

                    # Add new trees to descendant population
                    descendant.extend([mutation1, mutation2])

                # Elites + descendants
                descendant.sort(key=lambda x: x.fitness, reverse=True)
                descendant = elites + descendant[:population_size - elite_size]

                # Replace old population with new population
                population = descendant

        end = time.time()

        print('max_iter: ', max_iter)
        print('population_size: ', population_size)
        for key, value in times.items():
            print('{}: {}'.format(key, value))

        print('execution: ', end - start)
        self._tree = population[0]

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
            y_pred.append(self._tree.predict_one(X.iloc[i]))
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
