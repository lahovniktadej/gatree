import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.base import ClassifierMixin

# GATree
from gatree.tree.node import Node
from gatree.ga.selection import Selection
from gatree.ga.crossover import Crossover
from gatree.ga.mutation import Mutation
from gatree.gatree import GATree


class GATreeClassifier(ClassifierMixin, GATree):
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
        super().__init__(max_depth, random, fitness_function, n_jobs, random_state)

    @staticmethod
    def default_fitness_function(root, **fitness_function_kwargs):
        """
        Default fitness function for the genetic algorithm.

        Args:
            root (Node): Root node of the tree.

        Returns:
            float: The fitness value.
        """
        return 1 - accuracy_score(root.y_true, root.y_pred) + (0.002 * root.size())

    def fit(self, X, y, population_size=150, max_iter=2000, mutation_probability=0.1, elite_size=1,
            selection_tournament_size=2, fitness_function_kwargs={}):
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
            fitness_function_kwargs (dict, optional): Additional kwargs to be passed to the fitness_funciton.

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

        # Generation of initial population
        node = Node()
        population = []
        for _ in range(population_size):
            population.append(node.make_node(max_depth=self.max_depth, random=self.random,
                              att_indexes=self.att_indexes, att_values=self.att_values, class_count=self.class_count))

        for i in range(max_iter+1):
            # Clear previous evaluation
            for tree in population:
                tree.clear_evaluation()

            # Evaluation of population
            population = Parallel(n_jobs=self.n_jobs)(delayed(GATreeClassifier._predict_and_evaluate)(
                tree, X, y, self.fitness_function, True, **fitness_function_kwargs) for tree in population)

            # Sort population by fitness
            population.sort(key=lambda x: x.fitness, reverse=False)

            # Log best and average fitness
            self._best_fitness.append(population[0].fitness)
            self._avg_fitness.append(
                sum([tree.fitness for tree in population]) / len(population))

            if i != max_iter:
                # Elites
                elites = population[:elite_size]

                # Descendant generation
                descendant = []
                for _ in range(0, len(population), 2):
                    # Tournament selection
                    tree1, tree2 = Selection.selection(
                        population=population, selection_tournament_size=selection_tournament_size, random=self.random)

                    # Crossover between selected trees
                    crossover1 = Crossover.crossover(
                        tree1=tree1, tree2=tree2, random=self.random)
                    crossover2 = Crossover.crossover(
                        tree1=tree2, tree2=tree1, random=self.random)

                    # Mutation of new trees
                    mutation1 = crossover1
                    mutation2 = crossover2
                    if self.random.random() < mutation_probability:
                        mutation1 = Mutation.mutation(root=crossover1, att_indexes=self.att_indexes,
                                                      att_values=self.att_values, class_count=self.class_count,
                                                      random=self.random)
                    if self.random.random() < mutation_probability:
                        mutation2 = Mutation.mutation(root=crossover2, att_indexes=self.att_indexes,
                                                      att_values=self.att_values, class_count=self.class_count,
                                                      random=self.random)

                    # Add new trees to descendant population
                    descendant.extend([mutation1, mutation2])

                # Elites + descendants
                descendant.sort(key=lambda x: x.fitness, reverse=False)
                descendant = elites + descendant[:population_size - elite_size]

                # Replace old population with new population
                population = descendant

        self._tree = population[0]
