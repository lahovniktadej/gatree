import unittest
import pandas as pd
from gatree.methods.gatreeclustering import GATreeClustering
from sklearn import datasets
from sklearn.model_selection import train_test_split


class TestGATreeClustering(unittest.TestCase):
    """
    Test GATreeClustering class.
    """

    def test_fit(self):
        """
        Test fit method.
        """
        # Prepare test data
        iris = datasets.load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        X_train, _, y_train, _ = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        min_clusters = 5

        fitness_function_kwargs = {
            'min_clusters': min_clusters,
            'fitness_X': X_train
        }

        # Create a GATree instance
        gatree = GATreeClustering(min_clusters=min_clusters)

        # Fit the model
        gatree.fit(X=X_train, population_size=10, max_iter=10,
                   mutation_probability=0.25, elite_size=2, selection_tournament_size=2,
                   fitness_function_kwargs=fitness_function_kwargs
                   )

        # Assert that the tree is not None
        self.assertIsNotNone(gatree._tree)

        # Assert that the best fitness is not None
        self.assertIsNotNone(gatree._best_fitness)

    def test_predict(self):
        """
        Test predict method.
        """
        # Prepare test data
        iris = datasets.load_iris()
        X = pd.DataFrame(iris.data, columns=iris.feature_names)
        y = pd.Series(iris.target)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        min_clusters = 5
        max_clusters = 10

        fitness_function_kwargs = {
            'min_clusters': min_clusters,
            'fitness_X': X_train
        }

        # Create a GATree instance
        gatree = GATreeClustering(min_clusters=min_clusters, max_clusters=max_clusters)

        # Fit the model
        gatree.fit(X=X_train, population_size=10, max_iter=10,
                   mutation_probability=0.25, elite_size=2, selection_tournament_size=2,
                   fitness_function_kwargs=fitness_function_kwargs)

        # Predict the test data
        y_pred = gatree.predict(X_test)

        # Assert that the prediction is not an empty list
        self.assertNotEqual(len(y_pred), 0)

        # Assert that the length of the prediction is the same as the length of the test data
        self.assertEqual(len(y_pred), len(y_test))

        # Assert that the prediction is a list of integers
        self.assertIsInstance(y_pred, list)

        self.assertTrue(len(set(gatree._tree.get_root().y_pred)) >= min_clusters)

        self.assertTrue(len(set(gatree._tree.get_root().y_pred)) <= max_clusters)
