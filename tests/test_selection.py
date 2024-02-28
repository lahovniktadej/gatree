import unittest
import numpy as np
from gatree.tree.node import Node
from gatree.ga.selection import Selection


class TestSelection(unittest.TestCase):
    """
    Test the Selection class.

    Args:
        unittest.TestCase: TestCase class from the unittest module.
    """

    def test_selection(self):
        """
        Test the selection method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a population of evaluated trees for testing
        tree1 = Node().make_node(max_depth=1, random=random,
                                 att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)
        tree1.fitness = 1
        tree2 = Node().make_node(max_depth=1, random=random,
                                 att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)
        tree2.fitness = 2
        tree3 = Node().make_node(max_depth=1, random=random,
                                 att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)
        tree3.fitness = 3
        tree4 = Node().make_node(max_depth=1, random=random,
                                 att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)
        tree4.fitness = 4
        population = [tree1, tree2, tree3, tree4]
        selection_tournament_size = 2

        # Test the selection method
        selected_tree_1, selected_tree_2 = Selection.selection(
            population, selection_tournament_size, random)

        # Assert that the selected trees are instances of the Tree class
        self.assertIsInstance(selected_tree_1, Node)
        self.assertIsInstance(selected_tree_2, Node)

        # Assert that the selected trees are different
        self.assertNotEqual(selected_tree_1, selected_tree_2)

        # Assert that the selected trees are in the population
        self.assertIn(selected_tree_1, population)
        self.assertIn(selected_tree_2, population)

        # Assert that the fitness of the selected trees is correct
        self.assertEqual(selected_tree_1.fitness, 2)
        self.assertEqual(selected_tree_2.fitness, 1)
