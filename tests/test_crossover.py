import unittest
import numpy as np
from gatree.tree.node import Node
from gatree.ga.crossover import Crossover


class TestCrossover(unittest.TestCase):
    """
    Test the Crossover class.

    Args:
        unittest.TestCase: TestCase class from the unittest module.
    """

    def test_crossover(self):
        """
        Test the crossover method.
        """
        # Create test trees and a random number generator
        random = np.random
        tree1 = Node().make_node(max_depth=1, random=random,
                                 att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)
        tree2 = Node().make_node(max_depth=1, random=random,
                                 att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)

        # Perform crossover
        result = Crossover.crossover(tree1, tree2, random)

        # Check if result is a Node
        self.assertIsInstance(result, Node)

        # Check if result isn't equal to tree1 or tree2
        self.assertNotEqual(result, tree1)
        self.assertNotEqual(result, tree2)


if __name__ == '__main__':
    unittest.main()
