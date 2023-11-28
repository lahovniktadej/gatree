import unittest
import numpy as np
from gatree.tree.node import Node
from gatree.ga.mutation import Mutation


class TestMutation(unittest.TestCase):
    """
    Test Mutation class.

    Args:
        unittest.TestCase: TestCase class from the unittest module.
    """

    def test_change_class(self):
        """
        Test change_class method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        node = Node(att_index=-1, att_value=0)

        # Test the change_class method
        Mutation.change_class(node, 2, random)

        # Assert that the node is a leaf
        self.assertEqual(node.att_index, -1)

        # Assert that class has changed
        self.assertEqual(node.att_value, 1)

    def test_exchange_class_for_tree(self):
        """
        Test exchange_class_for_tree method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        node = Node(att_index=-1, att_value=0)
        parent = Node(att_index=0, att_value=0)
        parent.set_left(node)

        # Test the change_class method
        Mutation.exchange_class_for_tree(
            node, [0, 1], {0: [1, 2], 1: [3, 4]}, 2, random)
        node = parent.left

        # Assert that the node is not a leaf
        self.assertNotEqual(node.att_index, -1)

    def test_change_attribute(self):
        """
        Test change_attribute method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        node = Node(att_index=0, att_value=0)

        # Test the change_attribute method
        Mutation.change_attribute(node, [0, 1], {0: [0, 1], 1: [3]}, random)

        # Assert that the attribute has changed
        self.assertEqual(node.att_index, 1)

    def test_change_attribute_value(self):
        """
        Test change_attribute_value method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        node = Node(att_index=0, att_value=0)

        # Test the change_attribute method
        Mutation.change_attribute_value(node, {0: [0, 1]}, random)

        # Assert that the attribute has not changed
        self.assertEqual(node.att_index, 0)

        # Assert that the attribute value has changed
        self.assertEqual(node.att_value, 1)

    def test_exchange_tree_for_class(self):
        """
        Test exchange_tree_for_class method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        tree = Node().make_node(max_depth=1, random=random,
                                att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)

        # Test the exchange_tree_for_class method
        Mutation.exchange_tree_for_class(tree.left, 2, random)

        # Assert that the attribute has changed
        self.assertEqual(tree.left.att_index, -1)

    def test_exchange_tree_for_tree(self):
        """
        Test exchange_tree_for_tree method.
        """
        # Set the seed for reproducibility
        random = np.random.RandomState(0)

        # Create a node for testing
        tree = Node().make_node(max_depth=1, random=random,
                                att_indexes=np.arange(1), att_values={0: [0, 1]}, class_count=2)

        # Test the exchange_tree_for_class method
        Mutation.exchange_tree_for_tree(
            tree.left, np.arange(1), {0: [0, 1]}, 2, random)

        # Assert that the size of sub-tree has changed
        self.assertNotEqual(tree.left.size(), 1)
