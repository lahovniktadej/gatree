import unittest
import numpy as np
import pandas as pd
from gatree.tree.node import Node


class TestNode(unittest.TestCase):
    """
    Test Node class.

    Args:
        unittest.TestCase: TestCase class from the unittest module.
    """

    def setUp(self):
        """
        Initialize the test case.
        """
        # Create an instance of the Node class for testing
        self.node = Node(att_index=0, att_value=5)
        self.random = np.random.RandomState(0)

    def test_copy(self):
        """
        Test the copy method.
        """
        # Test the copy method
        copied_node = Node.copy(self.node)

        # Assert the copied node is a different object
        self.assertNotEqual(id(self.node), id(copied_node))

    def test_set_left(self):
        """
        Test the set_left method.
        """
        # Create a left node
        left_node = Node(att_index=-1, att_value=0)
        self.node.set_left(left_node)

        # Assert the left child is set correctly
        self.assertEqual(self.node.left, left_node)

        # Assert the parent of the left node is set correctly
        self.assertEqual(left_node.parent, self.node)

    def test_set_right(self):
        """
        Test the set_right method.
        """
        # Create a right node
        right_node = Node(att_index=-1, att_value=0)
        self.node.set_right(right_node)

        # Assert the right child is set correctly
        self.assertEqual(self.node.right, right_node)

        # Assert the parent of the right node is set correctly
        self.assertEqual(right_node.parent, self.node)

    def test_get_root(self):
        """
        Test the get_root method.
        """
        right_node = Node(att_index=1, att_value=3)
        self.node.set_right(right_node)
        root = self.node.right.get_root()

        # Assert the root node is returned correctly
        self.assertEqual(root, self.node)

    def test_get_children(self):
        """
        Test the get_children method.
        """
        # Create left and right nodes
        self.node.set_left(Node(att_index=-1, att_value=0))
        self.node.set_right(Node(att_index=-1, att_value=1))
        children = self.node.get_children()

        # Assert the children are returned correctly
        self.assertEqual(children, (self.node.left, self.node.right))

    def test_get_leaves(self):
        """
        Test the get_leaves method.
        """
        # Create leaves
        self.node.set_left(Node(att_index=-1, att_value=0))
        self.node.set_right(Node(att_index=-1, att_value=1))
        leaves = self.node.get_leaves()

        # Assert the leaves are returned correctly
        self.assertEqual(leaves, [self.node.left, self.node.right])

    def test_depth(self):
        """
        Test the depth method.
        """
        # Create left and right nodes
        self.node.set_left(Node(att_index=-1, att_value=0))
        self.node.set_right(Node(att_index=-1, att_value=1))

        # Assert the depth is returned correctly
        self.assertEqual(self.node.depth(), 1)
        self.assertEqual(self.node.left.depth(), 2)
        self.assertEqual(self.node.right.depth(), 2)

    def test_max_depth(self):
        """
        Test the max_depth method.
        """
        # Create left and right nodes
        self.node.set_left(Node(att_index=-1, att_value=0))
        self.node.set_right(Node(att_index=-1, att_value=1))

        # Assert the max depth is returned correctly
        self.assertEqual(self.node.max_depth(), 2)

    def test_size(self):
        """
        Test the size method.
        """
        # Create left and right nodes
        self.node.set_left(Node(att_index=-1, att_value=0))
        self.node.set_right(Node(att_index=-1, att_value=1))

        # Assert the size is returned correctly
        self.assertEqual(self.node.size(), 3)

    def test_make_node(self):
        """
        Test the make_node method.
        """
        # Create the arguments for the make_node method
        att_indexes = [0, 1, 2]
        att_values = {0: [1, 2, 3], 1: [4, 5, 6], 2: [7, 8, 9]}
        class_count = 2
        max_depth = 3

        # Generate a node
        node = self.node.make_node(max_depth=max_depth, random=self.random,
                                   att_indexes=att_indexes, att_values=att_values, class_count=class_count)

        # Assert the node is returned correctly
        self.assertIsInstance(node, Node)

    def test_clear_evaluation(self):
        """
        Test the clear_evaluation method.
        """
        # Set the evaluation
        self.node.y_true = [1, 2, 3]
        self.node.y_pred = [1, 2, 3]
        self.node.clear_evaluation()

        # Assert the evaluation is cleared correctly
        self.assertEqual(self.node.y_true, [])
        self.assertEqual(self.node.y_pred, [])

    def test_is_evaluated(self):
        """
        Test the is_evaluated method.
        """
        # Set the evaluation
        self.node.y_true = [1, 2, 3]
        self.node.y_pred = [1, 2, 3]

        # Assert the evaluation is evaluated correctly
        self.assertTrue(self.node.is_evaluated())

    def test_predict_one(self):
        """
        Test the predict_one method.
        """
        # Create leaves
        self.node.set_left(Node(att_index=-1, att_value=0))
        self.node.set_right(Node(att_index=-1, att_value=1))

        # Arguments for the predict_one method
        X = pd.Series([5])

        # Predict the class
        predicted = self.node.predict_one(X)

        # Assert the class is predicted correctly
        self.assertEqual(predicted, 1)
