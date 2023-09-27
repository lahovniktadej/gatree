from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class Node:
    def __init__(self, att_index=None, att_value=None):
        """ Initializes the node with the given parameters

        :param att_index: attribute index or -1 if this is a leaf
        :param att_value: attribute value or None if this is a leaf
        """
        if att_index is None:
            self.__init__(-1, att_index)  # leaf node
        else:
            self.parent = None
            self.left = None
            self.right = None
            self.att_index = att_index  # if -1 then this is leaf
            self.att_value = att_value

            self.fitness = None
            self.matrix = None

    @staticmethod
    def copy(node):
        """ Returns a deep copy of the given node

        :param node: node to be copied
        :return: deep copy of the given node
        """
        copy = Node(node.att_index, node.att_value)
        copy.parent = node.parent
        copy.left = node.left
        copy.right = node.right
        copy.fitness = node.fitness
        copy.matrix = node.matrix
        return copy

    def set_left(self, n):
        """ Sets the left child of this node to the given node

        :param n: node to be set as the left child
        """
        self.left = n
        n.parent = self

    def set_right(self, n):
        """ Sets the right child of this node to the given node

        :param n: node to be set as the right child
        """
        self.right = n
        n.parent = self

    def get_root(self):
        """ Returns the root node of the tree

        :return: root node of the tree
        """
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_children(self):
        """ Returns the children of this node

        :return: list of children of this node
        """
        return self.left, self.right

    def get_leaves(self):
        """ Returns the leaves of this node

        :return: list of leaves of this node
        """
        if self.left is None:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    def depth(self):
        """ Returns the depth of this node

        :return: depth of this node
        """
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

    def max_depth(self):
        """ Returns the maximum depth of the tree

        :return: maximum depth of the tree
        """
        return self.max_depth_helper(self.get_root()) - 1

    @staticmethod
    def max_depth_helper(n):
        """ Helper function for max_depth

        :param n: node to be used as the root
        :return: maximum depth of the tree
        """
        if n is None:
            return 0
        l_depth = Node.max_depth_helper(n.left)
        r_depth = Node.max_depth_helper(n.right)
        return max(l_depth, r_depth) + 1

    def size(self):
        """ Size is the number of all nodes (mid-trees nodes + leafs) in the trees.

        :return: number of all nodes in the trees
        """
        return self.size_helper(self.get_root())

    @staticmethod
    def size_helper(n):
        if n is None:
            return 0
        return Node.size_helper(n.left) + Node.size_helper(n.right) + 1

    def clear_evaluation(self, num_values):
        """ Clears the evaluation matrix of this node and all its children

        :param num_values: number of possible class values
        """
        self.matrix = [[0.0] * num_values for _ in range(num_values)]
        if self.left:
            self.left.clear_evaluation(num_values)
        if self.right:
            self.right.clear_evaluation(num_values)

    def is_evaluated(self):
        """ Returns true if this node and all its children are evaluated

        :return: true if this node and all its children are evaluated
        """
        if self.matrix is None:
            return False

        left_evaluated = self.left.is_evaluated() if self.left else True
        right_evaluated = self.right.is_evaluated() if self.right else True

        return left_evaluated and right_evaluated

    def predict_one(self, X, y):
        """ Predicts the class of the given instance

        :param X: instance to be predicted
        :param y: true class of the instance
        :return: predicted class
        """
        try:
            actual = int(y)
            predicted = int(self.att_value)

            if self.att_index != -1:
                if X[self.att_index] > self.att_value:
                    predicted = self.left.predict_one(X, y)
                    self.matrix[actual][predicted] += 1
                else:
                    predicted = self.right.predict_one(X, y)
                    self.matrix[actual][predicted] += 1
            else:
                self.matrix[actual][predicted] += 1

            return predicted
        except Exception as e:
            print(e)
            return -1

    def __str__(self):
        """ Returns a string representation of this node

        :return: string representation of this node
        """
        if self.att_index == -1:
            # Assuming att_value is a numeric index
            return str(int(self.att_value))
            # For nominal values, you might need a mapping from index to value
        else:
            return f"Attribute {self.att_index} > {self.att_value}"

    def __repr__(self):
        """ Returns a string representation of this node

        :return: string representation of this node"""
        return self.__str__()

    def __eq__(self, other):
        """ Returns true if this node is equal to the given node

        :param other: node to be compared
        :return: true if this node is equal to the given node
        """
        if not isinstance(other, Node):
            return False

        left1 = self.left
        left2 = other.left
        right1 = self.right
        right2 = other.right

        if self.att_index != other.att_index or self.att_value != other.att_value:
            return False

        if (left1 is not None or left2 is not None) and (left1 != left2):
            return False

        if (right1 is not None or right2 is not None) and (right1 != right2):
            return False

        return True

    def __ne__(self, other):
        """ Returns true if this node is not equal to the given node

        :param other: node to be compared
        :return: true if this node is not equal to the given node
        """
        return not self.__eq__(other)


if __name__ == "__main__":
    # Example initialization, replace with your actual initialization
    n = Node(0, 0.0)
    # Example matrix, replace with your actual matrix
    n.matrix = [[2.0, 3.0], [1.0, 5.0]]

    print(n.matrix)

    print("Accuracy:    ", n.get_accuracy())
    print("Weighted acc:", n.get_weighted_accuracy())
    print("Average acc: ", n.get_avg_accuracy())
    print("Min acc:     ", n.get_min_class_accuracy())
    print("Max acc:     ", n.get_max_class_accuracy())
    print("Precision:   ", n.get_precision())
    print("Recall:      ", n.get_recall())
    print("M Fscore:    ", n.get_multiplicated_fscore())
    print("Avg Fscore:  ", n.get_average_fscore())
    print("Weighted Fs: ", n.get_weighted_fscore())
    print()
    print("Class acc:   ", n.get_class_accuracies())
    print("Precisions:  ", n.get_precisions())
    print("Recalls:     ", n.get_recalls())
    print("Fscores:     ", n.get_fscores())

    print("DEPTH: ", n.max_depth())
