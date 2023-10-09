class Node:
    def __init__(self, att_index=None, att_value=None):
        """ Initializes the node with the given parameters

        :param att_index: attribute index or -1 if this is a leaf
        :param att_value: attribute value or None if this is a leaf
        """
        if att_index is None and att_value is not None:
            self.__init__(-1, att_value)  # leaf node
        else:
            self.parent = None
            self.left = None
            self.right = None

            self.att_index = att_index if att_index is not None else None  # if -1 then this is leaf
            self.att_value = att_value if att_value is not None else None

            self.fitness = None
            self.y_true = []
            self.y_pred = []

    @staticmethod
    def copy(node, parent=None):
        """ Returns a deep copy of the given node

        :param node: node to be copied
        :return: deep copy of the given node
        """
        copy = Node(node.att_index, node.att_value)
        copy.parent = parent
        copy.left = Node.copy(
            node.left, parent=node) if node.left is not None else None
        copy.right = Node.copy(
            node.right, parent=node) if node.right is not None else None
        copy.fitness = node.fitness
        copy.y_true = node.y_true
        copy.y_pred = node.y_pred
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
            return 1
        return self.parent.depth() + 1

    def max_depth(self):
        """ Returns the maximum depth of the tree

        :return: maximum depth of the tree
        """
        return self.max_depth_helper(self.get_root())

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
        """ Size is the number of all nodes (mid-trees nodes + leaves) in the trees.

        :return: number of all nodes in the trees
        """
        return self.size_helper(self.get_root())

    @staticmethod
    def size_helper(n):
        if n is None:
            return 0
        return Node.size_helper(n.left) + Node.size_helper(n.right) + 1

    def make_node(self, depth=0, max_depth=None, random=None, att_indexes=None, att_values=None, class_count=None):
        """ Randomly generates the node and its children.

        :param depth: current depth of the tree
        :param max_depth: maximum depth of the tree
        :param random: random number generator
        :param att_indexes: attribute indexes
        :param att_values: attribute values
        :param class_count: number of classes
        :return: randomly generated node with children"""
        node = None
        att_index = None
        value_index = None
        att_value = None

        try:
            # if it's the root, first level or 50/50 chance of building new children.
            # Must be below maximal depth.
            if (depth <= 1 or (random.choice([True, False])) and depth < max_depth):
                subset_index = random.randint(0, len(att_indexes))
                att_index = att_indexes[subset_index]
                value_index = random.randint(0, len(att_values[att_index]))
                att_value = att_values[att_index][value_index]
                node = Node(att_index, att_value)
                node.left = self.make_node(
                    depth + 1, max_depth, random, att_indexes, att_values, class_count)
                node.right = self.make_node(
                    depth + 1, max_depth, random, att_indexes, att_values, class_count)
            else:  # result (leaf)
                r = random.randint(0, class_count)
                node = Node(att_value=r)
        except Exception as e:
            print(f"{att_index};{att_value};{value_index}")
            print("Error:", e)
            node = None

        return node

    def clear_evaluation(self):
        """ Clears the evaluation of this node and all its children
        """
        self.y_true = []
        self.y_pred = []
        if self.left:
            self.left.clear_evaluation()
        if self.right:
            self.right.clear_evaluation()

    def is_evaluated(self):
        """ Returns true if this node and all its children are evaluated

        :return: true if this node and all its children are evaluated
        """
        if len(self.y_pred) == 0 and len(self.y_true) == 0:
            return False

        left_evaluated = self.left.is_evaluated() if self.left else True
        right_evaluated = self.right.is_evaluated() if self.right else True

        return left_evaluated or right_evaluated

    def predict_one(self, X, y):
        """ Predicts the class of the given instance

        :param X: instance to be predicted
        :param y: actual class of the given instance
        :return: predicted class
        """
        try:
            actual = int(y)
            predicted = int(self.att_value)

            if self.att_index != -1:
                if X[self.att_index] > self.att_value:
                    if self.left is not None:
                        predicted = self.left.predict_one(X, y)
                else:
                    if self.right is not None:
                        predicted = self.right.predict_one(X, y)

            self.y_true.append(actual)
            self.y_pred.append(predicted)

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
    from sklearn.metrics import accuracy_score

    # Tree generation
    left_child = Node(att_index=1, att_value=3)
    left_grandchild1 = Node(att_index=2, att_value=2)
    left_grandchild2 = Node(att_index=2, att_value=4)
    left_child.set_left(left_grandchild1)
    left_child.set_right(left_grandchild2)

    right_child = Node(att_index=1, att_value=7)
    right_grandchild1 = Node(att_index=2, att_value=6)
    right_grandchild2 = Node(att_index=2, att_value=8)
    right_child.set_left(right_grandchild1)
    right_child.set_right(right_grandchild2)

    parent_node = Node(att_index=0, att_value=5)
    parent_node.set_left(left_child)
    parent_node.set_right(right_child)

    # Create instance and predict class
    instance = [6, 4, 5]
    actual_class = 1
    predicted_class = parent_node.predict_one(instance, actual_class)

    # Calculate accuracy
    accuracy = accuracy_score(parent_node.y_true, parent_node.y_pred)
    print(f"Predicted class for instance: {parent_node.y_pred}")
    print(f"Accuracy: {accuracy}")
