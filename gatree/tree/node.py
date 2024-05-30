class Node:
    """
    Node class for decision trees. Each node represents a decision in the tree. The node can be a mid-tree node or a leaf node. Mid-tree nodes have an attribute index and an attribute value, while leaf nodes have a class index. The node can have a parent, left child, and right child. The node also has a fitness value, true class values, and predicted class values for evaluation.

    Args:
        att_index (int, optional): Attribute index or -1 if this is a leaf.
        att_value (any, optional): Attribute value or class index if this is a leaf.

    Attributes:
        parent (Node): Parent node.
        left (Node): Left child node.
        right (Node): Right child node.
        att_index (int, optional): Attribute index or -1 if this is a leaf.
        att_value (any, optional): Attribute value or class index if this is a leaf.
        fitness (None): Placeholder for fitness value.
        y_true (list): List of true class values for evaluation.
        y_pred (list): List of predicted class values for evaluation.
        prediction_ratio (dict): Dictionary of prediction ratios for each class in the node.
    """

    def __init__(self, att_index=None, att_value=None):
        """
        Initializes the node with the given parameters. If the attribute index is -1, then this is a leaf node.

        Args:
            att_index (int, optional): Attribute index or -1 if this is a leaf.
            att_value (any, optional): Attribute value or class index if this is a leaf.
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
            self.prediction_ratio = {}
            self.predictions = {}

    @staticmethod
    def copy(node, parent=None):
        """
        Returns a deep copy of the given node. The parent node can be specified for the copied node.

        Args:
            node (Node): Node to be copied.
            parent (Node, optional): Parent node of the copied node.

        Returns:
            Node: Deep copy of the given node.
        """
        copy = Node(node.att_index, node.att_value)
        copy.parent = parent
        copy.left = Node.copy(
            node.left, parent=copy) if node.left is not None else None
        copy.right = Node.copy(
            node.right, parent=copy) if node.right is not None else None
        copy.fitness = node.fitness
        copy.y_true = node.y_true
        copy.y_pred = node.y_pred
        copy.prediction_ratio = node.prediction_ratio
        return copy

    def set_left(self, n):
        """
        Sets the left child of this node to the given node. Also sets the parent of the given node to this node.

        Args:
            n (Node): Node to be set as the left child.
        """
        self.left = n
        n.parent = self

    def set_right(self, n):
        """
        Sets the right child of this node to the given node. Also sets the parent of the given node to this node.

        Args:
            n (Node): Node to be set as the right child.
        """
        self.right = n
        n.parent = self

    def get_root(self):
        """
        Returns the root node of the tree.

        Returns:
            Node: Root node of the tree.
        """
        if self.parent is None:
            return self
        return self.parent.get_root()

    def get_children(self):
        """
        Returns the children of this node.

        Returns:
            tuple: Tuple containing the left and right children nodes.
        """
        return self.left, self.right

    def get_leaves(self):
        """
        Returns the leaves of this node.

        Returns:
            list: List of leaves of this node.
        """
        if self.left is None:
            return [self]
        return self.left.get_leaves() + self.right.get_leaves()

    def depth(self):
        """
        Returns the depth of this node.

        Returns:
            int: Depth of this node.
        """
        if self.parent is None:
            return 1
        return self.parent.depth() + 1

    def max_depth(self):
        """
        Returns the maximum depth of the tree. The depth of the tree is the number of edges on the longest path from the root to a leaf.

        Returns:
            int: Maximum depth of the tree.
        """
        return self.max_depth_helper(self.get_root())

    @staticmethod
    def max_depth_helper(n):
        """
        Helper function for max_depth.

        Args:
            n (Node): Node to be used as the root.

        Returns:
            int: Maximum depth of the tree.
        """
        if n is None:
            return 0
        l_depth = Node.max_depth_helper(n.left)
        r_depth = Node.max_depth_helper(n.right)
        return max(l_depth, r_depth) + 1

    def size(self):
        """
        Size is the number of all nodes (mid-tree nodes + leaves) in the trees.

        Returns:
            int: Number of all nodes in the trees.
        """
        return self.size_helper(self.get_root())

    @staticmethod
    def size_helper(n):
        """
        Helper function for size.

        Args:
            n (Node): Node to be used as the root.

        Returns:
            int: Number of all nodes in the trees.
        """
        if n is None:
            return 0
        return Node.size_helper(n.left) + Node.size_helper(n.right) + 1

    def make_node(self, depth=0, max_depth=None, random=None, att_indexes=None, att_values=None, class_count=None):
        """
        Randomly generates the node and its children. The depth of the tree and the maximum depth of the tree can be specified. The random number generator, attribute indexes, attribute values, and number of classes must be provided.

        Args:
            depth (int, optional): Current depth of the tree.
            max_depth (int, optional): Maximum depth of the tree.
            random (Random, optional): Random number generator.
            att_indexes (numpy.ndarray, optional): Attribute indexes.
            att_values (dict, optional): Attribute values.
            class_count (int, optional): Number of classes.

        Returns:
            Node: Randomly generated node with children.
        """
        node = None
        att_index = None
        value_index = None
        att_value = None
        if max_depth == None:
            max_depth = depth

        try:
            # if it's the root, first level or 50/50 chance of building new children.
            # Must be below maximal depth.
            if (depth <= 1 or (random.choice([True, False])) and depth < max_depth):
                subset_index = random.randint(0, len(att_indexes))
                att_index = att_indexes[subset_index]
                value_index = random.randint(0, len(att_values[att_index]))
                att_value = att_values[att_index][value_index]
                node = Node(att_index=att_index, att_value=att_value)
                node.left = self.make_node(depth=depth + 1, max_depth=max_depth, random=random,
                                           att_indexes=att_indexes, att_values=att_values, class_count=class_count)
                node.left.parent = node
                node.right = self.make_node(depth=depth + 1, max_depth=max_depth, random=random,
                                            att_indexes=att_indexes, att_values=att_values, class_count=class_count)
                node.right.parent = node
            else:  # result (leaf)
                r = random.randint(0, class_count)
                node = Node(att_index=-1, att_value=r)
        except Exception as e:
            print(f"{att_index};{att_value};{value_index}")
            print("Error:", e)
            node = None

        return node

    def clear_evaluation(self):
        """
        Clears the evaluation of this node and all its children. The true class values and predicted class values are reset to empty lists.
        """
        self.y_true = []
        self.y_pred = []
        self.prediction_ratio = {}
        if self.left:
            self.left.clear_evaluation()
        if self.right:
            self.right.clear_evaluation()

    def is_evaluated(self):
        """
        Returns true if this node and all its children are evaluated. A node is evaluated if it has true class values and predicted class values.

        Returns:
            bool: True if this node and all its children are evaluated.
        """
        if len(self.y_pred) == 0 and len(self.y_true) == 0:
            return False

        left_evaluated = self.left.is_evaluated() if self.left else True
        right_evaluated = self.right.is_evaluated() if self.right else True

        return left_evaluated or right_evaluated

    def predict_one(self, X, y=None):
        """
        Predicts the class of the given instance. If the actual class is provided, the true class value and predicted class value are stored for evaluation.

        Args:
            X (list): Instance to be predicted.
            y (int): Actual class of the given instance.

        Returns:
            int: Predicted class.
        """
        try:
            if self.att_index != -1:
                if X.iloc[self.att_index] > self.att_value:
                    if self.left is not None:
                        predicted = self.left.predict_one(X, y)
                else:
                    if self.right is not None:
                        predicted = self.right.predict_one(X, y)
            else:
                predicted = int(self.att_value)

            if y is not None:
                self.y_true.append(int(y))
                self.y_pred.append(predicted)

            return predicted
        except Exception as e:
            print(e)
            return -1

    def calculate_prediction_ratio(self, ratio_scope, classes):
        """
        Calculates the prediction ratio for this node. The prediction ratio is the ratio of the predicted class values to the total number of instances in the node. The prediction ratio is stored in the prediction_ratio attribute.

        Args:
            ratio_scope (str): Scope of the prediction ratio. 'local' calculates the prediction ratio only for this node. 'global' calculates the prediction ratio for this node and its parent nodes.
            classes (list): List of class indexes.
        """
        for cls in classes:
            self.predictions[cls] = 0
            self.prediction_ratio[cls] = 0.0

        for cls in set(self.y_pred):
            self.predictions[cls] = self.y_pred.count(cls)

        if ratio_scope == 'local':
            for cls in set(self.y_pred):
                self.prediction_ratio[cls] = self.y_pred.count(
                    cls) / len(self.y_pred)
        if ratio_scope == 'global':
            for cls in set(self.y_pred):
                if not self.parent:
                    self.prediction_ratio[cls] = self.y_pred.count(
                        cls) / len(self.y_pred)
                else:
                    self.prediction_ratio[cls] = self.y_pred.count(
                        cls) / len(self.y_pred) + self.parent.calculate_prediction_global(cls, len(self.y_pred), f'{self.y_pred.count(cls)} / {len(self.y_pred)} + ')

        if self.left:
            self.left.calculate_prediction_ratio(ratio_scope, classes)
        if self.right:
            self.right.calculate_prediction_ratio(ratio_scope, classes)

    def calculate_prediction_global(self, cls, instances=None, formula=''):
        """
        Helper function for calculate_prediction_ratio. Recursively gathers the prediction ratio of the parent nodes.

        Args:
            cls (int): Class index.
            instances (int): Number of instances in the child node.
        """
        if self.parent:
            formula += f'{self.y_pred.count(cls)} / {len(self.y_pred)} * {instances} / {len(self.y_pred)} + '
            return self.y_pred.count(cls) / len(self.y_pred) * instances / len(self.y_pred) + self.parent.calculate_prediction_global(cls, instances, formula)
        formula += f'{instances} / {len(self.y_pred)} * {self.y_pred.count(cls)} / {len(self.y_pred)}'
        return instances / len(self.y_pred) * (self.y_pred.count(cls) / len(self.y_pred))

    def normalise_prediction_ratio(self, ratio_scope):
        """
        Normalises the prediction ratio for this node. The prediction ratio is normalised to the range [0, 1]. The normalised prediction ratio is stored in the prediction_ratio attribute.

        Args:
            ratio_scope (str): Scope of the prediction ratio. 'local' calculates the prediction ratio only for this node. 'global' calculates the prediction ratio for this node and its parent nodes.
        """
        if ratio_scope == 'local':
            return

        sum_prediction_ratio = sum([self.prediction_ratio[cls]
                                   for cls in self.prediction_ratio if self.prediction_ratio[cls] != 0])
        for cls in self.prediction_ratio:
            if self.prediction_ratio[cls] != 0:
                self.prediction_ratio[cls] /= sum_prediction_ratio

        if self.left:
            self.left.normalise_prediction_ratio(ratio_scope)
        if self.right:
            self.right.normalise_prediction_ratio(ratio_scope)

    def __str__(self):
        """
        Returns a string representation of this node.

        Returns:
            str: String representation of this node.
        """
        if self.att_index == -1:
            # Assuming att_value is a numeric index
            return str(int(self.att_value))
            # For nominal values, you might need a mapping from index to value
        else:
            return f"Attribute {self.att_index} > {self.att_value}"

    def __repr__(self):
        """
        Returns a string representation of this node.

        Returns:
            str: String representation of this node.
        """
        return self.__str__()
