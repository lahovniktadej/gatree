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

    def predict_one(self, instance):
        """ Predicts the class of the given instance

        :param instance: instance to be predicted
        :return: predicted class
        """
        try:
            actual = int(instance[-1])
            predicted = int(self.att_value)

            if self.att_index != -1:
                if instance[self.att_index] > self.att_value:
                    predicted = self.left.predict_one(instance)
                    self.matrix[actual][predicted] += 1
                else:
                    predicted = self.right.predict_one(instance)
                    self.matrix[actual][predicted] += 1
            else:
                self.matrix[actual][predicted] += 1

            return predicted
        except Exception as e:
            print(e)
            return -1

    def get_accuracy(self):
        """ Returns the accuracy of this node

        :return: accuracy of this node
        """
        y_true, y_pred = self.get_true_pred_arrays()
        return accuracy_score(y_true, y_pred)

    def get_weighted_accuracy(self):
        """ Returns the weighted accuracy of this node

        :return: weighted accuracy of this node
        """
        class_counts = [sum(row) for row in self.matrix]
        class_count_sum = sum(class_counts)
        weighted_acc_total = 0

        for i in range(len(self.matrix)):
            y_true = [i] * int(class_counts[i])
            y_pred = [j for j in range(len(self.matrix))
                      for _ in range(int(self.matrix[i][j]))]
            weighted_acc_total += accuracy_score(y_true,
                                                 y_pred) * class_counts[i]

        return weighted_acc_total / class_count_sum

    def get_avg_accuracy(self):
        """ Returns the average accuracy of this node

        :return: average accuracy of this node
        """
        accuracies = self.get_class_accuracies()
        return sum(accuracies) / len(accuracies)

    def get_class_accuracies(self):
        """ Returns the accuracies of each class of this node

        :return: accuracies of each class of this node
        """
        return self.get_recalls()

    def get_min_class_accuracy(self):
        """ Returns the minimum class accuracy of this node

        :return: minimum class accuracy of this node
        """
        accuracies = self.get_class_accuracies()
        return min(accuracies)

    def get_max_class_accuracy(self):
        """ Returns the maximum class accuracy of this node

        :return: maximum class accuracy of this node
        """
        accuracies = self.get_class_accuracies()
        return max(accuracies)

    def get_precision(self):
        """ Returns the precision of this node

        :return: precision of this node
        """
        y_true, y_pred = self.get_true_pred_arrays()
        return precision_score(y_true, y_pred, average='macro')

    def get_precisions(self):
        """ Returns the precisions of each class of this node

        :return: precisions of each class of this node
        """
        precisions = [0.0] * len(self.matrix[0])

        for i in range(len(self.matrix[0])):
            predicted_positives = sum(row[i] for row in self.matrix)
            if predicted_positives == 0:
                precisions[i] = 0
            else:
                precisions[i] = self.matrix[i][i] / predicted_positives

        return precisions

    def get_recall(self):
        """ Returns the recall of this node

        :return: recall of this node
        """
        y_true, y_pred = self.get_true_pred_arrays()
        return recall_score(y_true, y_pred, average='macro')

    def get_recalls(self):
        """ Returns the recalls of each class of this node

        :return: recalls of each class of this node
        """
        recalls = [0.0] * len(self.matrix)

        for i in range(len(self.matrix)):
            actual_positives = sum(self.matrix[i])
            if actual_positives == 0:
                recalls[i] = 0
            else:
                recalls[i] = self.matrix[i][i] / actual_positives

        return recalls

    def get_average_fscore(self):
        """ Returns the average fscore of this node

        :return: average fscore of this node
        """
        y_true, y_pred = self.get_true_pred_arrays()
        return f1_score(y_true, y_pred, average='macro')

    def get_multiplicated_fscore(self):
        """ Returns the multiplicated fscore of this node

        :return: multiplicated fscore of this node
        """
        fscores = self.get_fscores()
        fscore = fscores[0]

        for i in range(1, len(fscores)):
            fscore *= fscores[i]

        return fscore

    def get_weighted_fscore(self):
        """ Returns the weighted fscore of this node

        :return: weighted fscore of this node
        """
        y_true, y_pred = self.get_true_pred_arrays()
        return f1_score(y_true, y_pred, average='weighted')

    def get_fscores(self):
        """ Returns the fscores of each class of this node

        :return: fscores of each class of this node
        """
        y_true, y_pred = self.get_true_pred_arrays()
        return f1_score(y_true, y_pred, average=None)

    def get_class_accuracy(self, class_num):
        """ Returns the accuracy of the given class of this node

        :param class_num: index of the class
        :return: accuracy of the given class of this node
        """
        if class_num < 0 or class_num >= len(self.matrix):
            return 0

        correct = self.matrix[class_num][class_num]
        all_count = sum(row[class_num] for row in self.matrix)

        if all_count > 0:
            return correct / all_count
        else:
            return 0

    def get_true_pred_arrays(self):
        """" Returns the true and predicted class arrays of this node

        :return: true and predicted class arrays of this node
        """
        y_true = []
        y_pred = []

        for i in range(len(self.matrix)):
            for j in range(len(self.matrix)):
                y_true.extend([i] * int(self.matrix[i][j]))
                y_pred.extend([j] * int(self.matrix[i][j]))

        return y_true, y_pred

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
