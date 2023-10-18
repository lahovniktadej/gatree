from gatree.tree.node import Node


class Mutation:
    """
    Mutation operations for tree nodes.
    """
    @staticmethod
    def mutation(root, att_indexes, att_values, class_count, random):
        """
        Apply mutation on a tree node.

        Args:
            root (Node): The root node of the tree.
            att_indexes (list): List of attribute indexes.
            att_values (list): List of attribute values.
            class_count (int): Number of classes.
            random (Random): Random number generator.

        Returns:
            Node: The mutated node.
        """
        node = Node.copy(root)
        depth = node.max_depth()

        while True:
            if node.att_index == -1:  # for leaves
                Mutation.mutate_leaf(
                    node, att_indexes, att_values, class_count, random)
                break
            elif random.randint(0, depth) == 0:  # for mid-tree nodes
                Mutation.mutate_operator(
                    node, att_indexes, att_values, class_count, random)
                break

            # go to next random child
            if random.choice([True, False]):
                if node.left is not None:
                    node = node.left
                else:
                    break
            else:
                if node.right is not None:
                    node = node.right
                else:
                    break

        return node.get_root()

    @staticmethod
    def mutate_leaf(node, att_indexes, att_values, class_count, random):
        """
        Mutate a leaf node.

        Args:
            node (Node): The leaf node.
            att_indexes (list): List of attribute indexes.
            att_values (list): List of attribute values.
            class_count (int): Number of classes.
            random (Random): Random number generator.
        """
        if random.choice([True, False]):  # change class
            Mutation.change_class(node, class_count, random)
        else:  # exchange for new subtree
            Mutation.exchange_class_for_tree(
                node, att_indexes, att_values, class_count, random)

    @staticmethod
    def change_class(node, class_count, random):
        """
        Change class of a leaf node.

        Args:
            node (Node): The leaf node.
            class_count (int): Number of classes.
            random (Random): Random number generator.
        """
        result_old = node.att_value
        result_new = result_old

        while result_old == result_new:  # classes must be different
            result_new = random.randint(0, class_count)

        node.att_value = result_new

    @staticmethod
    def exchange_class_for_tree(node, att_indexes, att_values, class_count, random):
        """
        Exchange a leaf node for a new subtree.

        Args:
            node (Node): The leaf node.
            att_indexes (list): List of attribute indexes.
            att_values (list): List of attribute values.
            class_count (int): Number of classes.
            random (Random): Random number generator.
        """
        parent = node.parent
        left = False
        if parent.left == node:
            left = True

        n = Node()
        subtree = n.make_node(max_depth=node.depth(), random=random,
                              att_indexes=att_indexes, att_values=att_values, class_count=class_count)
        subtree.parent = parent

        if left:
            parent.set_left(subtree)
        else:
            parent.set_right(subtree)

    @staticmethod
    def mutate_operator(node, att_indexes, att_values, class_count, random):
        """
        Mutate a mid-tree node.

        Args:
            node (Node): The mid-tree node.
            att_indexes (list): List of attribute indexes.
            att_values (list): List of attribute values.
            class_count (int): Number of classes.
            random (Random): Random number generator.
        """
        rand = random.random()
        if rand < 0.25 or node.parent == None:  # exchange attribute for other attribute
            Mutation.change_attribute(node, att_indexes, att_values, random)
        elif rand < 0.5:  # exchange attribute value for other value
            Mutation.change_attribute_value(node, att_values, random)
        elif rand < 0.75:  # exchange attribute for class
            Mutation.exchange_tree_for_class(node, class_count, random)
        else:  # exchange for new subtree
            Mutation.exchange_tree_for_tree(
                node, att_indexes, att_values, class_count, random)

    @staticmethod
    def change_attribute(node, att_indexes, att_values, random):
        """
        Change attribute of a mid-tree node.

        Args:
            node (Node): The mid-tree node.
            att_indexes (list): List of attribute indexes.
            att_values (list): List of attribute values.
            random (Random): Random number generator.
        """
        att_index_old = node.att_index
        att_index_new = att_index_old

        att_value_old = node.att_value
        att_value_new = att_value_old

        while att_index_old == att_index_new:
            att_index_new = random.randint(0, len(att_indexes))
            att_value_new = random.randint(0, len(att_values[att_index_new]))

        node.att_index = att_index_new
        node.att_value = att_values[att_index_new][att_value_new]

    @staticmethod
    def change_attribute_value(node, att_values, random):
        """
        Change attribute value of a mid-tree node.

        Args:
            node (Node): The mid-tree node.
            att_values (list): List of attribute values.
            random (Random): Random number generator.
        """
        att_index = node.att_index

        att_value_old = node.att_value
        att_value_new = att_value_old

        while att_value_old == att_value_new:
            att_value_new = random.randint(0, len(att_values[att_index]))

        node.att_value = att_values[att_index][att_value_new]

    @staticmethod
    def exchange_tree_for_class(node, class_count, random):
        """
        Exchange a mid-tree node for a class.

        Args:
            node (Node): The mid-tree node.
            class_count (int): Number of classes.
            random (Random): Random number generator.
        """
        parent = node.parent
        left = False
        if parent.left == node:
            left = True

        leaf = Node(att_index=-1, att_value=random.randint(0, class_count))
        leaf.parent = parent

        if left:
            parent.set_left(leaf)
        else:
            parent.set_right(leaf)

    @staticmethod
    def exchange_tree_for_tree(node, att_indexes, att_values, class_count, random):
        """
        Exchange a mid-tree node for a new subtree.

        Args:
            node (Node): The mid-tree node.
            att_indexes (list): List of attribute indexes.
            att_values (list): List of attribute values.
            class_count (int): Number of classes.
            random (Random): Random number generator.
        """
        parent = node.parent
        left = False
        if parent.left == node:
            left = True

        n = Node()
        subtree = n.make_node(max_depth=node.depth(), random=random,
                              att_indexes=att_indexes, att_values=att_values, class_count=class_count)
        subtree.parent = parent

        if left:
            parent.set_left(subtree)
        else:
            parent.set_right(subtree)
