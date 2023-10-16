from tree.node import Node


class Mutation:
    @staticmethod
    def mutation(root, att_indexes, att_values, class_count, random):
        node = Node.copy(root)
        depth = node.max_depth()

        while True:
            if node.att_index == -1:  # for leaves
                Mutation.mutate_leaf(
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
        """ Mutate leaf node

        :param node: leaf node
        :param att_indexes: attribute indexes
        :param att_values: attribute values
        :param class_count: number of classes
        :param random: random number generator
        """
        if random.choice([True, False]):  # change class
            Mutation.change_class(node, class_count, random)
        else:  # exchange for new subtree
            Mutation.exchange_class_for_tree(
                node, att_indexes, att_values, class_count, random)

    @staticmethod
    def change_class(node, class_count, random):
        """ Change class of the leaf node

        :param node: leaf node
        :param class_count: number of classes
        :param random: random number generator
        """
        result_old = node.att_value
        result_new = result_old

        while result_old == result_new:  # classes must be different
            result_new = random.randint(0, class_count)

        node.att_value = result_new

    @staticmethod
    def exchange_class_for_tree(node, att_indexes, att_values, class_count, random):
        """ Exchange leaf node for new subtree

        :param node: leaf node
        :param att_indexes: attribute indexes
        :param att_values: attribute values
        :param class_count: number of classes
        :param random: random number generator
        """
        parent = node.parent
        left = False
        if parent.left == node:
            left = True

        print(node.depth())

        n = Node()
        subtree = n.make_node(max_depth=node.depth(), random=random,
                              att_indexes=att_indexes, att_values=att_values, class_count=class_count)
        subtree.parent = parent

        if left:
            parent.set_left(subtree)
        else:
            parent.set_right(subtree)
