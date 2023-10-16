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
        if random.choice([True, False]):
            Mutation.change_class(node, class_count, random)

    @staticmethod
    def change_class(node, class_count, random):
        result_old = node.att_value
        result_new = result_old

        while result_old == result_new:  # classes must be different
            result_new = random.randint(0, class_count)

        node.att_value = result_new
