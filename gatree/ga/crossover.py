from tree.node import Node


class Crossover:
    """ Crossover class for genetic algorithm """
    def crossover(tree1, tree2, random):
        """ Crossover two trees

        :param tree1: first tree
        :param tree2: second tree
        :param random: random number generator
        :return: new tree
        """
        n1 = Node.copy(tree1)
        n2 = Node.copy(tree2)
        size1 = n1.max_depth()
        size2 = n2.max_depth()

        while True:
            if (n1.left == None or random.randint(0, size1 - 1) == 0) and n1.parent != None:
                break

            if random.choice([True, False]):
                n1 = n1.left
            else:
                n1 = n1.right

        while True:
            if (n2.left == None or random.randint(0, size2 - 1) == 0) and n2.parent != None:
                break

            if random.choice([True, False]):
                n2 = n2.left
            else:
                n2 = n2.right

        p = n1.parent
        if p.left == n1:
            p.set_left(n2)
        else:
            p.set_right(n2)

        return p.get_root()
