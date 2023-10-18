from gatree.tree.node import Node


class Crossover:
    """
    Crossover operations for genetic algorithm.
    """
    def crossover(tree1, tree2, random):
        """
        Perform crossover between two trees.

        Args:
            tree1 (Node): The first tree for crossover.
            tree2 (Node): The second tree for crossover.
            random (Random): Random number generator.

        Returns:
            Node: The new tree resulting from crossover.
        """
        n1 = Node.copy(tree1)
        n2 = Node.copy(tree2)
        size1 = n1.max_depth()
        size2 = n2.max_depth()

        while True:
            if (n1.left == None or random.randint(0, size1) == 0) and n1.parent != None:
                break

            if random.choice([True, False]):
                n1 = n1.left
            else:
                n1 = n1.right

        while True:
            if (n2.left == None or random.randint(0, size2) == 0) and n2.parent != None:
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
