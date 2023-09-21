import random
from tree.node import Node


class Crossover:
    def crossover(self, p1, p2):
        n1 = Node.copy(p1)
        root_new = n1
        n2 = Node.copy(p2)
        size1 = n1.max_depth()
        size2 = n2.max_depth()

        print(n1.left)

        print(n1.parent)

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

        return root_new
