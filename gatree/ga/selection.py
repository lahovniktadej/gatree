class Selection:
    """
    Selection of trees for crossover and mutation. The selection operation is used to select two trees from the population for crossover or mutation. The selection operation is based on the tournament selection method, where a random subset of trees is selected and the best tree is chosen for crossover or mutation.

    Args:
        population (list): List of trees.
        selection_tournament_size (int): Number of trees to select.
        random (Random): Random number generator.
    """
    def selection(population, selection_tournament_size, random):
        """
        Select two trees for crossover or mutation. The selection operation is based on the tournament selection method, where a random subset of trees is selected and the best tree is chosen for crossover or mutation.

        Args:
            population (list): List of trees.
            selection_tournament_size (int): Number of trees to select.
            random (Random): Random number generator.

        Returns:
            tuple: The two selected trees.
        """
        valid = False
        while not valid:
            selection = []

            # Select two trees
            for _ in range(2):
                indices = random.choice(len(population),
                                        selection_tournament_size, replace=False)
                candidates = [population[i] for i in indices]
                candidates.sort(key=lambda x: x.fitness, reverse=False)
                selection.append(candidates[0])

            # Check if trees are different
            if selection[0] != selection[1]:
                valid = True

        return selection[0], selection[1]
