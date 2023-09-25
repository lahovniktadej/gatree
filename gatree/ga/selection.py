import random


class TournamentSelector:
    def getSelection(self, population, selection_tournament_size, elite_size):
        selection = []

        # Number of pairs
        selection_size = len(population) - elite_size
        if selection_size % 2 == 1:
            selection_size += 1

        for i in range(selection_size):
            ints = random.sample(range(len(population)),
                                 selection_tournament_size)
            candidates = [population[idx] for idx in ints]

            candidates.sort()

            # Check if spouse1 equals spouse2
            if i % 2 == 1 and candidates[0] == selection[-1]:
                i -= 1
                continue

            selection.append(candidates[0])

        return selection
