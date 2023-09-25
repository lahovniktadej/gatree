class SettingOne:
    def __init__(self):
        self.RUN_COUNT = 10  # Number of individual runs
        self.POPULATION_SIZE = 150  # Number of individuals in one generation
        self.ELITE_SIZE = 1  # Elite automatically advances to the next generation. If -1, then children compete with parents, and only the best advance
        self.GENERATION_LIMIT = 2000  # Number of generations
        # Stop evolution when the best individual stays the same for x generations
        self.STAGNATION_LIMIT = 50

        # -------- MUTATION -------------------------
        # 0 = STANDARD
        # 1 = SMART - WORST NODE
        self.MUTATION_METHOD = 0
        self.MUTATION_PROBABILITY = 0.1
        # self.MUTATION_ATTRIBUTE_PROB = 0.03  # mutation probability for attribute nodes
        # self.MUTATION_SPLIT_PROB = 0.07  # mutation probability for split border
        # self.MUTATION_DECISION_PROB = 0.09  # mutation probability for decision nodes
        # self.MUTATION_ATTRIBUTE_DECISION_PROB = 0.01  # mutation probability for exchange of attribute to decision nodes
        # self.MUTATION_DECISION_ATTRIBUTE_PROB = 0.01  # mutation probability for exchange of decision to attribute nodes

        # -------- SCORES --------------------------
        self.PENALTY_NUM_NODES = 0.002  # penalty score for each node
        # penalty score for each unused decision node
        self.PENALTY_NUM_UNUSED_NODES = 0.02
        self.PENALTY_DYNAMIC = 20  # penalty factor for dynamic fitness

        # -------- SELECTION -----------------------
        # 0 = ROULETTE
        # 1 = LINEAR RANKING
        # 2 = LOG RANKING
        # 3 = TOURNAMENT
        self.SELECTION_METHOD = 3
        self.SELECTION_TOURNAMENT_SIZE = 2

        # -1= NO SMART SELECTION
        # 0 = SIZE COMPARISON
        # 1 = ATTRIBUTE COMPARISON
        # 2 = ATTRIBUTE VALUE COMPARISON
        # 3 = CONFUSION MATRIX COMPARISON
        # 4 = CLASSIFICATION DIF. COMPARISON
        # 5 = FAMILY SELECTION (prevent inbreeding; use with FAMILY TREE LEVELS)
        self.SELECTION_SMART_METHOD = -1

        # -------- CALCULATOR ----------------------
        # 0 = ACCURACY
        # 1 = VILI (accuracy with size and unused penalty)
        # 2 = FSCORE (average fscore with size and unused penalty)
        # 3 = PERCLASS (accuracy per class with size and unused penalty)
        # 4 = PERCLASSINVERSE (accuracy with size and unused penalty)
        # (smaller the percentage of class more weight it has)
        # 5 = AVERAGE CLASS ACCURACY
        # 6 = DYNAMIC FSCORE : fit = (1-fsc) + numNodes / (numInstances * PENALTY_DYNAMIC)
        # 7 = DYNAMIC AVG CLASS ACCURACY : fit = (1-fsc) + numNodes / (numInstances * PENALTY_DYNAMIC)
        # 8 = SAME AS 7, BUT CLASS ACCURACIES ARE WEIGHTED
        self.CALCULATOR_METHOD = 6

        # -------- CROSSOVER -----------------------
        # 0 = STANDARD
        # 1 = VILI
        # 2 = ACCURACY
        # 3 = FSCORE ALL (ZMNOZEK VSEH)
        # 4 = FSCORE - REPLACE WORST FSCORE WITH BEST
        # 5 = ONLY NODE - AS STANDARD BUT SUBTREES ARE NOT EXCHANGED
        # 6 = FSCORE AVERAGE - REPLACES THE WORST AVERAGE FSCORED NODE WITH THE BEST ONE
        # 7 = FSCORE WEIGHTED - REPLACES THE WORST WEIGHTED FSCORED NODE WITH THE BEST ONE
        # 8 = ACCURACY AND USAGE OF NODES (0.8 acc + 0.4 (percentage of instances that go through that node))
        # 9 = ACCURACY AND USAGE OF NODES (0.8 acc * 0.4 (percentage of instances that go through that node))
        # 10= SAME AS 8, BUT REPLACES WORST WITH RANDOM (BEST SO FAR)
        # 11= SAME AS 9, BUT REPLACES WORST WITH RANDOM
        # 12= NODE USAGE AVG
        # 13= NODE USAGE SUM
        # -1= NO CROSSOVER
        self.CROSSOVER_METHOD = 0

        # If(True) crossovers 2,3,4 don't settle with worst leaves, but they exchange subtree with those leaves
        self.CROSSOVER_ONLYLEAVES = False

        # Maximum depth of the trees in randomly generated trees in the starting population. -1 makes MAX_DEPTH J48 result * 2 depth.
        self.MAX_DEPTH = -1

        # -------- DUPLICATES -------------------
        # 0 = allow duplicates
        # 1 = remove duplicates after mutation (population is shrunk for 1 generation)
        # 2 = remove duplicates, but maintain the population size all the time
        self.DUPLICATE_REMOVER = 0

        # -------- FAMILY TREE -----------------
        self.FAMILY_TREE_LEVELS = 0  # Levels for family trees (0-off)

        # -------- MULTI POPULATIONS ----------
        self.MULTI_POP_NUM = 1  # Number of independent populations (islands)
        self.MULTI_MIGRATION_STEP = 100  # Every X generation migration happens
        self.MULTI_MIGRATION_SIZE = 10  # How many emigrate to another population

        # ------- ENSEMBLES -------------------
        # -1 = NO BAGGING
        # 0  = BAGGING, RANDOM
        # 1  = BAGGING, ACCORDING TO WEIGHTS (USE WITH BOOSTING)
        # 2  = RANDOM FOREST
        self.BAGGING_METHOD = -1
        self.BAGGING_STEP = 200

        # -1 = NO BOOSTING
        # 0  = BOOSTING, RESETS EVOLUTION AFTER EVERY BOOST
        # 1  = BOOSTING, CONTINUE EVOLUTION AFTER EVERY BOOST
        # 2  = SAME AS 1, BUT NO DUPLICATED CLASSIFIERS IN ENSEMBLE
        # 3  = LOGITBOOST
        self.BOOSTING_METHOD = -1
        self.BOOSTING_STEP = 200

        # ------- FITNESS SKIPPING -------------------
        # Fitness is not calculated X number of generations.
        # Selection is made based on the family trees.
        # Use in combination with FAMILY TREE LEVELS.
        # -1 = Off (don't use).
        # Fitness is not calculated X number of generations
        self.FITNESS_SKIP_GENERATIONS = -1

        # ------- PROBLEM TYPE -----------------------
        # 0  = Classification
        # 1  = Clustering
        # 2  = Regression
        self.PROBLEM_TYPE = 0

        # ------- VALIDATION SET -----------------------
        self.VALIDATION_SET_PERCENTAGE = 0.5
        self.VALIDATION_SET_WEIGHT = 1

    def __str__(self):
        result = "{"
        for attr, value in self.__dict__.items():
            result += f"{attr}: {value}, "
        result = result[:-2] + "}"
        return result

    def main(self):
        s = SettingOne()
        print(s)
