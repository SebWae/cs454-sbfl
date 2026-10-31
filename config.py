CROSSOVER_RATE = 1.0
DATA_DIR = "./bug_data"
POP_SIZE = 40
ITER_GP = 100
ITER_RUN = 1
MAX_DEPTH = 4
MUTATE_PROB = 0.08
RANK_SMOOTHING = 1.5
SEED_SPLIT = 261109
SPLIT_FILE = "split.json"

# for generating formulas as trees 
ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "sqrt": 1}
OPERATORS = ["add", "sub", "mul", "div", "sqrt"]
OPERANDS = ["ep", "ef", "np", "nf", "1"]
