import os
import random
from collections import Counter, defaultdict

from config import (
    INIT_POP_SIZE,
    ITER_GP,
    ITER_RUN,
    MAX_DEPTH,
    OPERANDS,
    OPERATORS,
    SEED_SPLIT,
)
from gp import (
    init_pop,
    load_bugs,
    eval_pop,
)


if __name__ == "__main__":
    # number of total bugs and bugs per project
    bugs = [f[:-5] for f in os.listdir("./bug_data") if f.endswith(".json")]
    print(len(bugs), Counter(b.split("-")[0] for b in bugs))

    # divide bugs in train and eval set
    by_project = defaultdict(list)
    for b in bugs:                       # e.g. "Lang-1"
        by_project[b.split("-")[0]].append(b)

    train, val = [], []
    for proj, bugs in by_project.items():
        random.shuffle(bugs)
        k = round(0.7 * len(bugs))
        train += bugs[:k]
        val += bugs[k:]

    # load training bugs
    bugs = load_bugs(train)

    # initialize population of formulas
    pop = init_pop(INIT_POP_SIZE, OPERANDS, OPERATORS, max_depth=MAX_DEPTH)

    # evaluate current population
    fits = eval_pop(pop, bugs, mode="linear", s=1.5)

    # obtain probabilities
    probs = [f["prob"] for f in fits]

    # sample parents according to probabilities
    for _ in range(10):
        a, b = random.choices(pop, weights=probs, k=2)
        print("Parent A: ", a)
        print("Parent B: ", b)


    # for _ in range(ITER_RUN):
    #     for _ in range(ITER_GP):   
