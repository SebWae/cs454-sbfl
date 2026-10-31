import copy
import csv
import os
import random
from tqdm import tqdm

from config import POP_SIZE, ITER_GP, ITER_RUN, MAX_DEPTH, OPERANDS, OPERATORS
from gp import create_offsprings, eval_pop, init_pop, load_bugs, mutate
from utils import best_of, make_split, to_python

P_CROSS, P_MUT, N_ELITE = 0.9, 0.1, 1
RESULTS_CSV = "candidates.csv"


if __name__ == "__main__":
    train_ids, val_ids = make_split()
    train_bugs = load_bugs(train_ids)
    val_bugs = load_bugs(val_ids)

    write_header = not os.path.exists(RESULTS_CSV)
    fields = ["run", "formula", "python_expr", "train_acc1", "train_wef",
              "val_acc1", "val_wef"]

    for run in range(ITER_RUN):
        random.seed(run)                      # reproducible per run
        pop = init_pop(POP_SIZE, OPERANDS, OPERATORS, max_depth=MAX_DEPTH)

        best_tree, best_fit = None, None      # best over ALL generations
        for _ in tqdm(range(ITER_GP), desc=f"run {run}"):
            fits = eval_pop(pop, train_bugs, mode="linear", s=1.5)

            gen_best, gen_fit = best_of(pop, fits)
            if best_fit is None or (gen_fit["acc1"], -gen_fit["wef"]) > (best_fit["acc1"], -best_fit["wef"]):
                best_tree, best_fit = copy.deepcopy(gen_best), gen_fit

            probs = [f["prob"] for f in fits]

            # elitism: carry the generation's best over unchanged
            elite_idx = sorted(range(len(pop)),
                               key=lambda i: (fits[i]["acc1"], -fits[i]["wef"]),
                               reverse=True)[:N_ELITE]
            new_pop = [copy.deepcopy(pop[i]) for i in elite_idx]

            while len(new_pop) < POP_SIZE:
                p1, p2 = random.choices(pop, weights=probs, k=2)
                c1, c2 = create_offsprings(p1, p2, crossover_rate=P_CROSS, max_depth=MAX_DEPTH)
                c1 = mutate(c1, p=P_MUT, max_depth=MAX_DEPTH)
                c2 = mutate(c2, p=P_MUT, max_depth=MAX_DEPTH)
                new_pop.extend([c1, c2])
            pop = new_pop[:POP_SIZE]

        # the last generation was never evaluated inside the loop
        fits = eval_pop(pop, train_bugs, mode="linear", s=1.5)
        gen_best, gen_fit = best_of(pop, fits)
        if (gen_fit["acc1"], -gen_fit["wef"]) > (best_fit["acc1"], -best_fit["wef"]):
            best_tree, best_fit = copy.deepcopy(gen_best), gen_fit

        # score the candidate once on the held-out validation bugs
        val_fit = eval_pop([best_tree], val_bugs)[0]

        row = {
            "run": run,
            "formula": str(best_tree),
            "python_expr": to_python(best_tree),
            "train_acc1": best_fit["acc1"],
            "train_wef": round(best_fit["wef"], 3),
            "val_acc1": val_fit["acc1"],
            "val_wef": round(val_fit["wef"], 3),
        }
        with open(RESULTS_CSV, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row)
        print(row)