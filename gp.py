import json
import math 
import numpy as np
import os
import pandas as pd
import random

from config import (
    ARITY, 
    DATA_DIR, 
    RANK_SMOOTHING,
)


class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

    def __str__(self):
        if not self.children:
            return self.value
        return f"{self.value}({', '.join(map(str, self.children))})"


def _random_tree(depth, max_depth, operands, operators, method):
    """Build a random tree. Root has depth 0; leaves may sit at max_depth."""
    if depth == max_depth:
        return Node(random.choice(operands))        # depth limit: terminal only

    if method == "full":
        use_operator = True                          # operators until the limit
    else:  # grow
        # root must be an operator, otherwise the tree is a trivial single leaf
        if depth == 0:
            use_operator = True
        else:
            # pick uniformly over ALL symbols, so the operator/operand ratio
            # biases the average tree size
            pool = len(operators) + len(operands)
            use_operator = random.random() < len(operators) / pool

    if not use_operator:
        return Node(random.choice(operands))

    op = random.choice(operators)
    children = [
        _random_tree(depth + 1, max_depth, operands, operators, method)
        for _ in range(ARITY[op])
    ]
    return Node(op, children)


def init_pop(pop_size, operands, operators, max_depth):
    """Ramped half-and-half initialisation.

    The tree depth limit is ramped from 2 up to max_depth across the
    population, and for each depth limit half of the trees are built with
    'full' and half with 'grow'.
    """
    depths = list(range(2, max_depth + 1))           # e.g. [2, 3, 4]
    population = []
    for i in range(pop_size):
        depth_limit = depths[i % len(depths)]
        method = "full" if (i // len(depths)) % 2 == 0 else "grow"
        population.append(
            _random_tree(0, depth_limit, operands, operators, method)
        )
    return population


def get_spectrum(coverage_df, failing_tests):
    """
    Convert coverage data and failed test results to program execution spectrum

    Return: (e_p, n_p, e_f, n_f)
        - e_p: the number of passing tests that cover each elements
        - n_p: the number of passing tests that do not cover each elements
        - e_f: the number of failing tests that cover each elements
        - n_f: the number of failing tests that do not cover each elements
    """
    X = coverage_df.values.transpose()

    is_failing = np.array([test in failing_tests for test in coverage_df.columns])

    e_p = X[~is_failing].sum(axis=0)
    e_f = X[is_failing].sum(axis=0)
    n_p = np.sum(~is_failing) - e_p
    n_f = np.sum(is_failing) - e_f

    return e_p, n_p, e_f, n_f
    

def load_bug(bug_id, data_dir=DATA_DIR):
    with open(os.path.join(data_dir, f"{bug_id}.json")) as f:
        bug_info = json.load(f)
    coverage = pd.read_pickle(os.path.join(data_dir, f"{bug_id}-cov.pkl"))

    # get_spectrum returns (e_p, n_p, e_f, n_f) -- note the order
    e_p, n_p, e_f, n_f = get_spectrum(coverage, bug_info["failing_tests"])

    # statement -> method, then keep the unique spectra tuples per method
    # (max aggregation only needs the distinct tuples, which saves a lot of work)
    method_spectra = {}
    for line, ep, np_, ef, nf in zip(coverage.index, e_p.tolist(), n_p.tolist(),
                                     e_f.tolist(), n_f.tolist()):
        method = line.rsplit(":", 1)[0]          # strip the ":<line number>"
        method_spectra.setdefault(method, set()).add((ep, ef, np_, nf))

    buggy_methods = {l.rsplit(":", 1)[0] for l in bug_info["buggy_lines"]}
    buggy_methods &= method_spectra.keys()       # guard against missing methods

    return {
        "id": bug_id,
        "methods": {m: list(s) for m, s in method_spectra.items()},
        "buggy": buggy_methods,
    }


def load_bugs(bug_ids):
    return [load_bug(b) for b in bug_ids]


def eval_tree(node, env):
    v = node.value
    if not node.children:
        return 1.0 if v == "1" else env[v]
    a = [eval_tree(c, env) for c in node.children]
    if v == "add":  return a[0] + a[1]
    if v == "sub":  return a[0] - a[1]
    if v == "mul":  return a[0] * a[1]
    if v == "div":  return 1.0 if a[1] == 0 else a[0] / a[1]   # safe div from README
    if v == "sqrt": return math.sqrt(abs(a[0]))


def score(tree, spectrum):
    ep, ef, np_, nf = spectrum
    s = eval_tree(tree, {"ep": ep, "ef": ef, "np": np_, "nf": nf})
    return -math.inf if s != s else s            # NaN (inf - inf) -> worst score


def buggy_rank(tree, bug):
    # method score = max over its statements
    scores = {m: max(score(tree, t) for t in tuples)
              for m, tuples in bug["methods"].items()}
    ranks = []
    for b in bug["buggy"]:
        # conservative ties: everything scoring >= the buggy method counts
        # ahead of it, itself included (rank 1 only if strictly the top)
        ranks.append(sum(1 for s in scores.values() if s >= scores[b]))
    return min(ranks) if ranks else len(scores)   # best-ranked buggy method


def assign_ranks(results):
    """Rank 0 = worst, mu-1 = best. Order: acc@1 (higher better), then wef (lower better).
    Individuals with identical (acc1, wef) share the average of the ranks they span."""
    mu = len(results)
    key = lambda i: (results[i]["acc1"], -results[i]["wef"])
    order = sorted(range(mu), key=key)              # worst -> best
    ranks = [0.0] * mu
    pos = 0
    while pos < mu:
        end = pos
        while end + 1 < mu and key(order[end + 1]) == key(order[pos]):
            end += 1
        avg = (pos + end) / 2
        for k in range(pos, end + 1):
            ranks[order[k]] = avg
        pos = end + 1
    return ranks


def selection_probs(ranks, mode="linear", s=RANK_SMOOTHING):
    mu = len(ranks)
    if mu == 1:
        return [1.0]
    if mode == "linear":
        if not 1.0 <= s <= 2.0:
            raise ValueError("s must be in [1, 2]")
        total = sum(ranks)                           # = mu*(mu-1)/2, tie-averaging preserves it
        return [((2 - s) / mu) + ((r * (s - 1)) / total) for r in ranks]
    elif mode == "exponential":
        w = [1 - math.exp(-r) for r in ranks]
        total = sum(w)
        return [x / total for x in w]
    raise ValueError(f"Unknown mode: {mode}")


def eval_pop(population, bugs, mode="linear", s=RANK_SMOOTHING):
    results = []
    for tree in population:
        ranks_per_bug = [buggy_rank(tree, bug) for bug in bugs]
        top1 = sum(r == 1 for r in ranks_per_bug)
        results.append({
            "acc1": top1,
            "acc1_pct": 100.0 * top1 / len(bugs),
            "wef": sum(ranks_per_bug) / len(ranks_per_bug),
        })

    ranks = assign_ranks(results)
    probs = selection_probs(ranks, mode, s)
    for res, r, p in zip(results, ranks, probs):
        res["rank"] = r
        res["prob"] = p
    return results




# def rank_select():


# def create_offsprings():


# def mutate_offsprings():


