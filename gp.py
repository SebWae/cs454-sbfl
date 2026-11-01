import copy
import json
import math 
import numpy as np
import os
import pandas as pd
import random

from config import (
    ARITY, 
    CROSSOVER_RATE,
    DATA_DIR, 
    MAX_DEPTH,
    MUTATE_PROB,
    OPERANDS,
    OPERATORS,
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


def _collect_nodes(root):
    """Return a list of (parent, child_index, node) for every non-root node,
    plus the root itself as (None, None, root)."""
    nodes = [(None, None, root)]
    stack = [root]
    while stack:
        parent = stack.pop()
        for i, child in enumerate(parent.children):
            nodes.append((parent, i, child))
            stack.append(child)
    return nodes


def _depth(node):
    return 0 if not node.children else 1 + max(_depth(c) for c in node.children)


def create_offsprings(parent_a, parent_b, crossover_rate=CROSSOVER_RATE, max_depth=MAX_DEPTH):
    """Single-point subtree crossover.

    Picks one cutpoint in each parent and swaps the subtrees below them.
    Returns two new trees; the parents are left untouched.
    If max_depth is given, a swap that would produce a tree deeper than
    max_depth is rejected and copies of the parents are returned instead.
    """
    child_a = copy.deepcopy(parent_a)
    child_b = copy.deepcopy(parent_b)

    if random.random() >= crossover_rate:
        return child_a, child_b

    pa, ia, na = random.choice(_collect_nodes(child_a))
    pb, ib, nb = random.choice(_collect_nodes(child_b))

    # swap the subtrees
    if pa is None and pb is None:          # both cutpoints at the root
        child_a, child_b = nb, na
    elif pa is None:                       # cut at the root of A only
        pb.children[ib] = na
        child_a = nb
    elif pb is None:                       # cut at the root of B only
        pa.children[ia] = nb
        child_b = na
    else:
        pa.children[ia], pb.children[ib] = nb, na

    if max_depth is not None and (_depth(child_a) > max_depth or _depth(child_b) > max_depth):
        return copy.deepcopy(parent_a), copy.deepcopy(parent_b)

    return child_a, child_b


def _collect_nodes_with_depth(root):
    """Return a list of (parent, child_index, node, depth); the root has parent None."""
    nodes = [(None, None, root, 0)]
    stack = [(root, 0)]
    while stack:
        parent, d = stack.pop()
        for i, child in enumerate(parent.children):
            nodes.append((parent, i, child, d + 1))
            stack.append((child, d + 1))
    return nodes


def mutate(child, p=MUTATE_PROB, max_depth=MAX_DEPTH, operands=OPERANDS, operators=OPERATORS):
    """Subtree replacement (headless chicken) mutation.

    With probability p, pick a random node and replace the subtree below it
    with a newly generated random subtree. The new subtree is grown so that
    the resulting tree never exceeds max_depth (root = depth 0).
    Returns the (possibly new) root, so always use the return value.
    """
    if random.random() >= p:
        return child

    parent, idx, _, node_depth = random.choice(_collect_nodes_with_depth(child))

    # the new subtree starts at node_depth and may reach at most max_depth
    start = min(node_depth, max_depth)
    new_subtree = _random_tree(start, max_depth, operands, operators, "grow")

    if parent is None:                 # the cutpoint was the root
        return new_subtree
    parent.children[idx] = new_subtree
    return child
