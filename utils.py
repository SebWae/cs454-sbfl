from collections import Counter, defaultdict
import json
import numpy as np 
import os
import pandas as pd
import random

from config import DATA_DIR, SEED_SPLIT, SPLIT_FILE


def to_python(node):
    """Tree -> Python expression using the variable names expected by best.py."""
    if not node.children:
        return "1.0" if node.value == "1" else node.value
    a = [to_python(c) for c in node.children]
    v = node.value
    if v == "add":  return f"({a[0]} + {a[1]})"
    if v == "sub":  return f"({a[0]} - {a[1]})"
    if v == "mul":  return f"({a[0]} * {a[1]})"
    if v == "div":  return f"div({a[0]}, {a[1]})"
    if v == "sqrt": return f"(abs({a[0]}) ** 0.5)"


def best_of(pop, fits):
    i = max(range(len(pop)), key=lambda i: (fits[i]["acc1"], -fits[i]["wef"]))
    return pop[i], fits[i]


def make_split():
    if os.path.exists(SPLIT_FILE):
        with open(SPLIT_FILE) as f:
            d = json.load(f)
        return d["train"], d["val"]
    ids = [f[:-5] for f in os.listdir("./bug_data") if f.endswith(".json")]
    print(len(ids), Counter(b.split("-")[0] for b in ids))
    by_project = defaultdict(list)
    for b in sorted(ids):
        by_project[b.split("-")[0]].append(b)
    rng = random.Random(SEED_SPLIT)
    train, val = [], []
    for proj, lst in by_project.items():
        rng.shuffle(lst)
        k = round(0.7 * len(lst))
        train += lst[:k]
        val += lst[k:]
    with open(SPLIT_FILE, "w") as f:
        json.dump({"train": train, "val": val}, f)
    return train, val


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
