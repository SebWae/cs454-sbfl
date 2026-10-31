from collections import Counter, defaultdict
import json 
import os
import random

from config import SEED_SPLIT, SPLIT_FILE


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