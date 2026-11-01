import os
from collections import defaultdict

import pandas as pd

from best import div
from config import CSV_PATH, DATA_DIR
from utils import load_bug            


def score_table(bug):
    """One row per method: score, the spectrum that produced it, rank, buggy flag."""
    rows = []
    for m, tuples in bug["methods"].items():
        s, t = max((sbfl(*t), t) for t in tuples)       # best statement in the method
        rows.append({"method": m, "score": s,
                     "ep": t[0], "ef": t[1], "np": t[2], "nf": t[3],
                     "is_buggy": m in bug["buggy"]})
    df = pd.DataFrame(rows)
    # conservative rank: 1 + number of methods strictly ahead + tied others
    df["rank"] = df["score"].apply(lambda s: int((df["score"] >= s).sum()))
    return df.sort_values(["score"], ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    # pick the best candidate from the CSV
    df = pd.read_csv(CSV_PATH)
    best = df.sort_values(["val_acc1", "val_wef"], ascending=[False, True]).iloc[0]
    print(f"chosen run: {best['run']}  (val_acc1={best['val_acc1']}, val_wef={best['val_wef']})")
    print(f"tree:       {best['formula']}")
    print(f"python:     {best['python_expr']}")

    # converting the chosen candidate formula into a function
    code = compile(best["python_expr"], "<formula>", "eval")

    def sbfl(ep, ef, np, nf):          
        return eval(code, {"__builtins__": {}, "div": div, "abs": abs},
                    {"ep": ep, "ef": ef, "np": np, "nf": nf})

    # evaluate on all bugs
    ids = sorted(f[:-5] for f in os.listdir(DATA_DIR) if f.endswith(".json"))
    ranks = defaultdict(list)

    for b in ids:
        bug = load_bug(b)
        scores = {m: max(sbfl(*t) for t in tuples)           # max aggregation
                for m, tuples in bug["methods"].items()}
        # conservative ties; best-ranked buggy method
        r = min(sum(s >= scores[m] for s in scores.values()) for m in bug["buggy"])
        ranks[b.split("-")[0]].append(r)

    all_r = [r for rs in ranks.values() for r in rs]
    top1 = sum(r == 1 for r in all_r)
    print(f"\nbugs: {len(all_r)}")
    print(f"acc@1: {top1} / {len(all_r)} ({100 * top1 / len(all_r):.1f}%)")

    # wef per project 
    for p in ["Math", "Lang", "Time", "Chart"]:
        if ranks[p]:
            print(f"{p:6s} wef = {sum(ranks[p]) / len(ranks[p]):.3f}  (n={len(ranks[p])})")

    # more detailed assessment
    tables, summary = {}, []
    for b in ids:
        bug = load_bug(b)
        tables[b] = score_table(bug)
        r = int(tables[b].loc[tables[b]["is_buggy"], "rank"].min())   # best-ranked buggy method
        summary.append({"bug": b, "project": b.split("-")[0], "rank": r,
                        "n_methods": len(tables[b])})
    summary = pd.DataFrame(summary)

    print(f"acc@1: {(summary['rank'] == 1).sum()} / {len(summary)}")
    print(summary.groupby("project")["rank"].agg(wef="mean", acc1=lambda r: (r == 1).sum(), n="count"))

    pd.set_option("display.max_colwidth", 70, "display.width", 200)

    # 1. Top of the ranking for one bug: the buggy method should be near the top
    bug_prefix = "Math-9"
    print(tables[f"{bug_prefix}"].head(10))

    # 2. Only the buggy method(s) of that bug
    t = tables[f"{bug_prefix}"]
    print(t[t["is_buggy"]])

    # 3. A bug the formula got wrong, to see why
    worst = summary.sort_values("rank", ascending=False).head(5)
    print(worst)
    print(tables[worst.iloc[0]["bug"]].head(10))

    # 4. Distribution of ranks
    print(summary["rank"].describe())
    print((summary["rank"] <= 5).mean())   # fraction in the top 5