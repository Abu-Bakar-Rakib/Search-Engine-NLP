# -*- coding: utf-8 -*-
"""Compare two relevance_eval runs: metric deltas + queries that got better / worse.

Run: .venv/bin/python benchmarks/compare.py live baseline step1
Exit code 1 if the gate fails (nDCG@10 dropped by more than --tol), so CI / the skill can use it.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
METRICS = ["nDCG@10", "P@10", "Hit@1", "Zero%", "lat_p50_ms", "lat_p95_ms"]


def load(ds, tag):
    with open(os.path.join(HERE, "results", f"{ds}_{tag}.json"), encoding="utf-8") as f:
        return json.load(f)


def row(name, a, b):
    cells = []
    for m in METRICS:
        x, y = a.get(m, 0), b.get(m, 0)
        cells.append(f"{y:>7} ({y - x:+.3g})")
    return f"{name:<10}" + "".join(f"{c:>18}" for c in cells)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset")
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--tol", type=float, default=0.01, help="allowed nDCG@10 drop")
    a = ap.parse_args()
    o, n = load(a.dataset, a.old), load(a.dataset, a.new)

    print(f"{a.dataset}: {a.old} -> {a.new}   (value (delta))")
    print(f"{'':<10}" + "".join(f"{m:>18}" for m in METRICS))
    print(row("overall", o["overall"], n["overall"]))
    for grp in ("by_type", "by_lang"):
        for k in sorted(n[grp]):
            print(row(k, o[grp].get(k, {}), n[grp][k]))

    oq = {q["q"]: q for q in o["queries"]}
    moved = [(q["ndcg"] - oq[q["q"]]["ndcg"], q["q"], oq[q["q"]]["ndcg"], q["ndcg"])
             for q in n["queries"] if q["q"] in oq]
    for title, items in (("better", sorted(m for m in moved if m[0] > 0.05)[::-1]),
                         ("worse", sorted(m for m in moved if m[0] < -0.05))):
        print(f"\n{title} ({len(items)}):")
        for d, q, x, y in items:
            print(f"  {x:.2f} -> {y:.2f}  {q}")

    drop = o["overall"]["nDCG@10"] - n["overall"]["nDCG@10"]
    ok = drop <= a.tol
    print(f"\nGATE {'PASS' if ok else 'FAIL'}: nDCG@10 change {-drop:+.3f} (allowed drop {a.tol})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
