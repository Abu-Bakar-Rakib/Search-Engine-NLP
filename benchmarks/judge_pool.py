# -*- coding: utf-8 -*-
"""Judged relevance labels (qrels), built by pooling — the standard way search engines are evaluated.

1. `pool`  : run every labeled query through Light Search (top 10), add ids from extra pool files (e.g. Typesense's
             results/pool.jsonl), drop (query, product) pairs already judged, and write batches to judge.
2. A judge (LLM sub-agent or a person) grades each item 0/1/2 and writes `<batch>.out.json`.
3. `merge` : append the graded pairs to benchmarks/qrels/qrels.jsonl.

Grades: 2 = exactly what the shopper asked for, 1 = acceptable / related, 0 = wrong.
Later engine versions only need their NEW (query, product) pairs judged: run `pool` again.

Run: .venv/bin/python benchmarks/judge_pool.py pool --extra ../Typesense/results/pool.jsonl --batches 4
     .venv/bin/python benchmarks/judge_pool.py merge
"""
import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
QRELS_DIR = os.path.join(HERE, "qrels")
QRELS = os.path.join(QRELS_DIR, "qrels.jsonl")
SETS = ("live", "live_extra", "holdout")


def load_qrels():
    out = {}
    if os.path.exists(QRELS):
        for ln in open(QRELS, encoding="utf-8"):
            if ln.strip():
                d = json.loads(ln)
                out[(d["q"], str(d["id"]))] = d["grade"]
    return out


def queries():
    qs = []
    for s in SETS:
        qs += [json.loads(ln) for ln in open(os.path.join(HERE, "queries", f"{s}.jsonl"), encoding="utf-8") if ln.strip()]
    return qs


def pool(a):
    sys.path.insert(0, os.path.join(HERE, ".."))
    sys.path.insert(0, HERE)
    from relevance_eval import settings_for
    from lite_search import build_engine
    eng = build_engine(settings_for("live"))
    df = eng.cat.df.set_index(eng.cat.df["id"].astype(str))
    extra = {}
    for p in a.extra or []:
        for ln in open(p, encoding="utf-8"):
            d = json.loads(ln)
            extra.setdefault(d["q"], []).extend(str(i) for i in d["ids"])
    judged = load_qrels()
    items = []
    for qd in queries():
        ids = [str(r["id"]) for r in eng.search(qd["q"], k=10, use_cache=False)["results"]] + extra.get(qd["q"], [])
        todo = [i for i in dict.fromkeys(ids) if (qd["q"], i) not in judged and i in df.index]
        if todo:
            items.append({"q": qd["q"], "lang": qd.get("lang", "en"),
                          "price_limit": {k: qd[k] for k in ("min_price", "max_price") if qd.get(k)},
                          "items": [{"id": i, "title": str(df.at[i, "title"])[:160], "category": str(df.at[i, "category"])[:60],
                                     "price": float(df.at[i, "price"] or 0)} for i in todo]})
    os.makedirs(QRELS_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(QRELS_DIR, "batch_*.json")):
        os.remove(f)
    n = max(1, a.batches)
    for b in range(n):
        part = items[b::n]
        with open(os.path.join(QRELS_DIR, f"batch_{b}.json"), "w", encoding="utf-8") as f:
            json.dump(part, f, ensure_ascii=False, indent=1)
    print(f"{len(items)} queries, {sum(len(x['items']) for x in items)} pairs to judge in {n} batches -> {QRELS_DIR}")


def merge(a):
    judged = load_qrels()
    added = 0
    with open(QRELS, "a", encoding="utf-8") as f:
        for out in sorted(glob.glob(os.path.join(QRELS_DIR, "batch_*.out.json"))):
            src = json.load(open(out.replace(".out.json", ".json"), encoding="utf-8"))
            want = {(x["q"], str(i["id"])) for x in src for i in x["items"]}
            for d in json.load(open(out, encoding="utf-8")):
                for j in d["judgments"]:
                    key = (d["q"], str(j["id"]))
                    if key in want and key not in judged and j.get("grade") in (0, 1, 2):
                        f.write(json.dumps({"q": key[0], "id": key[1], "grade": j["grade"]}, ensure_ascii=False) + "\n")
                        judged[key] = j["grade"]
                        added += 1
            missing = want - set(judged)
            if missing:
                print(f"  {os.path.basename(out)}: {len(missing)} pairs not judged")
    print(f"added {added}; qrels now {len(judged)} pairs")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["pool", "merge"])
    ap.add_argument("--extra", nargs="*", help="pool files with {q, ids} lines")
    ap.add_argument("--batches", type=int, default=4)
    a = ap.parse_args()
    pool(a) if a.cmd == "pool" else merge(a)
