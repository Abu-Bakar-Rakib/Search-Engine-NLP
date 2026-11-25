# -*- coding: utf-8 -*-
"""Rule-family ablation (roadmap 3): switch each legacy rule family off, one at a time, and measure judged nDCG.

One engine load; the switches (`eng.off`, RULES_OFF) are flipped between runs, so all runs see the same catalog and queries.
  collect : run every labeled query per config, save top-10 ids  -> benchmarks/results/ablation_top10.json
            and write the (query, product) pairs nobody has judged yet -> benchmarks/qrels/batch_0.json
  score   : score the saved runs with the judged labels (totals only; tuning and holdout reported separately)

Run: .venv/bin/python benchmarks/ablate_rules.py collect [--only fam1,fam2]
     (judge new pairs: benchmarks/judge_pool.py merge, after the judge wrote batch_0.out.json)
     .venv/bin/python benchmarks/ablate_rules.py score
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from relevance_eval import K, catalog_rel_count, load_qrels, ndcg_graded, settings_for  # noqa: E402

FAMILIES = ["budget", "model", "apparel", "catspecial", "phoneacc", "typomap"]  # remaining after roadmap 3 (ear_anc, feature, everyday, affmap, cooling deleted)
OUT = os.path.join(HERE, "results", "ablation_top10.json")
SETS = {"tuning": ("live", "live_extra"), "holdout": ("holdout",)}


def queries():
    out = []
    for kind, files in SETS.items():
        for f in files:
            for ln in open(os.path.join(HERE, "queries", f"{f}.jsonl"), encoding="utf-8"):
                if ln.strip():
                    d = json.loads(ln)
                    d["set"] = kind
                    out.append(d)
    return out


def collect(a):
    from lite_search import build_engine
    eng = build_engine(settings_for("live"))
    qs = queries()
    configs = {"baseline": set(), **{f: {f} for f in (a.only.split(",") if a.only else FAMILIES)}}
    if not a.only:
        configs["all_off"] = set(FAMILIES)
    runs = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    for name, off in configs.items():
        eng.off = off
        eng._nlp.off = off
        eng._nlp._parse_cache.clear()
        eng._intent_score_cache.clear()
        runs[name] = {qd["q"]: [str(r["id"]) for r in eng.search(qd["q"], k=K, use_cache=False)["results"][:K]] for qd in qs}
        print("done", name, flush=True)
    json.dump(runs, open(OUT, "w", encoding="utf-8"))
    # unjudged pairs -> one batch for the judge
    judged = load_qrels()
    df = eng.cat.df.set_index(eng.cat.df["id"].astype(str))
    need = {}
    for run in runs.values():
        for q, ids in run.items():
            for i in ids:
                if (q, i) not in judged:
                    need.setdefault(q, set()).add(i)
    qd_by_q = {qd["q"]: qd for qd in qs}
    items = [{"q": q, "lang": qd_by_q[q].get("lang", "en"),
              "price_limit": {k: qd_by_q[q][k] for k in ("min_price", "max_price") if qd_by_q[q].get(k)},
              "items": [{"id": i, "title": str(df.at[i, "title"])[:160], "category": str(df.at[i, "category"])[:60],
                         "price": float(df.at[i, "price"] or 0)} for i in sorted(ids)]} for q, ids in need.items()]
    os.makedirs(os.path.join(HERE, "qrels"), exist_ok=True)
    n = max(1, a.batches)
    for b in range(n):
        json.dump(items[b::n], open(os.path.join(HERE, "qrels", f"batch_{b}.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"{len(items)} queries, {sum(len(x['items']) for x in items)} new pairs to judge in {n} batches")


def score(a):
    runs = json.load(open(OUT, encoding="utf-8"))
    qrels = load_qrels()
    qs = queries()
    ideal_n = {}
    from lite_search import Settings
    import pandas as pd  # noqa: F401
    from lite_search.catalog import load_dataframe
    df = load_dataframe(settings_for("live"))
    for qd in qs:
        ideal_n[qd["q"]] = max(catalog_rel_count(df, qd, "live"), sum(g > 0 for (q, _), g in qrels.items() if q == qd["q"]))
    rows = []
    for name, run in runs.items():
        res = {}
        for kind in SETS:
            vals, unj, tot = [], 0, 0
            for qd in qs:
                if qd["set"] != kind or ideal_n[qd["q"]] == 0:
                    continue
                grades = [qrels.get((qd["q"], i)) for i in run[qd["q"]]]
                unj += sum(g is None for g in grades)
                tot += len(grades)
                vals.append(ndcg_graded([g or 0 for g in grades], [2] * min(K, ideal_n[qd["q"]])))
            res[kind] = (round(sum(vals) / len(vals), 3), round(100 * unj / max(1, tot), 1))
        rows.append((name, res))
    base = dict(rows)["baseline"]
    print(f"{'config':12s} {'tuning':>8s} {'Δ':>7s} {'holdout':>8s} {'Δ':>7s}   unjudged% (tuning/holdout)")
    for name, r in rows:
        print(f"{name:12s} {r['tuning'][0]:8.3f} {r['tuning'][0]-base['tuning'][0]:+7.3f} {r['holdout'][0]:8.3f} "
              f"{r['holdout'][0]-base['holdout'][0]:+7.3f}   {r['tuning'][1]}/{r['holdout'][1]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "score"])
    ap.add_argument("--only", default=None)
    ap.add_argument("--batches", type=int, default=3)
    a = ap.parse_args()
    collect(a) if a.cmd == "collect" else score(a)
