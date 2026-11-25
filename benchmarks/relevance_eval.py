# -*- coding: utf-8 -*-
"""Relevance benchmark: labeled queries -> P@10, Hit@1, nDCG@10, zero-result rate, latency.

Each query line (JSONL) in benchmarks/queries/<dataset>.jsonl:
  {"q": "...", "lang": "en|bn|banglish", "type": "typo|price|brand|...",
   "rel": "<regex, case-insensitive>", "not": "<regex, optional>",
   "min_price": 0, "max_price": 0}
A result is relevant when its judged text matches `rel`, does not match `not`, and is inside the price range.
Judged text = title (live dataset) or title + category (curated dataset, which has clean categories).

Run:  .venv/bin/python benchmarks/relevance_eval.py live --tag baseline
      .venv/bin/python benchmarks/relevance_eval.py curated --tag baseline
"""
import argparse
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from lite_search import Settings, build_engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
K = 10


def settings_for(dataset: str) -> Settings:
    s = Settings.from_env()
    s.redis_url, s.warm_file = None, None          # no cross-run / cross-catalog cache pollution
    if dataset == "curated":
        s.database_url, s.catalog_csv = None, "data/all_products.csv"
        s.index_dir = ".lite_search_index_curated"  # separate: index save drops vectors not in the catalog
    return s


def judged_text(row, dataset):
    return f"{row['title']} || {row['category']}" if dataset == "curated" else str(row["title"])


def is_rel(row, qd, dataset):
    t = judged_text(row, dataset)
    if not re.search(qd["rel"], t, re.I):
        return False
    if qd.get("not") and re.search(qd["not"], t, re.I):
        return False
    p = float(row.get("price") or 0)
    if qd.get("min_price") and p < qd["min_price"]:
        return False
    if qd.get("max_price") and p > qd["max_price"]:
        return False
    return True


def catalog_rel_count(df, qd, dataset):
    texts = (df["title"] + " || " + df["category"]) if dataset == "curated" else df["title"]
    m = texts.str.contains(qd["rel"], case=False, regex=True)
    if qd.get("not"):
        m &= ~texts.str.contains(qd["not"], case=False, regex=True)
    if qd.get("min_price"):
        m &= df["price"] >= qd["min_price"]
    if qd.get("max_price"):
        m &= df["price"] <= qd["max_price"]
    return int(m.sum())


def ndcg(rels, n_rel):
    dcg = sum(r / math.log2(i + 2) for i, r in enumerate(rels))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(n_rel, K)))
    return dcg / ideal if ideal else 0.0


def summarize(rows):
    n = len(rows) or 1
    lat = sorted(r["ms"] for r in rows) or [0]
    return {
        "queries": len(rows),
        "P@10": round(sum(r["p10"] for r in rows) / n, 3),
        "Hit@1": round(sum(r["hit1"] for r in rows) / n, 3),
        "nDCG@10": round(sum(r["ndcg"] for r in rows) / n, 3),
        "Zero%": round(100 * sum(r["n_results"] == 0 for r in rows) / n, 1),
        "lat_p50_ms": round(lat[len(lat) // 2], 1),
        "lat_p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["live", "curated"])
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--sets", default=None, help="comma list of query files, default: <dataset>")
    ap.add_argument("--dense-only", action="store_true", help="rank by embedding cosine only (measures the model, not the engine)")
    a = ap.parse_args()

    queries = []
    for name in (a.sets or a.dataset).split(","):
        with open(os.path.join(HERE, "queries", f"{name}.jsonl"), encoding="utf-8") as f:
            queries += [json.loads(ln) for ln in f if ln.strip() and not ln.startswith("//")]
    if a.sets:
        a.tag = f"{a.tag}__{a.sets.replace(',', '+')}"
    if a.dense_only:
        a.tag += "__dense"

    t0 = time.perf_counter()
    eng = build_engine(settings_for(a.dataset))
    df = eng.cat.df
    print(f"[{a.dataset}] {eng.cat.n} products, engine ready in {time.perf_counter() - t0:.1f}s")
    eng.search("warm up", k=K, use_cache=False)

    rows, skipped = [], []
    for qd in queries:
        n_rel = catalog_rel_count(df, qd, a.dataset)
        if n_rel == 0:
            skipped.append(qd["q"])
            continue
        t = time.perf_counter()
        if a.dense_only:
            top = (eng.cat.emb @ eng.qemb.encode_query(qd["q"])).argsort()[::-1][:K]
            out = {"results": [df.iloc[i].to_dict() for i in top]}
        else:
            out = eng.search(qd["q"], k=K, use_cache=False)
        ms = (time.perf_counter() - t) * 1000
        res = out.get("results", [])[:K]
        rels = [1 if is_rel(r, qd, a.dataset) else 0 for r in res]
        rows.append({
            "q": qd["q"], "lang": qd.get("lang", "en"), "type": qd.get("type", "plain"),
            "n_rel_catalog": n_rel, "n_results": len(res), "ms": round(ms, 1),
            "p10": sum(rels) / K, "hit1": rels[0] if rels else 0, "ndcg": round(ndcg(rels, n_rel), 3),
            "intent": out.get("intent", {}), "notes": out.get("notes", []),
            "top": [{"t": str(r["title"])[:80], "c": str(r["category"])[:40], "p": r["price"], "rel": x}
                    for r, x in zip(res, rels)],
        })
        print(f"  {rows[-1]['p10']:.1f} {rows[-1]['ndcg']:.2f} {len(res):2d} {ms:7.0f}ms  {qd['q']}")

    by = lambda key: {v: summarize([r for r in rows if r[key] == v]) for v in sorted({r[key] for r in rows})}
    report = {"dataset": a.dataset, "tag": a.tag, "products": eng.cat.n, "overall": summarize(rows),
              "by_lang": by("lang"), "by_type": by("type"), "skipped_no_relevant_in_catalog": skipped,
              "queries": rows}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{a.dataset}_{a.tag}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps({k: report[k] for k in ("overall", "by_lang", "by_type")}, indent=1))
    print("skipped (0 relevant in catalog):", skipped)
    print("saved", path)


if __name__ == "__main__":
    main()
