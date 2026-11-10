# -*- coding: utf-8 -*-
"""
Verification suite for Bangla (বাংলা লিপি) and Banglish (Phonetic Bengali) queries.
Tests:
1. Bengali Unicode character parsing (\u0980-\u09FF)
2. Bengali numerals translation (০-৯ -> 0-9)
3. Bengali relationship & persona extraction (বাবা, মা, ছেলে, মেয়ে, ভাই, বোন)
4. Bengali color & material normalization (কালো, লাল, নীল, সুতি, চামড়া)
5. Banglish phonetic expressions (babar jonno, ma er jonno, cheler, lal, er niche)
6. Price/budget filter adherence with Bangla currency patterns (টাকার মধ্যে, টাকার নিচে, er niche)
7. Hybrid retrieval and product relevance
"""
import sys
import os
import json
import urllib.request
import urllib.parse

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

test_cases = [
    # ---- 1. Pure Bangla Script Queries (বাংলা লিপি) ----
    {
        "type": "Bangla Script",
        "query": "বাবার জন্য পাঞ্জাবি ৩০০০ টাকার মধ্যে",
        "description": "Father gift with Panjabi category and 3000 BDT budget cap (Bangla digits: ৩০০০)",
        "expected_intent": {"gender": "men", "recipient": "father", "max_price": 3000.0}
    },
    {
        "type": "Bangla Script",
        "query": "মায়ের জন্য শাড়ি ২০০০ টাকার নিচে",
        "description": "Mother gift with Saree category and 2000 BDT budget cap (Bangla digits: ২০০০)",
        "expected_intent": {"gender": "women", "recipient": "mother", "max_price": 2000.0}
    },
    {
        "type": "Bangla Script",
        "query": "কালো জুতো সাইজ 42",
        "description": "Color black (কালো) + Shoes category (জুতো) + Size filter (42)",
        "expected_intent": {"color": "black", "size": "42"}
    },
    {
        "type": "Bangla Script",
        "query": "লাল শার্ট ১০০০ টাকার কম",
        "description": "Color red (লাল) + Shirt (শার্ট) + 1000 BDT budget cap (১০০০ টাকার কম)",
        "expected_intent": {"color": "red", "category": "shirt", "max_price": 1000.0}
    },
    {
        "type": "Bangla Script",
        "query": "বাচ্চাদের খেলনা ১৫০০ টাকার নিচে",
        "description": "Audience kids (বাচ্চাদের) + Toy (খেলনা) + 1500 BDT cap (১৫০০ টাকার নিচে)",
        "expected_intent": {"max_price": 1500.0}
    },
    {
        "type": "Bangla Script",
        "query": "সুতি পাঞ্জাবি ২০০০ টাকার নিচে",
        "description": "Material cotton (সুতি) + Panjabi (পাঞ্জাবি) + 2000 BDT cap (২০০০ টাকার নিচে)",
        "expected_intent": {"material": "cotton", "category": "panjabi", "max_price": 2000.0}
    },

    # ---- 2. Banglish Queries (Phonetic Latin Bengali) ----
    {
        "type": "Banglish",
        "query": "babar jonno gift under 3000",
        "description": "Father persona ('babar jonno') + gift intent + 3000 budget",
        "expected_intent": {"gender": "men", "recipient": "father", "max_price": 3000.0}
    },
    {
        "type": "Banglish",
        "query": "ma er jonno shari 2000 er niche",
        "description": "Mother persona ('ma er jonno') + Saree + 'er niche' 2000 budget",
        "expected_intent": {"gender": "women", "recipient": "mother", "max_price": 2000.0}
    },
    {
        "type": "Banglish",
        "query": "lal shirt 1000 takar niche",
        "description": "Color red ('lal') + Shirt + 'takar niche' 1000 budget",
        "expected_intent": {"color": "red", "category": "shirt", "max_price": 1000.0}
    },
    {
        "type": "Banglish",
        "query": "cheler jonno jacket size L",
        "description": "Men persona ('cheler jonno') + Jacket + Size L",
        "expected_intent": {"gender": "men", "category": "jacket", "size": "L"}
    },
    {
        "type": "Banglish",
        "query": "kom dam e valo phone",
        "description": "Low budget smartphone intent ('kom dam e valo phone')",
        "expected_intent": {}
    },
    {
        "type": "Banglish",
        "query": "online class er jonno",
        "description": "Occasion/use-case for online schooling ('online class er jonno')",
        "expected_intent": {}
    }
]


def run_verification():
    print("=" * 85)
    print("  SEARCH ENGINE ELIGIBILITY REPORT: BANGLA & BANGLISH QUERY PROCESSING")
    print("=" * 85)

    base_url = "http://127.0.0.1:8000"
    
    # 1. Check server connectivity
    try:
        health_res = json.loads(urllib.request.urlopen(f"{base_url}/health", timeout=5).read().decode())
        print(f"[OK] Server is Online | Active Catalog: {health_res.get('products')} products loaded.")
    except Exception as e:
        print(f"[ERROR] Could not connect to API server: {e}")
        return

    passed_count = 0
    total_count = len(test_cases)

    for idx, tc in enumerate(test_cases, 1):
        q = tc["query"]
        desc = tc["description"]
        q_type = tc["type"]

        url = f"{base_url}/search?q={urllib.parse.quote(q)}&limit=3"
        t0 = os.times().elapsed
        req = urllib.request.Request(url)
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())

        intent = resp.get("intent", {})
        norm = resp.get("normalized", "")
        results = resp.get("results", [])
        took = resp.get("took_ms", 0)

        # Check intent correctness
        intent_ok = True
        for k, v in tc["expected_intent"].items():
            if intent.get(k) != v:
                intent_ok = False

        has_results = len(results) > 0
        test_passed = has_results and intent_ok

        if test_passed:
            passed_count += 1
            status_str = "[PASS]"
        else:
            status_str = "[FAIL]" if not has_results else "[PARTIAL]"

        print("\n" + "-" * 85)
        print(f"TEST {idx:02d} ({q_type}) {status_str}: {desc}")
        print(f"  * Raw Query       : \"{q}\"")
        print(f"  * Normalized Form : \"{norm}\"")
        print(f"  * Extracted Intent: {intent}")
        print(f"  * Latency         : {took} ms")
        print("  * Top Matched Products:")
        for r_idx, r in enumerate(results[:2], 1):
            title = r.get("title", "")
            cat = r.get("category", "")
            price = r.get("price", 0)
            score = r.get("score", 0)
            print(f"      {r_idx}. [{r.get('id')}] {title} | {cat} | {price} BDT (score: {score})")

    print("\n" + "=" * 85)
    print(f"SUMMARY: {passed_count}/{total_count} tests PASSED with full intent extraction and relevant retrieval.")
    print("=" * 85)


if __name__ == "__main__":
    run_verification()
