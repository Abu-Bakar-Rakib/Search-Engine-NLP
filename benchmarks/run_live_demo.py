# -*- coding: utf-8 -*-
"""
Live Search Engine Test Demo
Tests multilingual (English, Bangla, Banglish), intent-aware hybrid retrieval on lite_search.
"""
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lite_search import build_engine, Settings

def run_tests():
    print("=" * 85)
    print("  LIVE SEARCH ENGINE TEST SUITE: MULTILINGUAL, INTENT-AWARE HYBRID RETRIEVAL")
    print("=" * 85)

    print("\n[INIT] Initializing unified lite_search engine...")
    cfg = Settings.from_env()
    engine = build_engine(cfg)
    print(f"  -> Engine Ready! {engine.cat.n} products loaded.\n")

    test_queries = [
        ("gift for my father under 3000", "English Gift Recommendation with Budget Cap"),
        ("বাবার জন্য গিফট ৩০০০ টাকার মধ্যে", "Bangla Script: Father's Gift with Bangla Digits & Currency"),
        ("বাচ্চাদের খেলনা ১৫০০ টাকার নিচে", "Bangla Script: Kids Toy with 1500 BDT Budget Cap"),
        ("panjabi 2000 er niche", "Banglish: Ethnic Apparel with 'er niche' Price Filter"),
        ("online class er jonno", "Banglish: Use-Case Query (Online Classes -> Webcams, Laptops, Headphones)"),
        ("comfortable black nike running shoe under 5000", "Multi-Attribute: Brand + Color + Category + Budget"),
        ("keep my coffee hot on a long drive", "Semantic Use-Case: Insulated Travel Drinkware"),
        ("kom dam e valo phone", "Banglish: 'Low Price Good Smartphone'"),
        ("xyzabc 123456789 unknown junk query", "Out-of-Domain: No Exact Match Calibration Test")
    ]

    for idx, (query_text, desc) in enumerate(test_queries, 1):
        print("=" * 85)
        print(f"TEST {idx}/9: {desc}")
        print(f"QUERY: \"{query_text}\"")
        print("-" * 85)

        t0 = time.perf_counter()
        res = engine.search(query_text, k=3)
        duration = round((time.perf_counter() - t0) * 1000, 2)

        intent = res.get("intent", {})
        print("PARSED INTENT:")
        print(f"  * Normalized Query : \"{res.get('normalized', '')}\"")
        if intent.get("recipient"):
            print(f"  * Recipient        : {intent['recipient']} (Gender: {intent.get('gender', 'unisex')})")
        if intent.get("max_price"):
            print(f"  * Budget Limit     : <= {intent['max_price']} BDT")
        if intent.get("brand"):
            print(f"  * Brand            : {intent['brand']}")
        if intent.get("color"):
            print(f"  * Color            : {intent['color']}")
        if intent.get("material"):
            print(f"  * Material         : {intent['material']}")
        if intent.get("intents"):
            print(f"  * Context/Use-Case : {intent['intents']}")
        if intent.get("affinities"):
            print(f"  * Inferred Affin.  : {intent['affinities'][:5]}")

        print(f"\nSEARCH PERFORMANCE:")
        print(f"  * Latency: {res.get('took_ms', duration)} ms | Cache Hit: {res.get('cached', False)}")
        notes = res.get("notes", [])
        if notes:
            print(f"  * Notes  : {'; '.join(notes)}")

        print(f"\nTOP RETRIEVED PRODUCTS:")
        results = res.get("results", [])
        if results:
            for r_i, p in enumerate(results, 1):
                print(f"  #{r_i} [{p.get('id', p.get('product_id'))}] {p.get('title')}")
                print(f"      Category: {p.get('category')} | Brand: {p.get('brand')} | Price: BDT {p.get('price')}")
                print(f"      Scores  : Total={p.get('score', 0):.2f} | BM25={p.get('bm25', 0):.2f} | Cosine={p.get('cosine', 0):.3f} | Match={p.get('match', '')}")
                if p.get('decision_explanation'):
                    print(f"      Decision: {p['decision_explanation']}")
        else:
            print("      No products matched constraints.")
        print()

    print("=" * 85)
    print("  ALL LIVE TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 85)

if __name__ == "__main__":
    run_tests()
