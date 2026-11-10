# -*- coding: utf-8 -*-
"""Comprehensive test of the merged/upgraded multilingual search engine."""
import os
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from lite_search import build_engine
from lite_search.config import Settings


cfg = Settings.from_env()
print("Building engine...")
engine = build_engine(cfg)
print(f"Engine ready! {engine.info()['products']} products loaded.\n")

tests = [
    # Bengali Script Queries
    ("বাবার জন্য উপহার ৩০০০ টাকার নিচে", "Bengali script: Father gift under 3000"),
    ("মায়ের জন্য শাড়ি ২০০০ টাকার মধ্যে", "Bengali script: Saree for mother under 2000"),
    ("কালো জুতো সাইজ 42", "Bengali script: Black shoes size 42"),
    ("লাল শার্ট ১০০০ টাকার কম", "Bengali script: Red shirt under 1000"),
    ("সুতি পাঞ্জাবি ২০০০ টাকার নিচে", "Bengali script: Cotton panjabi under 2000"),

    # Banglish Phonetic Queries
    ("baba ke gift debo under 3000", "Banglish: Father gift under 3000"),
    ("lal shirt 1000 takar niche", "Banglish: Red shirt under 1000"),
    ("cheler jonno jacket size L", "Banglish: Men jacket size L"),
    ("panjabi under 2000", "Banglish: Panjabi under 2000"),

    # English Intent Queries
    ("gift for mother under 2000", "English: Gift for mother under 2000"),
    ("black nike running shoes under 5000", "English: Nike running shoes under 5000"),
    ("laptop for student under 50000", "English: Student laptop under 50000"),
    ("gym shoes for men", "English: Gym shoes for men"),
]

for query, label in tests:
    r = engine.search(query, k=3)
    print(f"=== {label} ===")
    print(f"Query: '{query}'")
    print(f"Normalized: '{r.get('normalized', '')}'")
    print(f"Intent: {r.get('intent', {})}")
    print(f"Notes: {r.get('notes', [])}")
    print(f"Source: {r.get('source', '')}")
    print(f"Took: {r.get('took_ms', 0)} ms (cached: {r.get('cached', False)})")
    for p in r.get("results", []):
        print(f"  {p['rank']}. {p['title']} [{p['category']}] - BDT {p['price']} (Score: {p['score']}, Cos: {p['cosine']}, BM25: {p['bm25']})")
    print()
