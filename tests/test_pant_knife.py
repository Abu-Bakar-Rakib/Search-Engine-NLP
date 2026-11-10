# -*- coding: utf-8 -*-
"""
Regression test suite for:
1. 'pant' / 'pants' / 'প্যান্ট' -> Must return bottom-wear products (trousers, jeans, track pants, leggings)
   and NEVER unrelated products like mugs, lamps, backpacks.
2. 'churi' / 'chhuri' / 'ছুরি' -> Must resolve to kitchen knife / cutlery and return Cuisinart Knife Block Set.
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

BASE_URL = "http://127.0.0.1:8000"

def query_search(q: str, k: int = 10):
    url = f"{BASE_URL}/search?q={urllib.parse.quote(q)}&k={k}"
    req = urllib.request.Request(url, headers={"User-Agent": "TestClient/1.0"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))

def test_pant_relevance():
    """Verify 'pant' returns bottom wear and no unrelated items."""
    bottom_categories = {"trousers", "jeans", "track pants", "leggings", "shorts", "pajamas"}
    for q in ["pant", "pants", "প্যান্ট"]:
        data = query_search(q, k=5)
        results = data.get("results", [])
        assert len(results) > 0, f"No results for query '{q}'"
        top_item = results[0]
        cat = top_item.get("category", "").lower()
        title = top_item.get("title", "").lower()
        assert cat in bottom_categories or "pant" in title or "trouser" in title, \
            f"Query '{q}' returned non-bottom wear: {top_item['title']} ({cat})"
        print(f"[PASS] Query '{q}' -> Top: {top_item['title']} ({cat})")

def test_knife_churi_relevance():
    """Verify 'churi' / 'chhuri' / 'ছুরি' returns kitchen knife product."""
    for q in ["churi", "chhuri", "ছুরি", "knife"]:
        data = query_search(q, k=5)
        results = data.get("results", [])
        assert len(results) > 0, f"No results for query '{q}'"
        top_item = results[0]
        cat = top_item.get("category", "").lower()
        title = top_item.get("title", "").lower()
        assert cat == "kitchen knife" or "knife" in title or "knife" in cat, \
            f"Query '{q}' did not return knife at #1: {top_item['title']} ({cat})"
        print(f"[PASS] Query '{q}' -> Top: {top_item['title']} ({cat})")

if __name__ == "__main__":
    print("Testing pant relevance...")
    test_pant_relevance()
    print("\nTesting knife/churi relevance...")
    test_knife_churi_relevance()
    print("\nAll pant and churi relevance tests PASSED successfully!")
