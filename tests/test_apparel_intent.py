# -*- coding: utf-8 -*-
"""
Test Suite: Apparel & Hypernym Intent Disambiguation
Validates queries like 'cheleder dress', 'meyeder dress', 'bacchader dress',
'cheleder jama', 'ছেলেদের পোশাক', ensuring proper gender-specific clothing retrieval
and zero opposite-gender false positives.
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

base_url = "http://127.0.0.1:8000"

test_cases = [
    {
        "query": "cheleder dress",
        "description": "Men's clothing / dress: should return men's apparel (shirt, panjabi, jeans), 0% women dresses",
        "expected_gender": "men",
        "forbidden_genders": ["women", "girls"],
        "expected_categories": ["shirt", "panjabi", "t-shirt", "jeans", "trousers", "jacket"]
    },
    {
        "query": "cheleder jama",
        "description": "Banglish apparel 'jama' for men: should return men's clothing",
        "expected_gender": "men",
        "forbidden_genders": ["women", "girls"],
        "expected_categories": ["panjabi", "shirt", "t-shirt", "jeans", "trousers"]
    },
    {
        "query": "dress for men",
        "description": "English 'dress for men': should return men's clothing / shirts / panjabi",
        "expected_gender": "men",
        "forbidden_genders": ["women", "girls"],
        "expected_categories": ["shirt", "panjabi", "t-shirt", "jeans"]
    },
    {
        "query": "ছেলেদের পোশাক",
        "description": "Native Bangla 'cheleder poshak': should return men's clothing",
        "expected_gender": "men",
        "forbidden_genders": ["women", "girls"],
        "expected_categories": ["shirt", "panjabi", "t-shirt", "jeans"]
    },
    {
        "query": "meyeder dress",
        "description": "Women's apparel 'meyeder dress': should return women's clothing (saree, kurti, dress, salwar kameez)",
        "expected_gender": "women",
        "forbidden_genders": ["men", "boys"],
        "expected_categories": ["saree", "kurti", "salwar kameez", "dress"]
    },
    {
        "query": "bacchader dress",
        "description": "Kids' apparel 'bacchader dress': should return kids' clothing, 0% adult women dresses",
        "expected_gender": ["kids", "girls", "boys"],
        "forbidden_categories_for_adults": ["women"],
        "expected_categories": ["dress", "kurti", "panjabi", "t-shirt", "pajamas"]
    },
    {
        "query": "summer dress",
        "description": "Specific dress query 'summer dress': should return women's dresses",
        "expected_gender": "women",
        "forbidden_genders": ["men", "boys"],
        "expected_categories": ["dress"]
    }
]


def main():
    print("=" * 85)
    print("  APPAREL & GENDER HYPERNYM INTENT TEST SUITE")
    print("=" * 85)

    passed_count = 0
    total = len(test_cases)

    for i, tc in enumerate(test_cases, 1):
        q = tc["query"]
        desc = tc["description"]
        url = f"{base_url}/search?q={urllib.parse.quote(q)}&k=5"
        
        req = urllib.request.Request(url)
        resp = json.loads(urllib.request.urlopen(req, timeout=10).read().decode("utf-8"))
        results = resp.get("results", [])
        intent = resp.get("intent", {})

        # Validation checks
        has_results = len(results) > 0
        no_forbidden = True
        has_expected_cat = False

        for r in results:
            g = str(r.get("gender", "")).lower()
            cat = str(r.get("category", "")).lower()
            
            if "forbidden_genders" in tc and g in tc["forbidden_genders"]:
                no_forbidden = False
            if "forbidden_categories_for_adults" in tc and g in tc["forbidden_categories_for_adults"]:
                no_forbidden = False
            if cat in tc.get("expected_categories", []):
                has_expected_cat = True

        test_passed = has_results and no_forbidden and has_expected_cat
        status = "[PASS]" if test_passed else "[FAIL]"
        if test_passed:
            passed_count += 1

        print(f"\nTEST {i:02d} {status}: {desc}")
        print(f"  * Query    : \"{q}\"")
        print(f"  * Intent   : {intent}")
        print(f"  * Products :")
        for r in results[:3]:
            print(f"      - {r.get('title')} | cat: {r.get('category')} | gender: {r.get('gender')}")

    print("\n" + "=" * 85)
    print(f"RESULT: {passed_count}/{total} tests PASSED!")
    print("=" * 85)
    if passed_count != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
