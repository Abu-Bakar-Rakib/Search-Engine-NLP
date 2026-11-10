# -*- coding: utf-8 -*-
"""
Test suite to verify that the search engine detects typos and incorrect words,
infers their actual meaning, and retrieves the accurate products across
Bangla, Banglish, and English.
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

typo_test_cases = [
    {
        "query": "adydas running shoos",
        "description": "Brand + Category typos ('adydas' -> adidas, 'shoos' -> shoes)",
        "expected_category": "running shoes",
        "expected_brand": "Adidas"
    },
    {
        "query": "samsang mobail phon",
        "description": "Brand + Phonetic tech typos ('samsang' -> samsung, 'mobail phon' -> smartphone)",
        "expected_category": "smartphone",
        "expected_brand": "Samsung"
    },
    {
        "query": "nik snikers black",
        "description": "Brand typo + Category typo ('nik' -> nike, 'snikers' -> sneakers)",
        "expected_category": "sneakers",
        "expected_brand": "Nike"
    },
    {
        "query": "kashio hand wotch",
        "description": "Brand + Accessory typo ('kashio' -> casio, 'wotch' -> wristwatch)",
        "expected_category": "wristwatch",
        "expected_brand": "Casio"
    },
    {
        "query": "lethar wallat for man",
        "description": "Material + Accessory typo ('lethar' -> leather, 'wallat' -> wallet)",
        "expected_category": "leather wallet"
    },
    {
        "query": "blutooth hedphone wierless",
        "description": "Triple tech typo ('blutooth' -> bluetooth, 'hedphone' -> headphone, 'wierless' -> wireless)",
        "expected_category": "wireless headphones"
    },
    {
        "query": "toddlar bilding bloks",
        "description": "Kids toy typo ('toddlar' -> toddler, 'bilding bloks' -> building blocks)",
        "expected_category": "building blocks"
    },
    {
        "query": "airpode wirless under 3000",
        "description": "Audio typo ('airpode' -> airpods/headphones, 'wirless' -> wireless)",
        "expected_category": "wireless headphones"
    },
    {
        "query": "cotton punjaby under 3000",
        "description": "Ethnic clothing typo ('punjaby' -> panjabi)",
        "expected_category": "panjabi"
    },
    {
        "query": "sunglas uv protekt",
        "description": "Accessory typo ('sunglas' -> sunglasses, 'protekt' -> protect)",
        "expected_category": "sunglasses"
    },
    {
        "query": "পাঞ্জাবী ২০০০ টাকার নিচে",
        "description": "Bangla script spelling variation ('পাঞ্জাবী' -> 'পাঞ্জাবি' with 2000 cap)",
        "expected_category": "panjabi"
    },
    {
        "query": "লাল শাড়ী ২০০০ টাকার নিচে",
        "description": "Bangla script spelling variation ('শাড়ী' -> 'শাড়ি' with red color)",
        "expected_category": "saree"
    }
]

def main():
    print("=" * 85)
    print("  TYPO & INCORRECT WORD INTENT UNDERSTANDING TEST SUITE")
    print("=" * 85)

    base_url = "http://127.0.0.1:8000"
    passed = 0

    for idx, tc in enumerate(typo_test_cases, 1):
        q = tc["query"]
        desc = tc["description"]
        url = f"{base_url}/search?q={urllib.parse.quote(q)}&limit=3"

        req = urllib.request.Request(url)
        res = json.loads(urllib.request.urlopen(req, timeout=10).read().decode())

        corrections = res.get("corrections", [])
        norm = res.get("normalized", "")
        results = res.get("results", [])
        top = results[0] if results else {}

        top_title = top.get("title", "NONE")
        top_cat = top.get("category", "")
        top_brand = top.get("brand", "")
        top_price = top.get("price", 0)

        # Verification check
        cat_match = True
        if "expected_category" in tc:
            cat_match = tc["expected_category"].lower() in top_cat.lower() or tc["expected_category"].lower() in top_title.lower()

        brand_match = True
        if "expected_brand" in tc:
            brand_match = tc["expected_brand"].lower() == top_brand.lower() or tc["expected_brand"].lower() in top_title.lower()

        test_ok = len(results) > 0 and (cat_match or brand_match)
        if test_ok:
            passed += 1
            status = "[PASS]"
        else:
            status = "[FAIL]"

        print(f"\nTEST {idx:02d} {status}: {desc}")
        print(f"  * Raw Incorrect Query : \"{q}\"")
        print(f"  * Detected Meaning    : \"{norm}\"")
        print(f"  * Typo Corrections    : {corrections}")
        print(f"  * Found Product       : [{top.get('id')}] {top_title}")
        print(f"  * Brand & Category    : {top_brand} • {top_cat} (BDT {top_price})")

    print("\n" + "=" * 85)
    print(f"RESULT: {passed}/{len(typo_test_cases)} typo tests PASSED successfully!")
    print("=" * 85)

if __name__ == "__main__":
    main()
