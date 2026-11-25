"""Price direction parsing (R5). Run: .venv/bin/python tests/test_price_parse.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_search.nlp import QueryParser  # noqa: E402

CASES = [
    ("iphone 13 pro max 128gb", None, None),          # "max 128gb" is a spec, not a cap
    ("saree above 2000", 2000, None),
    ("2000 taka upore saree", 2000, None),
    ("2000 টাকার উপরে শাড়ি", 2000, None),
    ("shoes under 5k", None, 5000),
    ("phone 15000 taka", None, 15000),
    ("between 1000 and 3000 tk headphone", 1000, 3000),
    ("1000 theke 2000 taka panjabi", 1000, 2000),
    ("500 takar niche t-shirt", None, 500),
    ("power bank 10000mah", None, None),
    ("laptop 60000", None, 60000),
]


def test_price():
    qp, bad = QueryParser(), []
    for q, lo, hi in CASES:
        mn, mx, _ = qp.extract_price(qp.normalize_text(q)[0])
        if (mn, mx) != (lo, hi):
            bad.append(f"{q!r}: got {(mn, mx)}, want {(lo, hi)}")
    assert not bad, "\n".join(bad)


def test_price_survives_lexicon():
    """A word list that translates currency words (টাকা -> money) must not break Bangla price phrases."""
    qp = QueryParser()
    qp.dp.merge_bangla_words({"টাকা": "money", "উপরে": "top", "niche": "bottom"})
    for q, lo, hi in [("৫০০০ টাকার উপরে ঘড়ি", 5000, None), ("500 takar niche t-shirt", None, 500)]:
        it = qp.parse(q)
        assert (it.min_price, it.max_price) == (lo, hi), (q, it.min_price, it.max_price)


if __name__ == "__main__":
    test_price()
    test_price_survives_lexicon()
    print(f"ok - {len(CASES)} price cases")
