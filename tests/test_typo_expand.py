"""Typo/prefix expansion (Step 3). Run: .venv/bin/python tests/test_typo_expand.py"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lite_search.typo import TermExpander  # noqa: E402

DF = {"keyboard": 1151, "keybord": 1, "shampoo": 1198, "laptop": 1791, "cooker": 734, "cookie": 40,
      "cooler": 300, "rice": 900, "cover": 2000, "clover": 5, "হেডফোন": 30, "tv": 500}


def test_expand():
    e = TermExpander(DF)
    x = e.expand(["keybord"])
    assert x["keybord"] == 1.0 and x["keyboard"] == 0.7, x          # misspelling in catalog still corrected
    assert "laptop" in e.expand(["leptop"])                          # unknown word, 1 edit
    assert "shampoo" in e.expand(["shampo"])                         # missing letter
    assert "cooker" in e.expand(["rice", "coo"])                     # last-word prefix
    assert "cooker" not in e.expand(["coo", "rice"])                 # prefix only for the last word
    assert set(e.expand(["cover"])) == {"cover"}                     # common real word is never "fixed"
    assert "হেডফোন" in e.expand(["হেডফোণ"])                          # Bangla script, 1 edit
    assert set(e.expand(["tv"])) == {"tv"}                           # short words are left alone


def test_bangla_loanwords():
    df = {"shampoo": 1198, "charger": 1200, "earbud": 300, "earbuds": 900, "blender": 400, "refrigerator": 500,
          "robot": 200, "year": 3000, "hair": 800}
    e = TermExpander(df)
    assert "shampoo" in e.expand(["শ্যামপু"], translit=True)        # missing hasant spelling
    assert "charger" in e.expand(["চারজার"], translit=True)
    assert "earbud" in e.expand(["ইয়ারবাদ"], translit=True) and "robot" not in e.expand(["ইয়ারবাদ"], translit=True)
    assert "refrigerator" in e.expand(["রেফ্রিজারেটার"], translit=True)
    assert set(e.expand(["হেয়ার"], translit=True)) == {"হেয়ার"}  # one-consonant skeleton: no guess
    assert set(e.expand(["শ্যামপু"])) == {"শ্যামপু"}                # off unless asked


if __name__ == "__main__":
    test_bangla_loanwords()
    test_expand()
    print("ok - typo/prefix expansion")
