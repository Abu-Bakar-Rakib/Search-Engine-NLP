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


if __name__ == "__main__":
    test_expand()
    print("ok - typo/prefix expansion")
