import re

import numpy as np

STOP = set("a an the for of and or in on with to is it my me i that who this at by from under over below above".split())
MISSING = {"", "unknown", "nan", "none", "null", "n/a"}


def toks(s: str):
    s = s.lower().replace("h&m", "hm")          # keep "H&M" one token (m = size M!)
    return [w for w in re.findall(r"[a-z0-9]+", s) if w not in STOP]


def alnum(s) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def word_in(value, text: str) -> bool:
    v = str(value).strip().lower()
    return bool(v) and re.search(rf"(?<!\w){re.escape(v)}(?!\w)", text) is not None


def cat_norm(s) -> str:
    return " ".join(w[:-1] if w.endswith("s") and len(w) > 3 else w
                    for w in re.findall(r"[a-z0-9]+", str(s).lower()))


def minmax(x):
    x = np.asarray(x, dtype="float64")
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def is_missing(v) -> bool:
    return str(v).strip().lower() in MISSING
