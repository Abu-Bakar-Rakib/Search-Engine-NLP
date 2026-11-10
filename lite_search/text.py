import re
import unicodedata

import numpy as np

BANGLA_STOP = {
    "এর", "র", "জন্য", "কে", "এবং", "ও", "বা", "তে", "হলো", "থেকে", "দিয়ে",
    "দিয়ে", "চাই", "দরকার", "দেবো", "দেবে", "দিন", "টাকা", "টাকার", "টা", "টি"
}
STOP = set("a an the for of and or in on with to is it my me i that who this at by from under over below above".split()) | BANGLA_STOP
MISSING = {"", "unknown", "nan", "none", "null", "n/a"}

_TOKEN_RE = re.compile(r"[a-z0-9\u0980-\u09FF]+")
_ALNUM_RE = re.compile(r"[^a-z0-9\u0980-\u09FF]")


def nfc(s) -> str:
    return unicodedata.normalize("NFC", str(s))


def toks(s: str):
    s = nfc(s).lower().replace("h&m", "hm")          # keep "H&M" one token (m = size M!)
    words = []
    for w in _TOKEN_RE.findall(s):
        if w in STOP:
            continue
        words.append(w)
        # Add singular stem for plural English words so 'shoes' matches 'shoe', 'headphones' matches 'headphone'
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3 and not re.search(r"[\u0980-\u09FF]", w):
            words.append(w[:-1])
    return words


def alnum(s) -> str:
    return _ALNUM_RE.sub("", nfc(s).lower())


def word_in(value, text: str) -> bool:
    v = nfc(value).strip().lower()
    t = nfc(text)
    return bool(v) and re.search(rf"(?<![\w\u0980-\u09FF]){re.escape(v)}(?![\w\u0980-\u09FF])", t) is not None


def cat_norm(s) -> str:
    words = []
    for w in _TOKEN_RE.findall(nfc(s).lower()):
        # Only strip trailing English plural 's' if word does not contain Bengali characters
        # and does not end in 'ss' (e.g. wireless, dress, sunglasses, fitness)
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3 and not re.search(r"[\u0980-\u09FF]", w):
            words.append(w[:-1])
        else:
            words.append(w)
    return " ".join(words)


def minmax(x):
    x = np.asarray(x, dtype="float64")
    if x.size == 0:
        return x
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def is_missing(v) -> bool:
    return str(v).strip().lower() in MISSING

