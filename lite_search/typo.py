"""Typo tolerance and partial-word (prefix) matching, built only from the catalog's own vocabulary.

Like Typesense: a query word also matches catalog words a small edit distance away, and the last query word
also matches words that start with it. Frequencies decide: a correction must be much more common than the
word typed, so real but rare words are not "fixed". Script-agnostic (English, Banglish, Bangla).
"""
import bisect
from collections import defaultdict
from typing import Dict

from .nlp import levenshtein_dist

MIN_DF = 3          # vocabulary = words in at least this many products
RATIO = 10          # a correction must be this many times more frequent than the typed word
MAX_FIX = 2         # corrections kept per word
PREFIX_MAX_DF = 20  # last word gets prefix matches only when it is rare/unknown (a fragment like "coo")
MAX_PREFIX = 5
W_TYPO = {1: 0.7, 2: 0.5}
W_PREFIX = 0.8
W_TRANSLIT = 0.7
MAX_TRANSLIT = 2
TRANSLIT_MAX_RATIO = 0.5  # edit distance / length between the transliteration and the catalog word

# ---- Bangla script -> Latin (phonetic, loanword-friendly) ----
_BN_CONS = {"ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng", "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "n",
            "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n", "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
            "প": "p", "ফ": "f", "ব": "b", "ভ": "v", "ম": "m", "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh",
            "স": "s", "হ": "h", "\u09dc": "r", "\u09dd": "r", "\u09df": "y", "ৎ": "t", "ং": "ng", "ঃ": "h"}
_BN_VOW = {"অ": "o", "আ": "a", "ই": "i", "ঈ": "i", "উ": "u", "ঊ": "u", "ঋ": "ri", "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
           "া": "a", "ি": "i", "ী": "i", "ু": "u", "ূ": "u", "ৃ": "ri", "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou"}


def has_bangla(t: str) -> bool:
    return any("\u0980" <= ch <= "\u09ff" for ch in t)


def translit(t: str) -> str:
    """Bangla script -> rough Latin spelling. The inherent vowel is dropped (loanwords rarely keep it);
    the y-phala after a consonant is dropped."""
    # য়/ড়/ঢ় are stored as base letter + nukta (NFC keeps them decomposed): fold to single code points first
    t = t.replace("\u09af\u09bc", "\u09df").replace("\u09a1\u09bc", "\u09dc").replace("\u09a2\u09bc", "\u09dd")
    out, chars = [], list(t.replace("\u09cd\u09af", "").replace("\u09bc", ""))
    for ch in chars:
        out.append(_BN_CONS.get(ch) or _BN_VOW.get(ch) or "")
    return "".join(out)


def phon_key(latin: str) -> str:
    """Consonant skeleton shared by English spellings and transliterated Bangla: shampoo / শ্যাম্পু -> SMP."""
    w = latin.lower()
    for a, b in (("chh", "C"), ("ch", "C"), ("sh", "S"), ("ph", "F"), ("kh", "K"), ("gh", "J"), ("th", "T"), ("dh", "T"),
                 ("bh", "P"), ("jh", "J"), ("ng", "N"), ("ck", "K")):
        w = w.replace(a, b)
    m = {"k": "K", "c": "K", "q": "K", "g": "J", "j": "J", "z": "J", "s": "S", "x": "KS", "t": "T", "d": "T",
         "p": "P", "b": "P", "v": "P", "f": "F", "r": "R", "l": "L", "m": "M", "n": "N"}
    key = "".join(ch if ch.isupper() else m.get(ch, "") for ch in w)
    return "".join(c for i, c in enumerate(key) if i == 0 or c != key[i - 1])


def _deletes(w: str, k: int) -> set:
    out, frontier = {w}, {w}
    for _ in range(k):
        frontier = {x[:i] + x[i + 1:] for x in frontier for i in range(len(x))}
        out |= frontier
    return out


def max_edits(w: str) -> int:
    return 0 if len(w) < 4 else 1 if len(w) < 7 else 2   # Typesense defaults: min_len_1typo=4, min_len_2typo=7


class TermExpander:
    def __init__(self, doc_freq: Dict[str, int]):
        self.df = doc_freq
        vocab = [w for w, n in doc_freq.items() if n >= MIN_DF and not w.isdigit() and 3 <= len(w) <= 20]
        self.index = defaultdict(list)          # SymSpell-style: deleted form -> words
        for w in vocab:
            for d in _deletes(w, max_edits(w)):
                self.index[d].append(w)
        self.sorted_vocab = sorted(vocab)
        self.keys = defaultdict(list)           # phonetic key -> Latin catalog words (for Bangla loanwords)
        for w in vocab:
            if w.isascii() and w.isalpha() and len(w) >= 3:
                self.keys[phon_key(w)].append(w)

    def transliterations(self, t: str):
        if not has_bangla(t) or self.df.get(t, 0) >= PREFIX_MAX_DF:
            return []
        lat = translit(t)
        key = phon_key(lat)
        if len(lat) < 3 or len(key) < 2:  # one-consonant skeletons match too many words
            return []
        hits = []
        for w in self.keys.get(key, ()):
            d = levenshtein_dist(lat, w)
            if d <= TRANSLIT_MAX_RATIO * max(len(lat), len(w)):
                hits.append((d, -self.df[w], w))
        return [w for _, _, w in sorted(hits)[:MAX_TRANSLIT]]

    def corrections(self, t: str):
        k = max_edits(t)
        if k == 0 or t.isdigit():
            return []
        floor = max(MIN_DF, RATIO * self.df.get(t, 0))
        seen = {}
        for d in _deletes(t, k):
            for w in self.index.get(d, ()):
                if w != t and w not in seen and self.df[w] >= floor:
                    dist = levenshtein_dist(t, w)
                    if dist <= k:
                        seen[w] = dist
        return sorted(seen.items(), key=lambda x: (x[1], -self.df[x[0]]))[:MAX_FIX]

    def prefixes(self, t: str):
        if len(t) < 3 or t.isdigit() or self.df.get(t, 0) >= PREFIX_MAX_DF:
            return []
        i = bisect.bisect_left(self.sorted_vocab, t)
        hits = []
        while i < len(self.sorted_vocab) and self.sorted_vocab[i].startswith(t):
            if self.sorted_vocab[i] != t:
                hits.append(self.sorted_vocab[i])
            i += 1
        return sorted(hits, key=lambda w: -self.df[w])[:MAX_PREFIX]

    def expand(self, tokens, translit: bool = False) -> Dict[str, float]:
        """tokens -> {term: weight}. Original words weigh 1.0; corrections and prefix matches less."""
        out: Dict[str, float] = {}
        for n, t in enumerate(tokens):
            out[t] = max(out.get(t, 0.0), 1.0)
            for w, dist in self.corrections(t):
                out[w] = max(out.get(w, 0.0), W_TYPO[dist])
            if translit:
                for w in self.transliterations(t):
                    out[w] = max(out.get(w, 0.0), W_TRANSLIT)
            if n == len(tokens) - 1:
                for w in self.prefixes(t):
                    out[w] = max(out.get(w, 0.0), W_PREFIX)
        return out
