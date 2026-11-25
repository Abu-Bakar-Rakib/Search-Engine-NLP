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

    def expand(self, tokens) -> Dict[str, float]:
        """tokens -> {term: weight}. Original words weigh 1.0; corrections and prefix matches less."""
        out: Dict[str, float] = {}
        for n, t in enumerate(tokens):
            out[t] = max(out.get(t, 0.0), 1.0)
            for w, dist in self.corrections(t):
                out[w] = max(out.get(w, 0.0), W_TYPO[dist])
            if n == len(tokens) - 1:
                for w in self.prefixes(t):
                    out[w] = max(out.get(w, 0.0), W_PREFIX)
        return out
