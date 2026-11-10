"""Query understanding without any model: price / size / gender / category / brand / colour / material,
Bangla-Banglish normalisation and product-group detection. Built from the catalog's own vocabulary."""
import json
import re
from typing import Dict, Optional

import numpy as np

from . import lexicon as L
from .text import STOP, alnum, cat_norm, word_in

SCHEMA_KEYS = ["category", "brand", "size", "color", "material", "gender", "requirement", "min_price", "max_price"]
SIGNAL_KEYS = ["category", "brand", "size", "color", "material", "gender", "min_price", "max_price"]

# ---------------- price ----------------
_NUM = r"(?:[$₹৳£€]\s*)?(\d+(?:\.\d+)?)\s*(k)?\b"
_CUR = r"(?:\s*(?:usd|dollars?|taka|tk|bdt|rs|inr|rupees?|eur|euros?|gbp|pounds?))?"
_LOW = r"(?:under|below|less than|upto|up to|within|maximum|max|cheaper than|not more than|<)"
_HIGH = r"(?:over|above|more than|at least|minimum|min|starting from|>)"
_BETWEEN = rf"(?:between|from)\s+{_NUM}{_CUR}\s*(?:and|to|-)\s*{_NUM}{_CUR}"


def _val(num, k):
    v = float(num) * (1000 if k else 1)
    return int(v) if v.is_integer() else v


def extract_price(query):
    q = query.lower().replace(",", "")
    m = re.search(_BETWEEN, q)
    if m:
        a, b = _val(m.group(1), m.group(2)), _val(m.group(3), m.group(4))
        return min(a, b), max(a, b)
    m = re.search(rf"{_LOW}\s*{_NUM}", q)
    if m:
        return None, _val(m.group(1), m.group(2))
    m = re.search(rf"{_HIGH}\s*{_NUM}", q)
    if m:
        return _val(m.group(1), m.group(2)), None
    return None, None


def strip_price_phrase(query):
    q = re.sub(_BETWEEN, " ", query, flags=re.I)
    q = re.sub(rf"(?:{_LOW}|{_HIGH})\s*{_NUM}{_CUR}", " ", q, flags=re.I)
    return re.sub(r"\s+", " ", q).strip()


# ---------------- size ----------------
STD_SIZES = {"xs", "s", "m", "l", "xl", "xxl", "xxxl"}
_SZ = r"(xxxl|xxl|xl|xs|s|m|l|\d{1,2})"


def find_size(q):
    q = q.lower()
    for pat in (rf"\bsize\s*[:\-]?\s*{_SZ}\b", rf"\b{_SZ}\s+size\b"):
        m = re.search(pat, q)
        if m:
            return m.group(1).upper()
    for t in q.split():
        t = t.strip(".,;:!?()")
        if t in STD_SIZES:
            return t.upper()
    return None


def strip_size_phrase(query):
    q = query.lower()
    q = re.sub(rf"\bsize\s*[:\-]?\s*{_SZ}\b", " ", q)
    q = re.sub(rf"\b{_SZ}\s+size\b", " ", q)
    q = " ".join(t for t in q.split() if t.strip(".,;:!?()") not in STD_SIZES)
    return re.sub(r"\s+", " ", q).strip()


def retrieval_query(q):
    """Text used for BM25 / embeddings: no price phrase, no size words."""
    a = strip_price_phrase(q) or q
    return strip_size_phrase(a) or a


# ---------------- LLM output validation (never trust the model blindly) ----------------
def parse_json(text) -> Dict:
    try:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        data = json.loads(m.group(0)) if m else {}
    except Exception:
        data = {}
    return {k: data.get(k) for k in SCHEMA_KEYS}


def validate_intent(query, d) -> Dict:
    q = query.lower()
    out = {k: d.get(k) for k in SCHEMA_KEYS}
    if out["brand"] and alnum(out["brand"]) not in alnum(query):
        out["brand"] = None
    for key in ("color", "material", "size"):
        if out[key] and not word_in(out[key], q):
            out[key] = None
    if out["size"]:
        out["size"] = str(out["size"]).upper()
    g = str(out["gender"]).lower() if out["gender"] else None
    out["gender"] = g if (g in L.GENDER_CUES and any(word_in(c, q) for c in L.GENDER_CUES[g])) else None
    if out["requirement"]:
        parts = [p.strip() for p in str(out["requirement"]).split(",")]
        parts = [p for p in parts if p and not re.search(r"\d|under|below|over|above|between|price|cheap", p.lower())]
        out["requirement"] = ", ".join(parts) or None
    lo, hi = extract_price(query)
    digits = re.sub(r"[\s,]", "", q)
    for name, rule_val in (("min_price", lo), ("max_price", hi)):
        if rule_val is not None:
            out[name] = rule_val
        else:
            try:
                v = out[name]
                ok = v is not None and re.search(rf"(?<!\d){int(float(v))}(?!\d)", digits) is not None
            except (ValueError, TypeError):
                ok = False
            out[name] = out[name] if ok else None
    return out


def has_signal(p) -> bool:
    return any(p.get(k) is not None for k in SIGNAL_KEYS)


# Generic words / shop noise that must never be hijacked as a brand or exact leaf category
_NOISY_BRANDS = {
    "no brand", "non-brand", "non brand", "generic", "unknown", "other", "none", "na", "n/a",
    "smart", "power", "watch", "laptop", "mobile", "phone", "smartphone", "tablet", "cooler", "heater",
    "conditioner", "adapter", "cable", "earphone", "earphones", "speaker", "speakers", "blender",
    "kettle", "cooker", "iron", "monitor", "printer", "router", "keyboard", "camera", "battery",
    "light", "lamp", "glass", "bottle", "cup", "mug", "bag", "shoe", "shoes", "sandal", "sandals",
    "sneaker", "sneakers", "pant", "pants", "jeans", "kurti", "hoodie", "jacket", "dress", "set",
    "combo", "pack", "collection", "original", "official", "imported", "new", "best", "good",
    "quality", "premium", "pure", "fresh", "natural", "soft", "face", "skin", "hair", "body",
    "eye", "lip", "men", "women", "kids", "baby", "boy", "girl", "unisex", "black", "white",
    "red", "blue", "green", "yellow", "pink", "gold", "silver", "brown", "grey", "fashion",
    "style", "classic", "super", "pro", "max", "ultra", "mini", "plus", "air", "home", "life",
    "care", "beauty", "health", "shop", "store", "gadgets", "gadget", "electronics", "accessories",
    "fan", "charger", "cover", "case", "mouse", "headphone", "earbuds", "shirt", "t-shirt",
    "saree", "panjabi", "rice", "cream", "wash", "serum", "oil", "gel", "soap", "shampoo",
    "lipstick", "sunscreen", "trimmer", "toner", "cleanser", "lotion", "powder", "mask",
    "pencil", "notebook", "book", "pen", "toy", "gift",
}

_NOISY_CATS = {
    "featured", "other", "others", "gadget", "gadgets", "skin", "face", "hair", "body",
    "men", "women", "kids", "baby", "electronics", "electronics store", "electronics & tech",
    "fashion", "fashion 5", "men's fashion", "women's fashion", "men’s fashion & style",
    "phones & tablets", "phones & gadgets", "audio & gadgets", "computers & it", "appliances",
    "beauty & personal care", "grocery", "grocery & food", "health & wellness", "health & fitness",
    "hardware & tools", "books & stationery", "home, kitchen & furniture", "jewellery & accessories",
    "sports, auto & outdoor", "baby, kids & toys", "accessories", "makeup",
}


# ---------------- rule parser + normaliser, built from the catalog ----------------
class RuleParser:
    def __init__(self, catalog, aliases=None, groups=None):
        self.cat = catalog
        self.aliases = {**L.ALIASES, **(aliases or {})}
        self.groups = groups or L.GROUPS
        by_len = lambda x: -len(x)
        raw_brands = catalog.vocab_values("brand")
        self.brands = sorted([b for b in raw_brands if b.lower() not in _NOISY_BRANDS and len(b.strip()) >= 2], key=by_len)
        brand_alnum = {alnum(b) for b in self.brands}
        raw_cats = catalog.vocab_values("category")
        self.cats = sorted(
            [c for c in raw_cats if c.lower() not in _NOISY_CATS and alnum(c) not in brand_alnum],
            key=lambda c: -len(cat_norm(c)),
        )
        self.colors = sorted(catalog.vocab_values("color"), key=by_len)
        self.mats = sorted(catalog.vocab_values("material"), key=by_len)
        self.known = set(catalog.doc_freq) | STOP | {"size", "between", "upto", "within", "maximum", "max", "minimum",
                                                    "min", "least", "more", "less", "than", "cheaper", "tk", "taka", "bdt"}
        self.group_masks = {}
        for g, (_, cats) in self.groups.items():
            m = catalog.category_mask(cats)
            if m.any():
                self.group_masks[g] = m

    # ---- normalisation: Banglish -> English ----
    def alias_pass(self, query: str) -> str:
        q = " " + query.lower() + " "
        for pat, rep in L.PRICE_RULES:
            q = re.sub(pat, rep, q)
        for a, b in L.PHRASES:
            q = q.replace(" " + a + " ", " " + b + " ")
        return " ".join(w for w in (self.aliases.get(t, t) for t in q.split()) if w)

    def needs_translation(self, q: str) -> bool:
        ts = re.findall(r"[a-z0-9]+", q)
        unknown = [w for w in ts if w.isalpha() and len(w) > 2 and w not in self.known]
        return bool(unknown) and not any(w in self.cat.doc_freq for w in ts)

    # ---- rule-based intent ----
    def parse(self, query: str) -> Dict:
        # Strip negated clauses before extracting positive brand/color/category signals
        q_pos = re.sub(
            r"\b(?:not|except|without|no)\s+[a-z0-9\u0980-\u09FF\s\-]+?(?=\s+(?:under|below|above|between|for|size|with|and|,|\.|$)|$)",
            " ",
            query.lower(),
        )
        q_pos = re.sub(r"[a-z0-9\u0980-\u09FF]+\s+(?:ছাড়|ছাড়া|ব্যতীত|বিনা)", " ", q_pos)
        q = re.sub(r"\s+", " ", q_pos).strip()
        qtoks = {alnum(w) for w in q.split()}
        p = {k: None for k in SCHEMA_KEYS}
        qn = " " + cat_norm(q) + " "
        p["category"] = next((c for c in self.cats if " " + cat_norm(c) + " " in qn), None)
        if re.search(r"\biphones?\b", q) and not any(w in q for w in ("case", "cover", "protector", "charger", "cable", "holder")):
            p["category"] = "smartphone"
            p["brand"] = "Apple"
        elif not p["category"] and re.search(r"\b(?:running\s+(?:.*?\s+)?shoes?|shoes?\s+(?:.*?\s+)?(?:running|jogging)|trainers?\s+(?:for\s+)?jogging)\b", q):
            p["category"] = "running shoes"
        p["gender"] = next((g for g in ("women", "men", "girls", "boys", "unisex")
                            if any(word_in(c, q) for c in L.GENDER_CUES[g])), None)
        p["size"] = find_size(q)
        if p["gender"] in ("boys", "girls") and p["size"] in ("S", "M", "L", "XL", "XXL", "38", "40", "42", "44"):
            p["gender"] = "men" if p["gender"] == "boys" else "women"

        # Disambiguate generic apparel words (e.g. 'dress' meaning clothing for men)
        if p["category"] == "dress" and (p["gender"] in ("men", "boys") or any(word_in(w, q) for w in ("chele", "cheler", "cheleder", "purush", "purusher", "men", "mens", "boy", "boys"))):
            p["category"] = None
            if not p["gender"]:
                p["gender"] = "men"
        p["color"] = next((c for c in self.colors if word_in(c, q)), None)
        p["material"] = next((m for m in self.mats if word_in(m, q)), None)
        # Skip single-token brand matching if the token is part of a multi-word product term (e.g. 'power bank', 'smart watch')
        q_masked = re.sub(r"\b(?:power\s+bank|smart\s+watch|hand\s+watch|face\s+wash)\b", " ", q)
        q_masked_toks = {alnum(w) for w in q_masked.split()}
        p["brand"] = next((b for b in self.brands if word_in(b, q_masked) or (" " not in b and alnum(b) in q_masked_toks)), None)
        if p["brand"] and p["color"] and alnum(p["brand"]) == alnum(p["color"]):
            p["brand"] = None
        p["min_price"], p["max_price"] = extract_price(query)
        return p

    # ---- product groups ("shoes" -> all footwear categories) ----
    def detect_groups(self, q: str):
        ql = q.lower()
        t = set(re.findall(r"[a-z0-9\u0980-\u09FF]+", ql))
        names = []
        for g in self.group_masks:
            triggers = self.groups[g][0]
            single_hit = bool(t & triggers)
            phrase_hit = any(
                (" " in ph or "-" in ph) and re.search(rf"(?<![\w\u0980-\u09FF]){re.escape(ph)}(?![\w\u0980-\u09FF])", ql)
                for ph in triggers
            )
            if single_hit or phrase_hit:
                names.append(g)
        # Accessory disambiguation: 'mobile er charger' / 'phone cover' should not drag in laptops/phones under 'electronics'
        if "phone_accessories" in names and "electronics" in names:
            if not any(w in t for w in ("laptop", "laptops", "computer", "tablet", "monitor", "keyboard", "mouse", "printer", "router", "webcam")):
                names.remove("electronics")
        # 'air conditioner' contains the token 'conditioner' — don't confuse it with hair conditioner
        if "haircare" in names and "air conditioner" in ql:
            if not any(w in t for w in ("hair", "shampoo", "scalp", "dandruff")):
                names.remove("haircare")
        if not names:
            return None, []
        mask = np.zeros(self.cat.n, dtype=bool)
        for g in names:
            mask |= self.group_masks[g]
        return mask, names
