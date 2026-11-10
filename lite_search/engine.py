"""Search orchestration:  cache -> normalise -> RULES+NLP -> (LLM) -> RRF(BM25+Dense+Intent) -> rerank -> cache.

Upgraded with advanced Bangla/Banglish NLP, structured intent (persona, affinities, hard/soft constraints),
Reciprocal Rank Fusion (RRF), and neural cross-encoder reranking from the Desktop SEARCH ENGINE project.
"""
import json
import logging
import math
import os
import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import lexicon as L
from .cache import Cache, sha
from .catalog import Catalog, load_dataframe
from .config import Settings
from .embedder import CachedQueryEmbedder, make_embedder
from .llm import make_llm
from .nlp import DomainPack, QueryParser, StructuredIntent
from .parser import RuleParser, has_signal, retrieval_query, validate_intent
from .text import STOP, alnum, cat_norm, minmax, nfc, toks, word_in

log = logging.getLogger("lite_search")
ATTR_W = {"category": 2.5, "brand": 16.0, "color": 8.0, "material": 8.0, "gender": 0.6}
BOOST = {"category": 0.15, "brand": 0.20, "color": 0.10, "material": 0.10}
RELAX_ORDER = ["size", "gender", "price"]
OUT_COLS = ["id", "title", "category", "brand", "color", "material", "gender", "sizes", "price", "image_url"]

# RRF / retrieval weights (from domain pack retrieval_config)
_RRF_K = 60
_BM25_W = 0.35
_DENSE_W = 0.65
_INTENT_W = 0.30
_CANDIDATE_POOL = 50

# Affinity boost weights
_AFF_BOOST = 12.0
_BRAND_BOOST = 14.0
_COLOR_BOOST = 7.0
_MATERIAL_BOOST = 6.0
_CATEGORY_BOOST = 10.0


def _qkey(q: str) -> str:
    return " ".join(q.lower().split())


class SearchEngine:
    def __init__(self, catalog: Catalog, embedder, cache: Cache, llm, settings: Settings):
        self.cat, self.cache, self.llm, self.s = catalog, cache, llm, settings
        self.qemb = CachedQueryEmbedder(embedder, cache, settings.ttl_embedding)
        aliases, groups = {}, None
        if settings.lexicon_file:
            with open(settings.lexicon_file, encoding="utf-8") as f:
                lex = json.load(f)
            aliases = lex.get("aliases", {})
            groups = {g: (set(v[0]), v[1]) for g, v in lex.get("groups", {}).items()} or None
            if groups:
                groups = {**L.GROUPS, **groups}
        self.parser = RuleParser(catalog, aliases, groups)
        self._ce = None
        # Advanced NLP parser (Bangla/Banglish + persona + intent)
        self._nlp = QueryParser(DomainPack.default())
        self._nlp.set_catalog(catalog)
        # Intent compatibility score cache
        self._intent_score_cache: Dict[Tuple, np.ndarray] = {}

    # ------------------------------------------------------------ tier 0: normalise
    def _normalize(self, query: str):
        key = _qkey(query)
        hit = self.cache.get_json("norm", key)
        if hit:
            return hit["q"], hit["note"]
        q = self.parser.alias_pass(query)
        note = f"dictionary: '{query}' -> '{q}'" if q != key else ""
        if self.parser.needs_translation(q) and self.llm.available:
            t = self.llm.translate(query)
            if t and set(t.split()) & set(self.cat.doc_freq):
                q, note = t, f"LLM translation: '{query}' -> '{t}'"
        q = q.strip() or key or "product"
        self.cache.set_json("norm", key, {"q": q, "note": note}, self.s.ttl_intent)
        return q, note

    # ------------------------------------------------------------ tier 1 + 2: rules, then LLM
    def _intent(self, q_norm: str, grp_mask):
        key = sha(q_norm)
        hit = self.cache.get_json("intent", key)
        if hit:
            return hit["intent"], hit["source"] + "(cached)"
        intent, source, cacheable = self.parser.parse(q_norm), "rules", True
        if not has_signal(intent) and grp_mask is None and self.llm.available:
            raw = self.llm.parse(q_norm)
            if raw is None:
                cacheable = False                              # API hiccup: try again next time
            else:
                llm_intent = validate_intent(q_norm, raw)
                if has_signal(llm_intent) or llm_intent.get("requirement"):
                    intent, source = llm_intent, "llm"
        if not has_signal(intent) and grp_mask is None:
            source = "embedding"                               # nothing structured found: pure semantic match
        if cacheable:
            self.cache.set_json("intent", key, {"intent": intent, "source": source}, self.s.ttl_intent)
        return intent, source

    # ------------------------------------------------------------ scoring helpers
    def _mask(self, p, active):
        c = self.cat
        m = np.ones(c.n, dtype=bool)
        if "size" in active and p.get("size"):
            m &= c.size_mask(p["size"])
        if "gender" in active and p.get("gender"):
            if p["gender"] in ("kids", "baby", "children"):
                m &= c.kid_item | np.isin(c.gender, ["kids", "baby", "toddler", "boys", "girls", "unisex"])
            elif p["gender"] in ("men", "boys"):
                m &= np.isin(c.gender, ["men", "boys", "unisex"])
            elif p["gender"] in ("women", "girls"):
                m &= np.isin(c.gender, ["women", "girls", "unisex"])
            else:
                m &= (c.gender == p["gender"]) | (c.gender == "unisex")
        if "price" in active:
            if p.get("max_price") is not None:
                m &= c.price <= p["max_price"]
            if p.get("min_price") is not None:
                m &= c.price >= p["min_price"]
        return m

    def _boost(self, p):
        c, b = self.cat, np.zeros(self.cat.n)
        if p.get("category"):
            pc = cat_norm(p["category"]); pt = set(pc.split())
            vals = np.array([BOOST["category"] if pc == u else (BOOST["category"] / 2 if pt & set(u.split()) else 0.0)
                             for u in c.cat_uni])
            b += vals[c.cat_code]
        if p.get("brand"):
            b += BOOST["brand"] * (c.brand_n == alnum(p["brand"]))
        if p.get("color"):
            b += BOOST["color"] * (c.color_n == alnum(p["color"]))
        if p.get("material"):
            pm = alnum(p["material"])
            b += BOOST["material"] * np.array([(pm in m) or (m in pm) if m else False for m in c.mat_uni])[c.mat_code]
        return b

    def _attr_bonus(self, p):
        c, flags = self.cat, {}
        if p.get("color"):
            flags["color"] = c.color_n == alnum(p["color"])
        if p.get("brand"):
            flags["brand"] = c.brand_n == alnum(p["brand"])
        if p.get("material"):
            pm = alnum(p["material"])
            flags["material"] = np.array([(pm in m) or (m in pm) if m else False for m in c.mat_uni])[c.mat_code]
        if p.get("category"):
            flags["category"] = c.cat_n == cat_norm(p["category"])
        if p.get("gender"):
            flags["gender"] = c.gender == p["gender"]
        bonus = sum((ATTR_W[n] * f for n, f in flags.items()), np.zeros(c.n))
        return bonus, flags

    def _price_tier(self, p):
        lo, hi, s = p.get("min_price"), p.get("max_price"), self.s
        if s.hard_price or (lo is None and hi is None):
            return np.zeros(self.cat.n), None
        price = self.cat.price
        inside, over = np.ones(self.cat.n, dtype=bool), np.zeros(self.cat.n)
        if hi is not None:
            inside &= price <= hi; over = np.maximum(over, (price - hi) / max(hi, 1))
        if lo is not None:
            inside &= price >= lo; over = np.maximum(over, (lo - price) / max(lo, 1))
        near = 1.0 - np.minimum(over, 2.0) / 2.0
        return np.where(inside, s.price_tier, s.price_near * near), inside

    def _audience(self, q, p, intent=None):
        toks = set(q.lower().split())
        recipient = (intent.recipient if intent else None) or p.get("recipient")
        gender = (intent.target_gender if intent else None) or p.get("gender")
        if gender in ("boys", "girls") or (toks & L.KID_WORDS):
            return -self.s.kid_penalty * (~self.cat.kid_item)
        if gender in ("women", "men") or recipient in ("mother", "father", "wife", "husband"):
            return -self.s.kid_penalty * self.cat.kid_item
        # For general queries without explicit kid words, mildly penalize kids items so adult products rank higher
        return -15.0 * self.cat.kid_item

    def _rerank(self, q_clean: str, cand: List[int], top_n: int = 20) -> np.ndarray:
        """Neural Cross-Encoder reranking with Query-Document Pair Caching.
        Reranks the top candidates (up to top_n) with ms-marco-MiniLM on CPU,
        caching pairs to ensure sub-millisecond response on repeats."""
        if not cand:
            return np.zeros(0, dtype=np.float32)
        if self._ce is None:
            try:
                import torch
                torch.set_num_threads(min(4, os.cpu_count() or 4))
            except Exception:
                pass
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.s.ce_model, device="cpu")
            self._ce_pair_cache = {}

        eval_cand = cand[:top_n]
        scores = np.zeros(len(cand), dtype=np.float32)
        pairs_to_predict = []
        indices_to_predict = []

        for j, idx in enumerate(eval_cand):
            # Document snippet: first 200 chars (title, category, brand, key features)
            doc_snippet = self.cat.texts[idx][:200]
            pair_key = (q_clean, idx)
            if pair_key in self._ce_pair_cache:
                scores[j] = self._ce_pair_cache[pair_key]
            else:
                pairs_to_predict.append((q_clean, doc_snippet))
                indices_to_predict.append((j, pair_key))

        if pairs_to_predict:
            try:
                import torch
                with torch.inference_mode():
                    preds = self._ce.predict(pairs_to_predict, batch_size=32, show_progress_bar=False)
            except Exception:
                preds = self._ce.predict(pairs_to_predict, batch_size=32, show_progress_bar=False)
            for (j, pair_key), pred in zip(indices_to_predict, preds):
                score_val = float(pred)
                scores[j] = score_val
                if len(self._ce_pair_cache) < 15000:
                    self._ce_pair_cache[pair_key] = score_val

        # Beyond top_n, maintain baseline rank order
        if len(cand) > top_n:
            min_top = float(scores[:top_n].min()) if top_n > 0 else 0.0
            scores[top_n:] = min_top - 1.0

        return scores

    # ------------------------------------------------------------ RRF fusion helpers
    def _rrf_ranks(self, scores: np.ndarray, min_val: Optional[float] = 0.0) -> np.ndarray:
        """Convert scores to RRF rank contribution: 1 / (k + rank).
        Items with score <= min_val get 0.0 (they are not in the retrieved candidate set).
        """
        out = np.zeros_like(scores, dtype=np.float64)
        if min_val is not None:
            pos = np.where(scores > min_val)[0]
        else:
            pos = np.arange(len(scores))
        if len(pos) == 0:
            return out
        order = pos[np.argsort(-scores[pos])]
        ranks = np.arange(len(order))
        out[order] = 1.0 / (_RRF_K + ranks)
        return out

    def _intent_compat_scores(self, intent: StructuredIntent) -> np.ndarray:
        """Score products by affinity/persona/intent category compatibility."""
        c = self.cat
        if not intent.affinities:
            return np.zeros(c.n, dtype=np.float32)
        cache_key = (tuple(intent.affinities), intent.recipient)
        if cache_key in self._intent_score_cache:
            return self._intent_score_cache[cache_key].copy()

        scores = np.zeros(c.n, dtype=np.float32)
        aff_lower = [a.lower().rstrip("s") for a in intent.affinities]
        for i, c_low in enumerate(c.cat_lower):
            c_stem = c_low.rstrip("s")
            if any(a in c_low or c_stem in a for a in aff_lower):
                scores[i] += _AFF_BOOST
        for i, t_low in enumerate(c.title_lower):
            for a in aff_lower:
                if a in t_low:
                    scores[i] += _AFF_BOOST * 0.5
                    break
        # If gifting to mother, suppress 'mom fit' denim style false positives
        if intent.recipient == "mother" and hasattr(c, "is_mom_fit"):
            scores -= 30.0 * c.is_mom_fit

        if len(self._intent_score_cache) < 2000:
            self._intent_score_cache[cache_key] = scores.copy()
        return scores

    def _soft_boost_nlp(self, intent: StructuredIntent) -> np.ndarray:
        """Apply soft boosts from NLP structured intent (brand, color, material, affinity)."""
        c = self.cat
        boost = np.zeros(c.n, dtype=np.float64)
        if intent.brand:
            boost += _BRAND_BOOST * (c.brand_n == alnum(intent.brand))
        if intent.color:
            boost += _COLOR_BOOST * (c.color_n == alnum(intent.color))
        if intent.material:
            pm = alnum(intent.material)
            boost += _MATERIAL_BOOST * np.array([
                (pm in m) or (m in pm) if m else False for m in c.mat_uni
            ])[c.mat_code]
        boost += self._intent_compat_scores(intent)
        return boost

    # ------------------------------------------------------------ public API
    def search(self, query: str, k: int = 10, use_cache: bool = True) -> Dict:
        t0 = time.perf_counter()
        query = (query or "").strip()
        if not query:
            return {"query": query, "results": [], "intent": {}, "source": "none", "notes": ["empty query"],
                    "cached": False, "took_ms": 0.0}
        rkey = sha(f"{self.cat.version}|{k}|{self.s.use_rerank}|{_qkey(query)}")
        if use_cache:
            hit = self.cache.get_json("result", rkey)
            if hit:
                elapsed = round((time.perf_counter() - t0) * 1000, 2)
                hit["cached"], hit["is_cache_hit"] = True, True
                hit["took_ms"], hit["latency_ms"] = elapsed, elapsed
                return hit
        out = self._search(query, k)
        elapsed = round((time.perf_counter() - t0) * 1000, 2)
        out["took_ms"], out["latency_ms"] = elapsed, elapsed
        out["cached"], out["is_cache_hit"] = False, False
        if use_cache:
            self.cache.set_json("result", rkey, out, self.s.ttl_result)
        return out

    def _search(self, query: str, k: int) -> Dict:
        s, c = self.s, self.cat

        # ── Step 1: Advanced NLP parse (Bangla/Banglish, persona, intent, price, size) ──
        nlp_intent = self._nlp.parse(query)
        q_norm = nlp_intent.normalized_query
        q_clean = nlp_intent.retrieval_query
        notes: List[str] = []

        if nlp_intent.corrections:
            notes.append(f"spelling corrected: {', '.join(nlp_intent.corrections)}")
        if nlp_intent.language == "bn":
            notes.append(f"Bangla/Banglish query detected -> translated: '{q_clean}'")
        if nlp_intent.recipient:
            notes.append(f"persona: {nlp_intent.recipient}")
        if nlp_intent.intents:
            notes.append(f"intent: {', '.join(nlp_intent.intents)}")

        # ── Step 2: Rule parser (existing lite_search rules) ──
        # Strip negated phrases before positive RuleParser extraction so 'not X' never sets brand/color=X
        q_rule = re.sub(
            r"\b(?:not|except|without|no)\s+[a-z0-9\u0980-\u09FF\s\-]+?(?=\s+(?:under|below|above|between|for|size|with|and|,|\.|$)|$)",
            " ",
            q_norm,
        )
        q_rule = re.sub(r"[a-z0-9\u0980-\u09FF]+\s+(?:ছাড়|ছাড়া|ব্যতীত|বিনা)", " ", q_rule)
        q_rule = re.sub(r"\s+", " ", q_rule).strip() or q_norm

        grp_mask, grp_names = self.parser.detect_groups(q_clean)
        if grp_mask is None:
            grp_mask, grp_names = self.parser.detect_groups(q_rule)
        parsed, source = self._intent(q_rule, grp_mask)

        # Merge NLP intent into rule parser intent (NLP fills gaps & wins on compound accessory types)
        ex_b_set = {alnum(b) for b in nlp_intent.excluded_brands}
        ex_c_set = {alnum(c_) for c_ in nlp_intent.excluded_colors}
        if parsed.get("brand") and alnum(parsed["brand"]) in ex_b_set:
            parsed["brand"] = None
        if parsed.get("color") and alnum(parsed["color"]) in ex_c_set:
            parsed["color"] = None

        if not parsed.get("color") and nlp_intent.color and alnum(nlp_intent.color) not in ex_c_set:
            parsed["color"] = nlp_intent.color
        if not parsed.get("brand") and nlp_intent.brand and alnum(nlp_intent.brand) not in ex_b_set:
            # If brand token is identical to color token (e.g. 'yellow'), treat as color, not brand
            if not (parsed.get("color") and alnum(nlp_intent.brand) == alnum(parsed["color"])):
                parsed["brand"] = nlp_intent.brand
        if parsed.get("brand") and parsed.get("color") and alnum(parsed["brand"]) == alnum(parsed["color"]):
            parsed["brand"] = None
        if not parsed.get("material") and nlp_intent.material:
            parsed["material"] = nlp_intent.material
        if not parsed.get("max_price") and nlp_intent.max_price:
            parsed["max_price"] = nlp_intent.max_price
        if not parsed.get("min_price") and nlp_intent.min_price:
            parsed["min_price"] = nlp_intent.min_price
        if not parsed.get("size") and nlp_intent.size:
            parsed["size"] = nlp_intent.size
        if not parsed.get("gender") and nlp_intent.target_gender:
            parsed["gender"] = nlp_intent.target_gender
        if nlp_intent.product_type and (
            not parsed.get("category")
            or nlp_intent.product_type in ("phone case", "phone charger", "charger", "beard trimmer", "sunscreen", "smartwatch", "power bank")
        ):
            parsed["category"] = nlp_intent.product_type

        # ── Step 2.5: Intelligent Apparel & Persona Disambiguation ──
        is_kid_query = (nlp_intent.recipient in ("kid", "baby") or
                        any(w in q_norm or w in q_clean or w in query.lower()
                            for w in ("baccha", "bacchader", "shishu", "shishuder", "chotoder", "kid", "kids", "baby", "বাচ্চা", "বাচ্চাদের")))
        effective_gender = parsed.get("gender") or nlp_intent.target_gender
        if is_kid_query:
            effective_gender = "kids"
            parsed["gender"] = "kids"
        elif not effective_gender and any(w in q_norm or w in q_clean or w in query.lower() for w in ("chele", "cheler", "cheleder", "purush", "purusher", "men", "mens", "ছেলে", "ছেলেদের")):
            effective_gender = "men"
            parsed["gender"] = "men"
        elif not effective_gender and any(w in q_norm or w in q_clean or w in query.lower() for w in ("meye", "meyer", "meyeder", "mohila", "mohilader", "women", "womens", "ladies", "মেয়ে", "মেয়েদের")):
            effective_gender = "women"
            parsed["gender"] = "women"

        # "boys" as gender → treat as kids, NOT adult men
        q_all = (q_norm + " " + q_clean + " " + query.lower()).strip()
        if effective_gender == "boys" or any(
                w in q_all for w in ("for boys", "little boy", "choto chele", "baccha chele")):
            effective_gender = "kids"
            parsed["gender"] = "kids"
            is_kid_query = True

        # Detect casual/everyday intent (only for apparel queries)
        is_casual_query = any(w in q_all for w in (
            "casual", "everyday", "daily wear", "daily use", "regular outfit",
            "simple outfit", "daily outfit", "casual outfit", "casual wear"
        ))
        has_non_apparel = any(w in q_all for w in (
            "phone", "mobile", "smartphone", "laptop", "tablet", "computer",
            "camera", "watch", "smartwatch", "headphone", "earphone", "audio",
            "speaker", "ac", "air conditioner", "refrigerator", "fridge", "fan",
            "tv", "television", "charger", "power bank", "multimeter"
        )) or (parsed.get("category") in {
            "smartphone", "laptop", "tablet", "monitor", "electronics", "gadget",
            "camera", "headphone", "headphones", "earbuds", "audio", "air conditioner",
            "refrigerator", "fan", "microwave", "blender", "power bank"
        })
        if is_casual_query and not has_non_apparel:
            if not nlp_intent.intents:
                nlp_intent.intents.append("casual")
            if not nlp_intent.affinities:
                nlp_intent.affinities = ["t-shirt", "jeans", "sneakers", "shirt", "hoodie"]
        elif has_non_apparel:
            if "casual" in nlp_intent.intents:
                nlp_intent.intents.remove("casual")
            if nlp_intent.affinities:
                nlp_intent.affinities = [
                    a for a in nlp_intent.affinities
                    if a not in ("t-shirt", "jeans", "sneakers", "shirt", "hoodie", "polo-shirt", "pant")
                ]

        is_generic_apparel = any(w in q_all for w in (
            "dress", "dresses", "clothes", "clothing", "apparel", "wear", "outfit", "attire",
            "jama", "jaama", "kapor", "kapod", "poshak", "posak",
            "পোশাক", "পোষাক", "ড্রেস", "জামা", "কাপড়", "বস্ত্র"
        ))
        is_specific_dress = any(w in q_all for w in (
            "maxi", "midi", "mini", "wrap", "summer dress", "floral dress", "evening dress", "frock", "cocktail", "gown"
        ))

        # Override persona affinities when specific product type is in the query
        # e.g., "kids shoes" → shoe affinities, not toy affinities
        FOOTWEAR_WORDS = {"shoe", "shoes", "boot", "boots", "sneaker", "sneakers",
                          "sandal", "sandals", "slipper", "slippers", "juta", "juto"}
        FOOTWEAR_CATS = {"running shoes", "sneakers", "formal shoes", "sandals", "walking shoes",
                         "boots", "slippers", "school shoes", "sports shoes"}
        query_has_footwear = any(re.search(rf"\b{re.escape(w)}\b", q_all) for w in FOOTWEAR_WORDS)

        _PRODUCT_AFFINITY_MAP = [
            ({"shoe", "shoes", "juta", "juto", "footwear"}, ["running shoes", "sneakers", "formal shoes", "sandals"]),
            ({"boot", "boots"}, ["boots"]),
            ({"sneaker", "sneakers"}, ["sneakers"]),
            ({"sandal", "sandals"}, ["sandals"]),
            ({"backpack", "bagpack"}, ["backpack"]),
            ({"jacket"}, ["jacket"]),
            ({"hoodie"}, ["hoodie"]),
            ({"watch", "ghori"}, ["wristwatch", "smartwatch"]),
            ({"smartwatch"}, ["smartwatch"]),
        ]
        for _pwords, _paffs in _PRODUCT_AFFINITY_MAP:
            if any(re.search(rf"\b{re.escape(pw)}\b", q_all) for pw in _pwords):
                nlp_intent.affinities = _paffs
                break

        MEN_CLOTHING_CATS = {
            "panjabi", "shirt", "t-shirt", "jeans", "trousers", "jacket", "fotua",
            "hoodie", "sweater", "lungi", "pajamas", "shorts", "coat", "clothing", "track pants"
        }
        WOMEN_CLOTHING_CATS = {
            "saree", "kurti", "salwar kameez", "salwar", "dress", "jeans",
            "jacket", "sweater", "leggings", "scarf", "clothing"
        }
        KIDS_CLOTHING_CATS = {
            "dress", "panjabi", "pajamas", "kurti", "t-shirt", "jeans", "shorts", "romper", "clothing"
        }

        # For kids + footwear queries, route to kids footwear group (not toys)
        if is_kid_query and query_has_footwear:
            parsed["category"] = None
            grp_mask = c.kid_item & np.isin(c.cat_n, list(FOOTWEAR_CATS))
            grp_names = ["kids' footwear"]
            nlp_intent.affinities = ["running shoes", "sandals", "sneakers", "school shoes"]

        # Generic apparel query with gender context → route to gender-specific clothing
        elif is_generic_apparel and not is_specific_dress:
            if effective_gender in ("men",):
                parsed["category"] = None
                grp_mask = np.isin(c.cat_n, list(MEN_CLOTHING_CATS)) & np.isin(c.gender, ["men", "unisex"])
                grp_names = ["men's clothing"]
                if not nlp_intent.affinities:
                    nlp_intent.affinities = ["panjabi", "shirt", "t-shirt", "jeans", "trousers", "jacket"]
            elif effective_gender in ("women", "girls"):
                parsed["category"] = None
                grp_mask = np.isin(c.cat_n, list(WOMEN_CLOTHING_CATS)) & np.isin(c.gender, ["women", "girls", "unisex"])
                grp_names = ["women's clothing"]
                if not nlp_intent.affinities:
                    nlp_intent.affinities = ["saree", "kurti", "salwar kameez", "dress"]
            elif effective_gender in ("kids", "baby") or is_kid_query:
                parsed["category"] = None
                grp_mask = c.kid_item & np.isin(c.cat_n, list(KIDS_CLOTHING_CATS))
                grp_names = ["kids' clothing"]
                if not nlp_intent.affinities:
                    nlp_intent.affinities = ["dress", "panjabi", "pajamas", "kurti", "t-shirt"]

        is_accessory_query = any(
            w in q_clean.lower() or w in q_norm.lower()
            for w in ("case", "cover", "protector", "charger", "adapter", "cable", "holder", "stand", "keyboard", "strap", "battery")
        )
        is_device_query = (cat_norm(parsed["category"]) in ("smartphone", "laptop", "tablet") if parsed.get("category") else False) and not is_accessory_query

        if parsed.get("category"):
            c_target = cat_norm(parsed["category"])
            # If leaf category 'dress' requested for male, redirect to men's clothing
            if c_target == "dress" and effective_gender in ("men",):
                parsed["category"] = None
                grp_mask = np.isin(c.cat_n, list(MEN_CLOTHING_CATS)) & np.isin(c.gender, ["men", "unisex"])
                grp_names = ["men's clothing"]
            else:
                # Exact or boundary-matched substring within noisy shop category strings (e.g. 'electronics & tech (oraimo earbuds)')
                cat_pat = re.compile(rf"(?<![a-z0-9]){re.escape(c_target)}(?![a-z0-9])") if len(c_target) >= 3 else None
                uni_cat = np.array([
                    (u == c_target) or bool(cat_pat and cat_pat.search(u))
                    for u in c.cat_uni
                ], dtype=bool)
                cm = uni_cat[c.cat_code].copy()

                accessory_cats = {"backpack", "phone case", "charger", "case", "cover", "screen protector", "cable", "stand"}
                if not is_accessory_query:
                    cm |= np.array([c_target in t and c.cat_n[i] not in accessory_cats and not c.is_phone_accessory[i] for i, t in enumerate(c.title_lower)])

                if c_target == "smartwatch":
                    cm |= np.array([
                        ("wearable" in cat) or ("smartwatch" in t) or ("smart watch" in t) or ("smart" in t and "watch" in t)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target == "phone case":
                    _phone_cues = ("phone", "mobile", "iphone", "samsung", "galaxy", "xiaomi", "redmi", "realme", "motorola", "pixel", "vivo", "oppo", "oneplus", "nokia")
                    cm |= np.array([
                        (("case" in t or "cover" in t or "case" in cat or "cover" in cat) and any(pw in t or pw in cat for pw in _phone_cues))
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target in ("phone charger", "charger"):
                    cm |= np.array([
                        ("charger" in t or "adapter" in t or "charger" in cat or "adapter" in cat)
                        and not ("cable" in cat and "charger" not in cat and "charger" not in t)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                    cm &= np.array([
                        ("charger" in t or "adapter" in t or "charger" in cat or "adapter" in cat)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target in ("beard trimmer", "trimmer"):
                    cm |= np.array([
                        ("trimmer" in t or "shaver" in t or "trimmer" in cat)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target == "sunscreen":
                    cm |= np.array([
                        ("sunscreen" in t or "sun block" in t or "sunblock" in t or "uvshield" in t or "spf" in t or "sunscreen" in cat)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target == "face wash":
                    cm |= np.array([
                        ("face wash" in t or "facewash" in t or "ফেস ওয়াশ" in t or "ফেসওয়াশ" in t or "face wash" in cat)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target == "saree":
                    cm |= np.array([
                        ("saree" in t or "sarees" in t or "শাড়ি" in t or "শাড়ি" in t or "saree" in cat)
                        for cat, t in zip(c.cat_lower, c.title_lower)
                    ])
                elif c_target == "smartphone":
                    cm |= np.array([
                        ("phone" in cat or "smartphone" in cat or "mobile" in cat or "phone" in t or "smartphone" in t)
                        and not c.is_phone_accessory[i]
                        for i, (cat, t) in enumerate(zip(c.cat_lower, c.title_lower))
                    ])
                    cm &= ~c.is_phone_accessory
                elif c_target in ("headphone", "wireless headphone", "headphones", "wireless headphones"):
                    cm |= c.is_earbud | np.array(["headphone" in cat or "earphone" in cat or "headphone" in t or "earphone" in t for cat, t in zip(c.cat_lower, c.title_lower)])
                    if not is_accessory_query:
                        cm &= ~np.array(["adapter" in cat or "adapter" in t or "dongle" in t or "splitter" in t for cat, t in zip(c.cat_lower, c.title_lower)])
                elif c_target in ("earbud", "earbuds"):
                    cm |= c.is_earbud | np.array(["headphone" in cat or "earphone" in cat or "headphone" in t for cat, t in zip(c.cat_lower, c.title_lower)])
                    if not is_accessory_query:
                        cm &= ~np.array(["adapter" in cat or "adapter" in t or "dongle" in t for cat, t in zip(c.cat_lower, c.title_lower)])
                elif c_target in ("bicycle", "bike"):
                    cm |= (c.cat_n == "bicycle") | c.is_bike
                elif c_target in ("jacket", "raincoat") and any(w in q_clean or w in q_norm for w in ("rain", "waterproof", "rainy")):
                    cm |= (c.cat_n == "jacket") | (c.cat_n == "raincoat")
                if cm.any():
                    grp_mask, grp_names = cm, [str(parsed["category"])]

        if "room_cooling" in nlp_intent.intents or ("air conditioner" in q_clean and "fan" in q_clean):
            cm_cool = np.array([
                bool(re.search(r"\b(?:air\s*conditioner|air\s*cooler|multicool|ceiling\s*fan|table\s*fan|portable\s*fan|cooling\s*fan|exhaust\s*fan|fan|ac)\b", cat + " " + t))
                and not any(bad in t for bad in ("sandal", "shoe", "shirt", "hanger", "jacket", "laptop", "stand", "holder", "cpu cooler", "cooling pad"))
                for cat, t in zip(c.cat_lower, c.title_lower)
            ], dtype=bool)
            if cm_cool.any():
                grp_mask, grp_names = cm_cool, ["room cooling"]

        # ── Step 3: Dual retrieval — BM25 + Dense embeddings ──
        cos = c.emb @ self.qemb.encode_query(q_clean)
        q_bm25 = f"{q_clean} {re.sub(r'[0-9০-৯]+', ' ', nfc(query.lower()))}" if nlp_intent.language == "bn" else q_clean
        bm = c.bm25(q_bm25)

        # ── Step 4: Intent compatibility scores (persona/affinity) ──
        intent_scores = self._intent_compat_scores(nlp_intent).astype(np.float64)
        # Normalize intent scores to [0,1] range for RRF
        intent_norm = intent_scores / (intent_scores.max() + 1e-9)

        # ── Step 5: RRF Fusion (BM25 + Dense + Intent) ──
        rrf = (
            _BM25_W * self._rrf_ranks(bm, min_val=0.0) +
            _DENSE_W * self._rrf_ranks(cos, min_val=0.0) +
            _INTENT_W * self._rrf_ranks(intent_norm, min_val=0.0)
        )

        # ── Step 6: Hard constraint filtering with relaxation ──
        active, relaxed = [f for f in RELAX_ORDER if s.hard_price or f != "price"], []
        while True:
            mask = self._mask(parsed, active)
            if mask.sum() >= s.min_results or not active:
                break
            relaxed.append(active.pop(0))

        # Apply hard mask to RRF scores
        base = np.where(mask, rrf, -1e9)

        # ── Step 7: Soft boosts (attribute matching + NLP affinities) ──
        bonus, flags = self._attr_bonus(parsed)
        nlp_boost = self._soft_boost_nlp(nlp_intent)
        p_bonus, p_in = self._price_tier(parsed)
        aud = self._audience(q_norm, parsed, nlp_intent)
        grp_bonus = s.group_tier * grp_mask if grp_mask is not None else np.zeros(c.n)
        bm_bonus = np.where(bm > 0, 15.0 * (bm / (bm.max() + 1e-9)), 0.0)
        # Dedicated boost for in-ear audio queries (earphones, earbuds, airpods)
        ear_bonus = np.zeros(c.n)
        if any(w in q_clean or w in q_norm for w in ("earbud", "earphone", "airpod", "ইয়ারবাড", "ইয়ারফোন", "ইয়ারবাড", "ইয়ারফোন")):
            ear_bonus = np.where(c.is_earbud, 20.0, 0.0)

        # Dedicated boost for noise cancelling / flight audio queries
        anc_bonus = np.zeros(c.n)
        if any(w in q_clean or w in q_norm for w in ("noise cancel", "cancellation", "anc", "flight", "plane", "travel")):
            anc_bonus = np.where(c.is_anc, 25.0, 0.0)
            if any(w in q_clean or w in q_norm for w in ("headphone", "headphones")):
                anc_bonus += np.where(c.is_anc & np.isin(c.cat_n, ["headphone", "wirelessheadphone"]), 20.0, 0.0)
                anc_bonus += np.where(c.is_anc & (c.cat_n == "earbud"), -15.0, 0.0)

        # Budget boost when query implies budget/cheap/affordable but no numeric max_price is given
        is_budget_query = bool(re.search(r"\b(?:budget|budget-friendly|cheap|affordable|low\s*cost|কম\s*বাজেটে|সাশ্রয়ী)\b", q_clean.lower() + " " + q_norm.lower()))
        budget_bonus = np.zeros(c.n)
        if is_budget_query and not parsed.get("max_price"):
            rel_mask = grp_mask if (grp_mask is not None and grp_mask.any()) else np.ones(c.n, dtype=bool)
            valid_p = c.price[rel_mask & (c.price > 0)]
            if len(valid_p):
                median_p = np.median(valid_p)
                budget_bonus = np.where(rel_mask & (c.price > 0) & (c.price <= median_p), 18.0, 0.0)

        # Distinctive model tokens boost (e.g. 's24', 'ultra', 'm3', '15 pro', '14 pro', 'wh-1000xm5')
        model_bonus = np.zeros(c.n)
        q_text_all = (query.lower() + " " + q_clean.lower() + " " + q_norm.lower())
        q_tokens = [w for w in re.findall(r"[a-z0-9]+", q_text_all) if w not in STOP]
        distinctive_tokens = [w for w in q_tokens if any(ch.isdigit() for ch in w) or w in ("ultra", "pro", "max", "air", "plus", "mini", "lite", "slim")]
        if distinctive_tokens:
            for i in range(c.n):
                t_lower = c.title_lower[i]
                matched = sum(1 for dt in distinctive_tokens if re.search(rf"\b{re.escape(dt)}\b", t_lower))
                if matched == len(distinctive_tokens):
                    model_bonus[i] = 25.0
                elif matched > 0:
                    model_bonus[i] = 12.0 * (matched / len(distinctive_tokens))

        # Specialized feature bonuses (inverter AC, genuine AC vs purifier, waterproof rain gear, programming laptops)
        feature_bonus = np.zeros(c.n)
        if "inverter" in q_text_all and "non-inverter" not in q_text_all and "non inverter" not in q_text_all:
            is_inv = np.array(["inverter" in t and "non-inverter" not in t and "non inverter" not in t for t in c.title_lower])
            is_non_inv = np.array(["non-inverter" in t or "non inverter" in t for t in c.title_lower])
            feature_bonus += np.where(is_inv, 25.0, 0.0)
            feature_bonus += np.where(is_non_inv, -25.0, 0.0)

        if any(w in q_text_all for w in ("ac", "air conditioner")):
            is_purifier = np.array(["air purifier" in t for t in c.title_lower])
            feature_bonus += np.where(is_purifier, -25.0, 0.0)

        if any(w in q_text_all for w in ("waterproof", "rain", "rainy", "বৃষ্টি")):
            if parsed.get("category") in ("jacket", "raincoat") or any(w in q_text_all for w in ("jacket", "raincoat")):
                is_rain_gear = np.array(["waterproof" in t or "rain" in t or "raincoat" in t for t in c.title_lower])
                is_warm_gear = np.array(["puffer" in t or "fleece" in t for t in c.title_lower])
                feature_bonus += np.where(is_rain_gear, 25.0, 0.0)
                feature_bonus += np.where(is_warm_gear, -15.0, 0.0)

        if any(w in q_text_all for w in ("programming", "coding", "developer")):
            is_prog_laptop = np.array([
                bool(re.search(r"16gb|core\s*i5|core\s*i7|ryzen\s*5|ryzen\s*7|thinkpad|victus|gaming|ideapad|pavilion|inspiron|latitude", t + " " + desc))
                for t, desc in zip(c.title_lower, c.desc_lower)
            ])
            feature_bonus += np.where(is_prog_laptop, 20.0, 0.0)
            if is_budget_query:
                # Promote budget-friendly workhorse laptops (50k - 85k BDT) and penalize expensive 100k+ laptops
                mid_budget_prog = (c.cat_n == "laptop") & (c.price >= 45000) & (c.price <= 85000)
                expensive_laptop = (c.cat_n == "laptop") & (c.price > 100000)
                feature_bonus += np.where(mid_budget_prog, 20.0, 0.0)
                feature_bonus += np.where(expensive_laptop, -25.0, 0.0)

        # Building blocks for kids
        if any(w in q_text_all for w in ("build thing", "building block", "build", "নির্মাণ")):
            is_building = (c.cat_n == "buildingblocks") | np.array(["building block" in t for t in c.title_lower])
            feature_bonus += np.where(is_building, 25.0, 0.0)

        # Long distance running shoes
        if any(w in q_text_all for w in ("distance", "marathon", "cushion")):
            is_distance_shoe = np.array([bool(re.search(r"\b(pegasus|ultraboost|vomero|solarboost|kayano|cumulus|gel-kayano|gel-cumulus|distance)\b", t)) for t in c.title_lower])
            feature_bonus += np.where(is_distance_shoe, 25.0, 0.0)

        # Gift for mom
        if any(w in q_text_all for w in ("gift for mom", "gift for mother", "for mom", "for mother", "মায়ের জন্য", "মার জন্য")):
            is_mom_jeans = np.array([bool(re.search(r"\bmom\s*fit|\bmom\s*jeans\b", t)) for t in c.title_lower])
            feature_bonus += np.where(is_mom_jeans, -35.0, 0.0)
            is_mom_gift = np.isin(c.cat_n, ["necklace", "saree", "kurti", "salwarkameez", "handbag", "perfume"])
            feature_bonus += np.where(is_mom_gift, 20.0, 0.0)

        # Title keyword matching: when 'iphone' is queried, products containing 'iphone' in title rank highest
        if "iphone" in q_text_all:
            is_iphone = np.array(["iphone" in t for t in c.title_lower])
            feature_bonus += np.where(is_iphone, 25.0, 0.0)

        # Sony flagship headphones boost
        if "sony" in q_text_all and any(w in q_text_all for w in ("headphone", "headphones")):
            is_sony_flagship = np.array([bool(re.search(r"\b(wh-|wh\b|wf-|1000xm|ch720|ch520)", t)) for t in c.title_lower])
            feature_bonus += np.where(is_sony_flagship, 15.0, 0.0)

        # Flagship athletic shoes for broad shoes queries
        if any(w in q_text_all for w in ("good shoes", "best shoes", "জুতো", "জুতা")):
            is_flagship_shoe = np.array([bool(re.search(r"\b(pegasus|ultraboost|vomero|cumulus|gel-cumulus|air max|stan smith|galaxy|duramo|softride|venturini)\b", t)) for t in c.title_lower])
            is_premier_runner = np.array([bool(re.search(r"\b(pegasus|ultraboost|vomero|cumulus|gel-cumulus|kayano|venturini)\b", t)) for t in c.title_lower])
            is_school_kids_shoe = np.array([bool(re.search(r"\b(school|kids|baby|boy|girl)\b", t)) for t in c.title_lower])
            feature_bonus += np.where(is_flagship_shoe, 20.0, 0.0)
            feature_bonus += np.where(is_premier_runner, 12.0, 0.0)
            feature_bonus += np.where(is_school_kids_shoe, -25.0, 0.0)

        # Boost for everyday/daily use phones (mid-range 15000-36000 BDT) & authentic smartphone brands
        everyday_bonus = np.zeros(c.n)
        is_phone_query = (parsed.get("category") == "smartphone" or "phone" in q_text_all) and not is_accessory_query
        if is_phone_query:
            AUTH_PHONE_BRANDS = {"samsung", "xiaomi", "apple", "oneplus", "vivo", "realme", "walton", "oppo", "google", "motorola", "symphony", "poco", "honor", "nothing"}
            SPAM_PHONE_BRANDS = {"dell", "hp", "logitech", "anker", "canon", "lenovo"}
            is_auth = np.isin(c.brand_n, list(AUTH_PHONE_BRANDS))
            is_spam = np.isin(c.brand_n, list(SPAM_PHONE_BRANDS))
            everyday_bonus += np.where(is_auth, 12.0, 0.0)
            everyday_bonus += np.where(is_spam, -35.0, 0.0)

            if any(w in q_clean or w in q_norm for w in ("everyday", "daily", "প্রতিদিনের", "best phone", "regular")):
                mid_range = (c.cat_n == "smartphone") & (c.price >= 12000) & (c.price <= 36000)
                pop_models = np.array([bool(re.search(r"\b(a15|redmi|note|nord|primo|galaxy\s*a)\b", t)) for t in c.title_lower])
                everyday_bonus += np.where(mid_range & pop_models, 20.0, np.where(mid_range, 10.0, 0.0))

        # Quality brand boost for broad queries like "good shoes" / "best shoes"
        quality_bonus = np.zeros(c.n)
        if any(w in q_clean or w in q_norm for w in ("good", "best", "ভালো")):
            top_brands = {"nike", "adidas", "asics", "skechers", "hoka", "puma", "bata", "apex"}
            quality_bonus = np.where(np.isin(c.brand_n, list(top_brands)), 5.0, 0.0)

        tiers = grp_bonus + bonus + bm_bonus + ear_bonus + anc_bonus + budget_bonus + everyday_bonus + quality_bonus + model_bonus + feature_bonus + nlp_boost + p_bonus + aud

        sel = np.where(base > -1e8, base + tiers, base)

        # Primary Category Constraint:
        cat_active = grp_mask is not None and grp_mask.any()
        if cat_active:
            sel = np.where(grp_mask, sel, -1e9)

        # Device vs Accessory guard: never return cases/gimbals/SIM pins when user asked for the device itself
        if is_device_query:
            sel = np.where(c.is_phone_accessory, -1e9, sel)

        # Uncatalogued item check
        if "uncatalogued_item_" in nlp_intent.normalized_query:
            sel = np.full(c.n, -1e9)

        # Negative exclusions: strictly exclude negative brands (in brand field OR title), colors, categories
        ex_b_masks = []
        if nlp_intent.excluded_brands:
            for ex_b in nlp_intent.excluded_brands:
                ex_al = alnum(ex_b)
                ex_low = nfc(ex_b).lower()
                bm_ex = (c.brand_n == ex_al) | np.array([word_in(ex_low, t) for t in c.title_lower], dtype=bool)
                ex_b_masks.append(bm_ex)
                sel = np.where(bm_ex, -1e9, sel)
        if nlp_intent.excluded_colors:
            for ex_c in nlp_intent.excluded_colors:
                sel = np.where(c.color_n == alnum(ex_c), -1e9, sel)
        if nlp_intent.excluded_categories:
            for ex_cat in nlp_intent.excluded_categories:
                sel = np.where(c.cat_n == cat_norm(ex_cat), -1e9, sel)

        cat_name = str(parsed.get("category") or (grp_names[0] if grp_names else "items"))

        # Brand filtering (matches brand column OR brand word in title since multi-shop catalogs often leave brand empty)
        req_brand = parsed.get("brand") or nlp_intent.brand
        brand_mask = None
        brand_strict = False
        if req_brand and "uncatalogued_item_" not in nlp_intent.normalized_query:
            req_b_al = alnum(req_brand)
            req_b_low = nfc(req_brand).lower()
            brand_mask = (c.brand_n == req_b_al) | np.array([word_in(req_b_low, t) for t in c.title_lower], dtype=bool)
            if cat_active:
                if (grp_mask & brand_mask).any():
                    sel = np.where(brand_mask, sel, -1e9)
                    brand_strict = True
                else:
                    sel = np.full(c.n, -1e9)
                    brand_strict = True
                    notes.append(f"No {cat_name} found from brand '{req_brand}'")
            else:
                if brand_mask.any():
                    sel = np.where(brand_mask, sel, -1e9)
                    brand_strict = True
                else:
                    sel = np.full(c.n, -1e9)
                    brand_strict = True

        # Color filtering with secondary fallback:
        req_color = parsed.get("color") or nlp_intent.color
        color_mask = None
        color_strict = False
        if req_color:
            color_mask = c.color_n == alnum(req_color)
            if cat_active:
                if (grp_mask & color_mask).any():
                    sel = np.where(color_mask, sel, -1e9)
                    color_strict = True
                else:
                    # Secondary fallback: Category exists, but requested color is not available
                    color_strict = False
                    notes.append(f"No {req_color} {cat_name} in catalog - showing other popular {cat_name} options")
            else:
                if color_mask.any():
                    sel = np.where(color_mask, sel, -1e9)
                    color_strict = True
                else:
                    color_strict = False

        # Material filtering with secondary fallback:
        req_mat = parsed.get("material") or nlp_intent.material
        mat_mask = None
        mat_strict = False
        if req_mat:
            pm = alnum(req_mat)
            mat_mask = np.array([(pm in m) or (m in pm) if m else False for m in c.mat_uni])[c.mat_code]
            if cat_active:
                if (grp_mask & mat_mask).any():
                    sel = np.where(mat_mask, sel, -1e9)
                    mat_strict = True
                else:
                    mat_strict = False
                    notes.append(f"No {req_mat} {cat_name} in catalog - showing other {cat_name} options")
            else:
                if mat_mask.any():
                    sel = np.where(mat_mask, sel, -1e9)
                    mat_strict = True
                else:
                    mat_strict = False

        pool = max(s.pool, k, _CANDIDATE_POOL)
        valid_indices = [int(i) for i in range(c.n) if sel[i] > -1e8]
        if valid_indices:
            cand = [int(i) for i in np.argpartition(-sel, min(pool, len(valid_indices) - 1))[:min(pool, len(valid_indices))] if sel[i] > -1e8]
            cand.sort(key=lambda i: -sel[i])
        else:
            cand = []

        if not cand:
            # Fallback to dense similarity only if minimum similarity threshold is met (prevents PlayStation -> Smartwatch)
            cos_threshold = s.cos_min
            above_thresh = [int(i) for i in np.where(cos >= cos_threshold)[0]]
            if is_device_query:
                above_thresh = [i for i in above_thresh if not c.is_phone_accessory[i]]
            for ex_bm in ex_b_masks:
                above_thresh = [i for i in above_thresh if not ex_bm[i]]
            req_g = parsed.get("gender") or nlp_intent.target_gender
            if req_g in ("men",):
                above_thresh = [i for i in above_thresh if c.gender[i] in ("men", "unisex")]
            elif req_g in ("women", "girls"):
                above_thresh = [i for i in above_thresh if c.gender[i] in ("women", "girls", "unisex")]
            elif req_g in ("kids", "baby", "boys"):
                above_thresh = [i for i in above_thresh if c.kid_item[i] or c.gender[i] in ("boys", "girls", "kids")]
            if above_thresh and (not req_brand or not brand_strict) and "uncatalogued_item_" not in nlp_intent.normalized_query:
                above_thresh.sort(key=lambda i: -cos[i])
                cand = above_thresh[:pool]
                notes.append("showing products by semantic meaning")
            else:
                cand = []
                notes.append(f"No relevant products found for '{query}'")

        # ── Step 8: Neural cross-encoder reranking ──
        top_rerank = min(25, max(k + 8, 15))
        ce_raw = self._rerank(q_clean, cand, top_n=top_rerank) if (s.use_rerank and len(cand) >= 1) else np.zeros(len(cand))
        confident = (grp_mask is not None) or bool(flags) or (p_in is not None) or (len(cand) > 0 and bm[cand].max() > 0) \
            or (len(cand) > 0 and cos[cand].max() >= s.cos_min) or bool(nlp_intent.brand or nlp_intent.recipient or nlp_intent.intents) \
            or (s.use_rerank and len(cand) > 0 and ce_raw.max() >= s.ce_min)

        if confident and cand:
            if s.use_rerank and len(cand) > 1:
                blend = (1 - s.ce_weight) * minmax(base[cand]) + s.ce_weight * minmax(ce_raw)
            else:
                blend = minmax(base[cand]) if len(cand) > 1 else np.ones(1)
            final = blend + tiers[cand]
            label = (["intent" if grp_mask[i] else "related" for i in cand] if grp_mask is not None
                     else ["intent" if nlp_intent.intents or nlp_intent.recipient else "" for _ in cand])
        elif cand:
            final = (minmax(base[cand]) if len(cand) > 1 else np.ones(1)) + p_bonus[cand] + aud[cand]
            label = ["closest"] * len(cand)
        else:
            final = np.array([])
            label = []

        # Only return items up to k that are truly relevant (not empty/unmatched padding)
        if len(cand) > 0:
            core_req = np.ones(c.n, dtype=bool)
            if grp_mask is not None:
                core_req &= grp_mask
            if color_strict and color_mask is not None:
                core_req &= color_mask
            if brand_strict and brand_mask is not None:
                core_req &= brand_mask
            if mat_strict and mat_mask is not None:
                core_req &= mat_mask

            is_price_relaxed = ("price" in relaxed) or (not s.hard_price and p_in is not None and not (core_req & p_in).any())
            if is_price_relaxed and (parsed.get("max_price") or parsed.get("min_price")):
                if not any("closest" in n for n in notes):
                    notes.append("No items under requested price - showing closest available prices")

            # Prepare lexical token coverage & phone model generation checks
            bm_content_toks = [w for w in set(toks(q_clean)) if len(w) >= 2]
            gen_model_m = re.search(r"\b(?:iphone|galaxy\s*s)\s*(\d{2})\b", q_text_all) if is_device_query else None

            filtered_cand_order = []
            req_gender = parsed.get("gender") or nlp_intent.target_gender
            is_gender_relaxed = "gender" in relaxed
            for j in np.argsort(-final):
                idx = cand[j]
                if is_device_query and c.is_phone_accessory[idx]:
                    continue
                if gen_model_m and not word_in(gen_model_m.group(1), c.title_lower[idx]):
                    continue
                # Gender constraint check: When gender is specified and not relaxed, NEVER return opposite gender
                if req_gender and not is_gender_relaxed:
                    cand_g = str(c.df.at[idx, "gender"]).lower()
                    if req_gender in ("men",):  # "boys" is now normalized to "kids"
                        if cand_g in ("women", "girls"):
                            continue
                    elif req_gender in ("women", "girls"):
                        if cand_g in ("men", "boys"):
                            continue
                    elif req_gender in ("kids", "baby", "boys"):
                        # For kids queries, strictly return kids/boys/girls items only
                        if not c.kid_item[idx] and cand_g not in ("unisex", "boys", "girls", "kids", "toddler", "baby"):
                            continue
                # If brand requested and strict, it MUST match the brand
                if brand_strict and brand_mask is not None and not brand_mask[idx]:
                    continue
                # If color requested and strict, it MUST match the color
                if color_strict and color_mask is not None and not color_mask[idx]:
                    continue
                # If material requested and strict, it MUST match the material
                if mat_strict and mat_mask is not None and not mat_mask[idx]:
                    continue
                # If negative exclusions active, drop violators (brand field OR title)
                if ex_b_masks and any(bm_ex[idx] for bm_ex in ex_b_masks):
                    continue
                if nlp_intent.excluded_colors and any(c.color_n[idx] == alnum(co) for co in nlp_intent.excluded_colors):
                    continue
                # If category/group requested, it MUST match the category/group
                if grp_mask is not None and not grp_mask[idx]:
                    continue
                # Price constraint check
                if parsed.get("max_price") and c.price[idx] > parsed["max_price"]:
                    # When hard_price is False, allow closest prices if price filter was relaxed
                    if not is_price_relaxed:
                        continue
                if parsed.get("min_price") and c.price[idx] < parsed["min_price"]:
                    if not is_price_relaxed:
                        continue

                # Storage constraint check (e.g. 256GB requested, don't return 128GB phones or phones with no 256GB)
                storage_m = re.search(r"\b(\d+)\s*(?:gb|tb)\b", q_clean.lower()) or re.search(r"\b(\d+)\s*(?:gb|tb)\b", q_norm.lower())
                if storage_m:
                    req_storage = storage_m.group(0).upper().replace(" ", "")
                    prod_text = (str(c.df.at[idx, "title"]) + " " + str(c.df.at[idx, "description"])).upper().replace(" ", "")
                    prod_st = re.search(r"\b(\d+)\s*(?:GB|TB)\b", prod_text)
                    if prod_st and prod_st.group(0).upper().replace(" ", "") != req_storage:
                        continue
                    if req_storage not in prod_text:
                        continue

                # General relevance check with calibrated cutoff for out-of-domain queries
                has_affinity = bool(nlp_intent.affinities and any(
                    c.cat_n[idx] == cat_norm(aff) or cat_norm(aff) in c.cat_n[idx]
                    for aff in nlp_intent.affinities
                ))
                has_struct = (
                    (grp_mask is not None and grp_mask[idx])
                    or (brand_mask is not None and brand_mask[idx])
                    or bool(flags)
                    or has_affinity
                )
                if has_struct:
                    is_relevant = True
                else:
                    cand_tok_set = set(toks(c.texts[idx]))
                    tok_hits = sum(1 for w in bm_content_toks if w in cand_tok_set)
                    if bm[idx] == 0.0:
                        is_relevant = (
                            (ce_raw[j] >= -2.5 if s.use_rerank else cos[idx] >= 0.55)
                            and (cos[idx] >= 0.35)
                        )
                    elif len(bm_content_toks) >= 2 and tok_hits <= 1:
                        is_relevant = (ce_raw[j] >= -3.0 if s.use_rerank else cos[idx] >= 0.48)
                    else:
                        is_relevant = (
                            (bm[idx] > 1.0)
                            or (cos[idx] >= s.cos_min)
                            or (s.use_rerank and ce_raw[j] >= -4.0)
                        )
                if is_relevant:
                    filtered_cand_order.append(j)

            effective_order = filtered_cand_order if filtered_cand_order else []
            order = effective_order[:k]
            top = [cand[j] for j in order]
        else:
            order = []
            top = []
        mflags = dict(flags)
        if p_in is not None:
            mflags["price"] = p_in
            core = mask.copy()
            for n, f in flags.items():
                if n != "gender":
                    core &= f
            if not (core & p_in).any():
                notes.append("no product matches all conditions inside price range — showing closest prices")
            elif int(p_in[top].sum()) < len(top):
                notes.append("products inside price range first, then closest prices")
        if grp_names:
            notes.append("product group: " + ", ".join(grp_names))
        if relaxed:
            notes.append("relaxed filters: " + ", ".join(relaxed))

        # Build intent summary for response
        intent_out = {k_: v for k_, v in parsed.items() if v is not None}
        if nlp_intent.recipient:
            intent_out["recipient"] = nlp_intent.recipient
        if nlp_intent.intents:
            intent_out["intents"] = nlp_intent.intents
        if nlp_intent.affinities:
            intent_out["affinities"] = nlp_intent.affinities[:5]  # cap for brevity
        if nlp_intent.target_gender:
            intent_out["gender"] = nlp_intent.target_gender

        rows = []
        for rank, (j, i) in enumerate(zip(order, top), 1):
            r = {col: c.df.at[i, col] for col in OUT_COLS}
            r.update(
                rank=rank,
                product_id=r["id"],
                price=float(r["price"]),
                currency="BDT",
                score=round(float(final[j]), 3),
                cosine=round(float(cos[i]), 3),
                bm25=round(float(bm[i]), 3),
                match=label[j],
                matched="+".join(n for n, f in mflags.items() if hasattr(f, '__getitem__') and f[i]),
                decision_explanation=f"Rank {rank} via RRF hybrid fusion ({label[j] or 'attribute match'})"
            )
            rows.append(r)
        return {
            "query": query,
            "normalized": q_clean,
            "corrections": nlp_intent.corrections,
            "intent": intent_out,
            "parsed_intent": intent_out,
            "source": source,
            "notes": notes,
            "results": rows,
        }


    def similar(self, product_id: str, k: int = 10) -> Dict:
        """'More like this': nearest products by embedding (same text model as search)."""
        i = self.cat.id_to_index.get(str(product_id))
        if i is None:
            return {"id": product_id, "results": [], "error": "unknown product id"}
        sims = self.cat.emb @ self.cat.emb[i]
        sims[i] = -1.0
        if k < len(sims):
            part = np.argpartition(-sims, k)[:k]
            top = part[np.argsort(-sims[part])]
        else:
            top = np.argsort(-sims)[:k]
        rows = []
        for rank, j in enumerate(top, 1):
            r = {col: self.cat.df.at[j, col] for col in OUT_COLS}
            r.update(rank=rank, price=float(r["price"]), similarity=round(float(sims[j]), 3))
            rows.append(r)
        return {"id": product_id, "results": rows}

    def suggest(self, prefix: str, k: int = 8) -> Dict:
        """Autocomplete from the catalog's own categories, brands and titles."""
        q = _qkey(prefix)
        if len(q) < 2:
            return {"prefix": prefix, "suggestions": []}
        hit = self.cache.get_json("suggest", sha(f"{self.cat.version}|{k}|{q}"))
        if hit is not None:
            return hit
        if not hasattr(self, "_sug"):
            from collections import Counter
            cnt = Counter()
            for col, w in (("category", 3), ("brand", 2), ("title", 1)):
                for v in self.cat.df[col]:
                    if v:
                        cnt[v.lower()] += w
            self._sug = list(cnt.items())
        scored = []
        for term, w in self._sug:
            if term.startswith(q):
                scored.append((0, -w, len(term), term))
            elif f" {q}" in term:
                scored.append((1, -w, len(term), term))
        out = {"prefix": prefix, "suggestions": [t[3] for t in sorted(scored)[:k]]}
        self.cache.set_json("suggest", sha(f"{self.cat.version}|{k}|{q}"), out, self.s.ttl_result)
        return out

    def warm(self, queries: List[str], k: int = 10) -> int:
        """Pre-fill the cache (intents, embeddings, results) with common queries."""
        n = 0
        for q in queries:
            if q.strip():
                self.search(q, k=k); n += 1
        return n

    def info(self) -> Dict:
        return {"products": self.cat.n, "catalog_version": self.cat.version, "embedder": self.qemb.embedder.name,
                "llm": getattr(self.llm, "model", "disabled") if self.llm.available else "disabled",
                "cache": self.cache.info(), "rerank": self.s.use_rerank}


def build_engine(settings: Optional[Settings] = None, df=None) -> SearchEngine:
    """One-call factory. Pass a DataFrame (standard columns) to skip CSV/DB loading."""
    s = settings or Settings.from_env()
    from .catalog import standardise
    df = standardise(df, s.column_map) if df is not None else load_dataframe(s)
    embedder = make_embedder(s)
    cache = Cache(s.redis_url, s.cache_prefix, s.memory_cache_items)
    catalog = Catalog(df, embedder, s.index_dir)
    eng = SearchEngine(catalog, embedder, cache, make_llm(s), s)
    if s.warm_file:
        with open(s.warm_file, encoding="utf-8") as f:
            log.info("warmed %d queries", eng.warm([ln.strip() for ln in f]))
    else:
        try:
            eng.search("laptop", k=1, use_cache=False)
        except Exception:
            pass
    return eng
