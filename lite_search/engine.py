"""Search orchestration:  cache -> normalise -> RULES -> (LLM) -> EMBEDDING+BM25 ranking with tiers -> cache."""
import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np

from . import lexicon as L
from .cache import Cache, sha
from .catalog import Catalog, load_dataframe
from .config import Settings
from .embedder import CachedQueryEmbedder, make_embedder
from .llm import make_llm
from .parser import RuleParser, has_signal, retrieval_query, validate_intent
from .text import alnum, cat_norm, minmax

log = logging.getLogger("lite_search")
ATTR_W = {"category": 2.5, "brand": 16.0, "color": 8.0, "material": 8.0, "gender": 0.6}
BOOST = {"category": 0.15, "brand": 0.20, "color": 0.10, "material": 0.10}
RELAX_ORDER = ["size", "gender", "price"]
OUT_COLS = ["id", "title", "category", "brand", "color", "material", "gender", "sizes", "price"]


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

    def _audience(self, q, p):
        toks = set(q.lower().split())
        if p.get("gender") in ("boys", "girls") or (toks & L.KID_WORDS):
            return -self.s.kid_penalty * (~self.cat.kid_item)
        if p.get("gender") in ("women", "men"):
            return -self.s.kid_penalty * self.cat.kid_item
        return np.zeros(self.cat.n)

    def _rerank(self, q_clean, cand):
        if self._ce is None:
            from sentence_transformers import CrossEncoder
            self._ce = CrossEncoder(self.s.ce_model, device="cpu")
        return np.asarray(self._ce.predict([(q_clean, self.cat.texts[i]) for i in cand], show_progress_bar=False))

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
                hit["cached"], hit["took_ms"] = True, round((time.perf_counter() - t0) * 1000, 2)
                return hit
        out = self._search(query, k)
        out["took_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        out["cached"] = False
        if use_cache:
            self.cache.set_json("result", rkey, out, self.s.ttl_result)
        return out

    def _search(self, query: str, k: int) -> Dict:
        s, c = self.s, self.cat
        q_norm, note = self._normalize(query)
        notes = [note] if note else []
        grp_mask, grp_names = self.parser.detect_groups(q_norm)
        parsed, source = self._intent(q_norm, grp_mask)
        if grp_mask is None and parsed.get("category"):
            cm = c.cat_n == cat_norm(parsed["category"])
            if cm.any():
                grp_mask, grp_names = cm, [str(parsed["category"])]

        q_clean = retrieval_query(q_norm)
        cos = c.emb @ self.qemb.encode_query(q_clean)
        bm = c.bm25(q_clean)
        active, relaxed = [f for f in RELAX_ORDER if s.hard_price or f != "price"], []
        while True:
            mask = self._mask(parsed, active)
            if mask.sum() >= s.min_results or not active:
                break
            relaxed.append(active.pop(0))
        base = np.where(mask, s.alpha * minmax(cos) + (1 - s.alpha) * minmax(bm) + s.boost_scale * self._boost(parsed),
                        -1e9)
        bonus, flags = self._attr_bonus(parsed)
        p_bonus, p_in = self._price_tier(parsed)
        aud = self._audience(q_norm, parsed)
        grp_bonus = s.group_tier * grp_mask if grp_mask is not None else np.zeros(c.n)
        tiers = grp_bonus + bonus + p_bonus + aud

        sel = np.where(base > -1e8, base + tiers, base)
        pool = max(s.pool, k)
        cand = [int(i) for i in np.argpartition(-sel, min(pool, c.n - 1))[:pool] if sel[i] > -1e8]
        cand.sort(key=lambda i: -sel[i])
        if not cand:
            cand = [int(i) for i in np.argsort(-cos)[:pool]]
            notes.append("filters removed, showing nearest products by meaning")

        ce_raw = self._rerank(q_clean, cand) if (s.use_rerank and len(cand) > 1) else np.zeros(len(cand))
        confident = (grp_mask is not None) or bool(flags) or (p_in is not None) or bm[cand].max() > 0 \
            or cos[cand].max() >= s.cos_min or (s.use_rerank and ce_raw.max() >= s.ce_min)
        if confident:
            if s.use_rerank and len(cand) > 1:
                blend = (1 - s.ce_weight) * minmax(base[cand]) + s.ce_weight * minmax(ce_raw)
            else:
                blend = minmax(base[cand]) if len(cand) > 1 else np.ones(1)
            final = blend + tiers[cand]
            label = (["intent" if grp_mask[i] else "related" for i in cand] if grp_mask is not None
                     else [""] * len(cand))
        else:
            final = (minmax(base[cand]) if len(cand) > 1 else np.ones(1)) + p_bonus[cand] + aud[cand]
            label = ["closest"] * len(cand)
            notes.append(f"No exact match for '{query}' - showing the closest products")

        order = np.argsort(-final)[:k]
        top = [cand[j] for j in order]
        mflags = dict(flags)
        if p_in is not None:
            mflags["price"] = p_in
            core = mask.copy()
            for n, f in flags.items():
                if n != "gender":
                    core &= f
            if not (core & p_in).any():
                notes.append("no product matches all conditions inside the price range - showing the closest prices first")
            elif int(p_in[top].sum()) < len(top):
                notes.append("products inside the price range first, then the closest prices")
        if grp_names:
            notes.append("product group: " + ", ".join(grp_names))
        if relaxed:
            notes.append("relaxed filters: " + ", ".join(relaxed))

        rows = []
        for rank, (j, i) in enumerate(zip(order, top), 1):
            r = {col: c.df.at[i, col] for col in OUT_COLS}
            r.update(rank=rank, price=float(r["price"]), score=round(float(final[j]), 3),
                     cosine=round(float(cos[i]), 3), bm25=round(float(bm[i]), 3), match=label[j],
                     matched="+".join(n for n, f in mflags.items() if f[i]))
            rows.append(r)
        return {"query": query, "normalized": q_norm, "intent": {k_: v for k_, v in parsed.items() if v is not None},
                "source": source, "notes": notes, "results": rows}

    def similar(self, product_id: str, k: int = 10) -> Dict:
        """'More like this': nearest products by embedding (same text model as search)."""
        ids = self.cat.df["id"].to_numpy()
        hit = np.where(ids == str(product_id))[0]
        if not len(hit):
            return {"id": product_id, "results": [], "error": "unknown product id"}
        i = int(hit[0])
        sims = self.cat.emb @ self.cat.emb[i]
        sims[i] = -1
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
    return eng
