"""Product catalog: loading (CSV / SQL), normalisation, sparse BM25, persisted embeddings, fast filter arrays."""
import hashlib
import json
import logging
import os
from collections import Counter
from typing import Optional

import re
import numpy as np
import pandas as pd
import scipy.sparse as sp

from .text import alnum, cat_norm, is_missing, nfc, toks

log = logging.getLogger("lite_search.catalog")

STD_COLS = ["id", "title", "category", "brand", "color", "material", "gender", "sizes", "price", "description",
            "macro_category", "image_url", "source_url"]
GENDER_FIX = {"male": "men", "man": "men", "mens": "men", "men's": "men", "female": "women", "woman": "women",
              "womens": "women", "women's": "women", "ladies": "women", "boy": "boys", "girl": "girls",
              "kids": "unisex", "": "unisex"}


# ---------------------------------------------------------------- loading
def load_dataframe(settings) -> pd.DataFrame:
    if settings.database_url:
        from sqlalchemy import create_engine, text  # optional dependency
        eng = create_engine(settings.database_url, pool_pre_ping=True)
        with eng.connect() as c:
            df = pd.read_sql(text(settings.sql_query), c)
    elif settings.catalog_csv:
        csv_path = settings.catalog_csv
        if not os.path.exists(csv_path):
            for candidate in (os.path.join(os.path.dirname(__file__), "..", csv_path),
                              os.path.join("lite_search", csv_path)):
                if os.path.exists(candidate):
                    csv_path = candidate
                    break
        if csv_path.endswith(".jsonl") or csv_path.endswith(".json"):
            records = []
            with open(csv_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
                        if settings.max_rows and len(records) >= settings.max_rows:
                            break
            df = pd.DataFrame(records)
        else:
            df = pd.read_csv(csv_path, nrows=settings.max_rows)
    else:
        raise ValueError("Set CATALOG_CSV or DATABASE_URL (see README).")
    raw_cols = list(df.columns)
    out = standardise(df, settings.column_map)
    out.attrs["raw_columns"] = raw_cols
    return out


def standardise(df: pd.DataFrame, column_map=None) -> pd.DataFrame:
    df = df.rename(columns=column_map or {}).copy()
    n = len(df)
    out = pd.DataFrame(index=range(n))
    df = df.reset_index(drop=True)
    out["id"] = df["id"].astype(str) if "id" in df else [f"P{i + 1:06d}" for i in range(n)]
    for c, default in (("title", ""), ("category", ""), ("brand", ""), ("color", ""), ("material", ""),
                       ("gender", "unisex"), ("sizes", ""), ("description", ""),
                       ("macro_category", ""), ("image_url", ""), ("source_url", "")):
        s = df[c] if c in df else pd.Series([default] * n)
        out[c] = s.fillna(default).astype(str).str.strip()
        out.loc[out[c].map(is_missing), c] = default
    out["gender"] = out["gender"].str.lower().replace(GENDER_FIX)
    out["category"] = out["category"].str.lower()
    out["price"] = pd.to_numeric(df["price"] if "price" in df else 0, errors="coerce").fillna(0).astype("float64")
    return out[STD_COLS]


def row_text(r) -> str:
    parts = [r.title]
    macro = getattr(r, "macro_category", "")
    if macro and macro != r.category:
        parts.append(f"Category: {macro}")
    if r.category:
        parts.append(f"Category: {r.category}")
    if r.brand and r.brand.lower() != "no brand":
        parts.append(f"Brand: {r.brand}")
    if getattr(r, "color", ""):
        parts.append(f"Color: {r.color}")
    if getattr(r, "material", ""):
        parts.append(f"Material: {r.material}")
    if getattr(r, "gender", "") and r.gender not in ("", "unisex"):
        parts.append(f"For {r.gender}")
    if r.description:
        parts.append(str(r.description)[:1000])
    return ". ".join(parts)


# ---------------------------------------------------------------- catalog
class Catalog:
    def __init__(self, df: pd.DataFrame, embedder, index_dir: Optional[str] = None):
        self.df = df.reset_index(drop=True)
        self.n = len(self.df)
        self.texts = [row_text(r) for r in self.df.itertuples()]
        self.version = hashlib.sha1("\n".join(self.texts).encode()).hexdigest()[:10]
        self.emb = self._embeddings(embedder, index_dir)
        self._build_bm25()
        self._build_arrays()
        self._size_cache = {}

    # ---- embeddings, re-using vectors of unchanged products between runs ----
    def _embeddings(self, embedder, index_dir):
        hashes = [hashlib.sha1(t.encode()).hexdigest()[:16] for t in self.texts]
        known = {}
        path = None
        if index_dir:
            if not os.path.exists(index_dir):
                for candidate in (os.path.join(os.path.dirname(__file__), "..", index_dir),
                                  os.path.join("lite_search", index_dir)):
                    if os.path.exists(candidate):
                        index_dir = candidate
                        break
            os.makedirs(index_dir, exist_ok=True)
            tag = hashlib.sha1(embedder.name.encode()).hexdigest()[:8]
            path = os.path.join(index_dir, f"emb_{tag}.npz")
            if os.path.exists(path):
                try:
                    z = np.load(path, allow_pickle=False)
                    known = dict(zip(z["h"].tolist(), z["v"]))
                except Exception as e:
                    log.warning("could not read %s: %s", path, e)
        missing = [i for i, h in enumerate(hashes) if h not in known]
        if missing:
            log.info("encoding %d/%d products", len(missing), self.n)
            vecs = embedder.encode([self.texts[i] for i in missing], batch_size=64, show_progress=self.n > 2000)
            for i, v in zip(missing, vecs):
                known[hashes[i]] = v
            if path:
                keep = {h: known[h] for h in set(hashes)}        # drop vectors of removed products
                np.savez(path, h=np.array(list(keep)), v=np.stack(list(keep.values())))
        return np.stack([known[h] for h in hashes]).astype("float32")

    # ---- sparse BM25 (scales to 100k+ products) ----
    def _build_bm25(self, k1=1.5, b=0.75):
        docs = [toks(t) for t in self.texts]
        df = Counter(w for d in docs for w in set(d))
        self.doc_freq = df
        self.vocab = {w: i for i, w in enumerate(sorted(df))}
        self.dl = np.array([len(d) for d in docs], dtype="float32")
        self.avg = float(self.dl.mean()) if self.n else 1.0
        rows, cols, vals = [], [], []
        for i, d in enumerate(docs):
            for w, c in Counter(d).items():
                rows.append(i); cols.append(self.vocab[w]); vals.append(c)
        self.tf = sp.csc_matrix((vals, (rows, cols)), shape=(self.n, len(self.vocab)), dtype="float32")
        dfv = np.array([df[w] for w in sorted(df)], dtype="float32")
        self.idf = np.log(1 + (self.n - dfv + 0.5) / (dfv + 0.5))
        self.k1, self.b = k1, b

    def bm25(self, query: str) -> np.ndarray:
        out = np.zeros(self.n)
        for w in set(toks(query)):
            j = self.vocab.get(w)
            if j is None:
                continue
            s, e = self.tf.indptr[j], self.tf.indptr[j + 1]
            r, tf = self.tf.indices[s:e], self.tf.data[s:e]
            out[r] += self.idf[j] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[r] / self.avg))
        return out

    # ---- arrays for filters / boosts (string work is done per UNIQUE value, not per product) ----
    def _build_arrays(self):
        d = self.df
        self.price = d["price"].to_numpy("float64")
        self.gender = d["gender"].to_numpy()
        self.has_real_gender = bool((self.gender != "unisex").any())
        self.brand_n = np.array([alnum(x) for x in d["brand"]])
        self.color_n = np.array([alnum(x) for x in d["color"]])
        self.cat_n = np.array([cat_norm(x) for x in d["category"]])
        self.cat_uni, self.cat_code = np.unique(self.cat_n, return_inverse=True)
        self.macro_lower = [nfc(m).lower() for m in d["macro_category"]] if "macro_category" in d else [""] * self.n
        self.macro_n = np.array([cat_norm(x) for x in self.macro_lower])
        mat_n = np.array([alnum(x) for x in d["material"]])
        self.mat_uni, self.mat_code = np.unique(mat_n, return_inverse=True)
        self.sizes = [{x.strip().upper() for x in str(s).split(",") if x.strip()} for s in d["sizes"]]
        self.id_to_index = {str(pid): i for i, pid in enumerate(d["id"])}
        self.cat_lower = [nfc(c).lower() for c in d["category"]]
        self.title_lower = [nfc(t).lower() for t in d["title"]]
        self.desc_lower = [nfc(desc).lower() for desc in d["description"]]
        self.brand_lower = [nfc(b).lower() for b in d["brand"]]

        # Precomputed masks for high-frequency keyword checks
        _ear_re = re.compile(r"earbud|earphone|airpod|spacebuds|buds|tws|ইয়ারবাড|ইয়ারফোন|ইয়ারবাড|ইয়ারফোন", re.I)
        self.is_earbud = np.array([
            "earbud" in cat or "earphone" in cat or bool(_ear_re.search(t)) or bool(_ear_re.search(desc))
            for cat, t, desc in zip(self.cat_lower, self.title_lower, self.desc_lower)
        ], dtype=bool)

        _acc_re = re.compile(
            r"\b(?:sim\s*pin|gimbal|stabilizer|ring\s*light|teleprompter|phone\s*case|mobile\s*case|"
            r"back\s*case|back\s*cover|phone\s*cover|mobile\s*cover|screen\s*protector|tempered\s*glass|"
            r"phone\s*holder|mobile\s*holder|phone\s*stand|charging\s*stand|s\s*pen|adapter|charger|cable)\b|"
            r"(?:\bcase\b.*\bfor\b|\bcover\b.*\bfor\b|\bfor\s+iphone\b|\bfor\s+samsung\b|\bfor\s+realme\b|\bfor\s+motorola\b)",
            re.I
        )
        self.is_phone_accessory = np.array([
            bool(_acc_re.search(t)) or any(k in cat for k in ("case", "cover", "charger", "adapter", "cable", "studio equipment", "protector"))
            for cat, t in zip(self.cat_lower, self.title_lower)
        ], dtype=bool)

        _anc_re = re.compile(r"noise\s*cancel|anc|active\s*noise", re.I)
        self.is_anc = np.array([
            bool(_anc_re.search(t)) or bool(_anc_re.search(desc))
            for t, desc in zip(self.title_lower, self.desc_lower)
        ], dtype=bool)

        self.is_bike = np.array([
            "bike" in t or "bicycle" in t for t in self.title_lower
        ], dtype=bool)

        self.is_mom_fit = np.array([
            "mom fit" in t or "mom high" in t for t in self.title_lower
        ], dtype=bool)

        from .lexicon import KID_TITLE
        _toy_cats = {"toy", "building blocks", "action figure", "doll", "board game", "puzzle", "educational toy"}
        self.kid_item = np.array([
            (c in _toy_cats) or (g in ("boys", "girls")) or bool(KID_TITLE & set(t.lower().split()))
            for c, g, t in zip(d["category"], d["gender"], d["title"])])


    def size_mask(self, size: str) -> np.ndarray:
        size = str(size).upper()
        if size not in self._size_cache:
            self._size_cache[size] = np.array([(not s) or (size in s) for s in self.sizes])
        return self._size_cache[size]

    def category_mask(self, categories) -> np.ndarray:
        cats_norm = [cat_norm(c) for c in categories if c]
        if not cats_norm:
            return np.zeros(self.n, dtype=bool)
        cat_set = set(cats_norm)
        uni_hit = np.array([
            (u in cat_set) or any(re.search(rf"(?<![a-z0-9]){re.escape(cn)}(?![a-z0-9])", u) for cn in cats_norm if len(cn) >= 3)
            for u in self.cat_uni
        ], dtype=bool)
        return uni_hit[self.cat_code]

    def vocab_values(self, col: str, min_len=2, limit=5000):
        c = Counter(v.lower() for v in self.df[col] if len(v) >= min_len and not is_missing(v))
        return [v for v, _ in c.most_common(limit)]
