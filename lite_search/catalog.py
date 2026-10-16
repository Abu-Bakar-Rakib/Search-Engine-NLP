"""Product catalog: loading (CSV / SQL), normalisation, sparse BM25, persisted embeddings, fast filter arrays."""
import hashlib
import logging
import os
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

from .text import alnum, cat_norm, is_missing, toks

log = logging.getLogger("lite_search.catalog")

STD_COLS = ["id", "title", "category", "brand", "color", "material", "gender", "sizes", "price", "description"]
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
                       ("gender", "unisex"), ("sizes", ""), ("description", "")):
        s = df[c] if c in df else pd.Series([default] * n)
        out[c] = s.fillna(default).astype(str).str.strip()
        out.loc[out[c].map(is_missing), c] = default
    out["gender"] = out["gender"].str.lower().replace(GENDER_FIX)
    out["category"] = out["category"].str.lower()
    out["price"] = pd.to_numeric(df["price"] if "price" in df else 0, errors="coerce").fillna(0).astype("float64")
    return out[STD_COLS]


def row_text(r) -> str:
    return (f"{r.title}. Category: {r.category}. Brand: {r.brand}. Color: {r.color}. Material: {r.material}. "
            f"For {r.gender}. {r.description}")


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
        self.brand_n = np.array([alnum(x) for x in d["brand"]])
        self.color_n = np.array([alnum(x) for x in d["color"]])
        self.cat_n = np.array([cat_norm(x) for x in d["category"]])
        self.cat_uni, self.cat_code = np.unique(self.cat_n, return_inverse=True)
        mat_n = np.array([alnum(x) for x in d["material"]])
        self.mat_uni, self.mat_code = np.unique(mat_n, return_inverse=True)
        self.sizes = [{x.strip().upper() for x in str(s).split(",") if x.strip()} for s in d["sizes"]]
        from .lexicon import KID_TITLE
        self.kid_item = np.array([
            (c == "toy") or (g in ("boys", "girls")) or bool(KID_TITLE & set(t.lower().split()))
            for c, g, t in zip(d["category"], d["gender"], d["title"])])

    def size_mask(self, size: str) -> np.ndarray:
        size = str(size).upper()
        if size not in self._size_cache:
            self._size_cache[size] = np.array([(not s) or (size in s) for s in self.sizes])
        return self._size_cache[size]

    def category_mask(self, categories) -> np.ndarray:
        return np.isin(self.df["category"].to_numpy(), list(categories))

    def vocab_values(self, col: str, min_len=2, limit=5000):
        c = Counter(v.lower() for v in self.df[col] if len(v) >= min_len and not is_missing(v))
        return [v for v, _ in c.most_common(limit)]
