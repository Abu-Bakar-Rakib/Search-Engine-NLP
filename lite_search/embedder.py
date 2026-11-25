"""Embedders. All return L2-normalised float32 arrays. Query embeddings are cached (Redis/memory)."""
import re
import zlib
from typing import List

import numpy as np

from .cache import Cache, sha


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, device: str = None, max_len: int = 128):
        from sentence_transformers import SentenceTransformer  # optional dependency
        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device or "cpu")
        self.model.max_seq_length = max_len  # products are title-first; long descriptions only cost time
        self.dim = self.model.get_sentence_embedding_dimension()
        # E5 models are trained with these prefixes and lose quality without them
        self.qprefix, self.dprefix = ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")

    def encode(self, texts: List[str], batch_size: int = 64, show_progress: bool = False, query: bool = False) -> np.ndarray:
        pre = self.qprefix if query else self.dprefix
        texts = [pre + t for t in texts] if pre else texts
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show_progress).astype("float32")


class HashingEmbedder:
    """Dependency-free bag-of-words hashing embedder. Used for tests and as an emergency fallback
    (no semantic generalisation, but BM25 + rules still work)."""

    def __init__(self, dim: int = 256):
        self.name, self.dim = f"hashing-{dim}", dim

    def encode(self, texts: List[str], batch_size: int = 0, show_progress: bool = False, query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"[a-z0-9\u0980-\u09FF]+", t.lower()):
                out[i, zlib.crc32(w.encode()) % self.dim] += 1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(n, 1e-9)


class CachedQueryEmbedder:
    def __init__(self, embedder, cache: Cache, ttl: int):
        self.embedder, self.cache, self.ttl = embedder, cache, ttl

    def encode_query(self, text: str) -> np.ndarray:
        key = f"{self.embedder.name}:{sha(text)}"
        v = self.cache.get_vec("emb", key)
        if v is None or len(v) != self.embedder.dim:
            v = self.embedder.encode([text], query=True)[0]
            self.cache.set_vec("emb", key, v, self.ttl)
        return v


def make_embedder(settings):
    if settings.embedder == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.emb_model, max_len=settings.emb_max_len)
