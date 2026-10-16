"""Cache layer: Redis when available, otherwise a bounded in-process LRU+TTL. Same interface either way.

Redis failures never break search: every call degrades to a cache miss.
"""
import hashlib
import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any, Optional

import numpy as np

log = logging.getLogger("lite_search.cache")


def sha(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:20]


class MemoryBackend:
    def __init__(self, max_items: int = 20000):
        self.max, self.d, self.lock = max_items, OrderedDict(), threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self.lock:
            item = self.d.get(key)
            if not item:
                return None
            exp, val = item
            if exp and exp < time.time():
                del self.d[key]
                return None
            self.d.move_to_end(key)
            return val

    def set(self, key: str, val: bytes, ttl: Optional[int]):
        with self.lock:
            self.d[key] = ((time.time() + ttl) if ttl else 0, val)
            self.d.move_to_end(key)
            while len(self.d) > self.max:
                self.d.popitem(last=False)

    def size(self) -> int:
        return len(self.d)

    def flush(self):
        with self.lock:
            self.d.clear()


class RedisBackend:
    def __init__(self, url: str):
        import redis  # optional dependency
        self.r = redis.Redis.from_url(url, socket_timeout=0.3, socket_connect_timeout=0.5)
        self.r.ping()

    def get(self, key):
        try:
            return self.r.get(key)
        except Exception as e:
            log.warning("redis get failed: %s", e)
            return None

    def set(self, key, val, ttl):
        try:
            self.r.set(key, val, ex=ttl)
        except Exception as e:
            log.warning("redis set failed: %s", e)

    def size(self):
        try:
            return self.r.dbsize()
        except Exception:
            return -1

    def flush(self):
        pass  # never FLUSHDB on a shared server; bump cache_prefix instead


class Cache:
    """Namespaced JSON / vector cache with hit-miss statistics."""

    def __init__(self, redis_url: Optional[str] = None, prefix: str = "ls", max_items: int = 20000):
        self.prefix = prefix
        self.backend_name = "memory"
        if redis_url:
            try:
                self.backend = RedisBackend(redis_url)
                self.backend_name = "redis"
            except Exception as e:
                log.warning("Redis unavailable (%s) - using in-memory cache", e)
                self.backend = MemoryBackend(max_items)
        else:
            self.backend = MemoryBackend(max_items)
        self.stats = {"hit": 0, "miss": 0}

    def _k(self, ns: str, key: str) -> str:
        return f"{self.prefix}:{ns}:{key}"

    # ---- json ----
    def get_json(self, ns: str, key: str) -> Any:
        raw = self.backend.get(self._k(ns, key))
        if raw is None:
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return json.loads(raw)

    def set_json(self, ns: str, key: str, value: Any, ttl: Optional[int] = None):
        self.backend.set(self._k(ns, key), json.dumps(value, ensure_ascii=False).encode(), ttl)

    # ---- vectors (float32) ----
    def get_vec(self, ns: str, key: str) -> Optional[np.ndarray]:
        raw = self.backend.get(self._k(ns, key))
        if raw is None:
            self.stats["miss"] += 1
            return None
        self.stats["hit"] += 1
        return np.frombuffer(raw, dtype="float32")

    def set_vec(self, ns: str, key: str, vec: np.ndarray, ttl: Optional[int] = None):
        self.backend.set(self._k(ns, key), np.asarray(vec, dtype="float32").tobytes(), ttl)

    def info(self) -> dict:
        tot = self.stats["hit"] + self.stats["miss"]
        return {"backend": self.backend_name, "items": self.backend.size(), **self.stats,
                "hit_rate": round(self.stats["hit"] / tot, 3) if tot else 0.0}
