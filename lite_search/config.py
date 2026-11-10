"""All tunables in one place. Every field can be overridden by an environment variable (see from_env)."""
import os
from dataclasses import dataclass, field, fields
from typing import Dict, Optional


def load_dotenv(path: str = ".env"):
    """Tiny .env reader (no dependency). Real environment variables win over the file."""
    if not os.path.exists(path):
        pkg_env = os.path.join(os.path.dirname(__file__), "..", ".env")
        sub_env = os.path.join("lite_search", path)
        if os.path.exists(pkg_env):
            path = pkg_env
        elif os.path.exists(sub_env):
            path = sub_env
        else:
            return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            v = v.split(" #")[0].strip().strip('"').strip("'")
            os.environ.setdefault(k.strip(), v)


@dataclass
class Settings:
    # ---- data source (one of these) ----
    catalog_csv: Optional[str] = None
    database_url: Optional[str] = None          # e.g. postgresql://user:pw@host/db
    sql_query: str = "SELECT * FROM products"
    # map the company's column names -> standard names (title, category, brand, color, material,
    # gender, sizes, price, description, id). Example: {"product_name": "title", "cat": "category"}
    column_map: Dict[str, str] = field(default_factory=dict)
    max_rows: Optional[int] = None
    lexicon_file: Optional[str] = None          # JSON: {"aliases": {...}, "groups": {name: [[triggers], [categories]]}}
    warm_file: Optional[str] = None             # text file, one common query per line, pre-cached at start-up

    # ---- models ----
    emb_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedder: str = "sentence-transformers"     # "sentence-transformers" | "hashing" (no download, tests)
    index_dir: str = ".lite_search_index"       # product embeddings are persisted here
    use_rerank: bool = True                     # cross-encoder: more accurate, ~+40 ms and +90 MB RAM
    ce_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # ---- cache ----
    redis_url: Optional[str] = None             # e.g. redis://localhost:6379/0 ; None = in-process cache
    cache_prefix: str = "ls"
    ttl_intent: int = 7 * 86400
    ttl_embedding: int = 30 * 86400
    ttl_result: int = 600                       # keep short: prices/stock change
    memory_cache_items: int = 20000

    # ---- optional LLM fallback (only used when rules find nothing AND an API key exists) ----
    llm_provider: str = "none"                  # "none" | "anthropic"
    llm_model: str = "claude-haiku-5-5"
    llm_timeout: float = 4.0

    # ---- ranking ----
    alpha: float = 0.6            # semantic weight (1-alpha = BM25)
    boost_scale: float = 1.0
    pool: int = 20                # candidates kept for tiers / rerank
    ce_weight: float = 0.5
    group_tier: float = 50.0      # same product type as the query always ranks first
    price_tier: float = 6.0       # inside the price range
    price_near: float = 3.0       # outside the range, closer = higher
    kid_penalty: float = 30.0
    hard_price: bool = False
    cos_min: float = 0.30         # confidence gate for the "closest products" fallback
    ce_min: float = -8.0
    min_results: int = 3          # relax size/gender filters when fewer products survive

    # ---- server ----
    cors_origins: Optional[str] = None          # "*" or "https://shop.example.com,https://admin.example.com"
    reload_minutes: Optional[int] = None        # re-read the product DB every N minutes (None = only POST /reload)
    port: int = 8080
    host: str = "0.0.0.0"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        s = cls()
        for f in fields(cls):
            v = os.environ.get(f.name.upper())
            if v is None or f.name == "column_map":
                continue
            t = f.type
            if t in (bool, "bool"):
                v = v.lower() in ("1", "true", "yes")
            elif t in (int, "int"):
                v = int(v)
            elif t in (float, "float"):
                v = float(v)
            elif "Optional[int]" in str(t):
                v = int(v)
            setattr(s, f.name, v)
        cm = os.environ.get("COLUMN_MAP")  # "product_name:title,cat:category"
        if cm:
            s.column_map = dict(p.split(":", 1) for p in cm.split(","))
        return s
