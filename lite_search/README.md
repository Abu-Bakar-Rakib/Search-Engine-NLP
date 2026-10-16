# lite_search

Lightweight product search for an e-commerce backend. Runs on a plain CPU PC (no GPU, no big LLM).
Ported from `hybrid_updated_3.ipynb`; same ranking logic, now a reusable package + HTTP API with a Redis cache.

## How a query is answered

```
query ─► result cache ─► normalise (Banglish dictionary; LLM translate only if needed, cached)
      ─► 1. RULES   price / size / gender / category / brand / colour / material   (microseconds)
      ─► 2. LLM     only if rules found nothing AND an API key is set (cached, 4 s timeout, circuit breaker)
      ─► 3. EMBEDDINGS + BM25 over product vectors  (always used for ranking; sole signal if 1 and 2 found nothing)
      ─► tiers: product group > attribute matches > price range (soft) > audience   ─► cache ─► JSON
```
Response field `source` tells which tier understood the query: `rules`, `llm`, `embedding` (`…(cached)` when the
intent came from cache).

## What is cached (Redis, or in-process memory if Redis is absent/down)

| key | content | TTL |
|---|---|---|
| `ls:emb:*` | query embeddings | 30 d |
| `ls:intent:*` | parsed intent per normalised query ("common words") | 7 d |
| `ls:norm:*` | Banglish→English normalisation / LLM translation | 7 d |
| `ls:result:*` | final top-k result (includes catalog version, so a product change invalidates it) | 10 min |

Product embeddings are stored on disk in `INDEX_DIR` (not Redis); on restart/reload only new or changed products are
re-encoded. Redis errors never break search (they degrade to cache misses). Several server processes can share one Redis.

## Quick start (no Redis, no model download)

```bash
pip install -e .[api,dev]
CATALOG_CSV=data/demo_products.csv EMBEDDER=hashing python -m lite_search "red nike shoes under 5000"
CATALOG_CSV=data/demo_products.csv EMBEDDER=hashing python -m lite_search --serve      # http://localhost:8000/docs
pytest
```
`EMBEDDER=hashing` is for tests only (no semantic understanding). For real use install `sentence-transformers`
(default model `paraphrase-multilingual-MiniLM-L12-v2`, ~120 MB, handles English + Bangla script) and drop that variable.

## Server PC setup

```bash
cp .env.example .env            # edit data source + REDIS_URL
docker compose up -d            # redis + search API
# or without Docker:  redis-server &  ;  pip install -r requirements.txt && pip install -e . ;  python -m lite_search --serve
```

## Connect the company database

Set `DATABASE_URL` (SQLAlchemy URL; install the DB driver, e.g. `psycopg2-binary`, `pymysql`, `pyodbc`) and a query
that returns the product table. Map the company's column names to ours with `COLUMN_MAP`:

```
DATABASE_URL=postgresql://user:pass@host/shop
SQL_QUERY=SELECT product_id, name, cat, brand, colour, price, description FROM products WHERE active = 1
COLUMN_MAP=product_id:id,name:title,cat:category,colour:color
```
Standard columns: `id, title, category, brand, color, material, gender, sizes (comma list), price, description`.
Missing columns are fine (only `title` really matters). After the catalog changes call `POST /reload`
(or restart); unchanged products are not re-embedded. Use a **read-only** DB user.

## Testing on a new company database (checklist)

1. `pip install -r requirements.txt && pip install -e .` plus the DB driver (`psycopg2-binary`, `pymysql`, `pyodbc`, ...).
2. Copy `.env.example` to `.env` and set `DATABASE_URL`, `SQL_QUERY`, `COLUMN_MAP` (the app reads `.env` itself).
3. **`python -m lite_search --check`** - connects, shows which columns were found, fill-rate per column, price range,
   top categories/brands, sample rows and WARNINGS (e.g. "price mostly 0 - map your price column"). Fix the mapping until clean.
4. `python -m lite_search "red shirt under 1000"` - try real queries from the terminal.
5. `python -m lite_search --serve` - give the backend team `http://<server>:8000/docs`.
6. Add the shop's own words to `LEXICON_FILE` (aliases/groups) and common queries to `WARM_FILE`.

Nothing in the code is specific to one shop: the vocabulary (brands, colours, categories) is learned from the catalog.
Only `lexicon.py` (Bangla/Banglish words, product groups) is data you may extend, and it can be overridden by JSON.

## API (call from the backend)

* `GET /search?q=black nike shoes under 5000&k=10` or `POST /search {"query": "...", "k": 10}`
* `GET /suggest?q=runn` autocomplete from the catalog's categories, brands and titles
* `GET /similar/{product_id}?k=10` "more like this" (embedding neighbours)
* `GET /health` (product count, cache backend + hit rate), `POST /reload` (re-sync with the DB)
* `RELOAD_MINUTES=30` re-syncs automatically; `CORS_ORIGINS` only if a browser calls the API directly
* Optional auth: set `SEARCH_API_KEY`, backend sends header `X-API-Key`.

Result row: `id, title, category, brand, color, material, gender, sizes, price, rank, score, cosine, bm25, match, matched`.
Your backend should use the returned `id`s to load full product data/prices/stock from its own DB.

As a library: `from lite_search import build_engine; eng = build_engine(); eng.search("juta", k=5)`.

## Optional LLM fallback
`LLM_PROVIDER=anthropic` + `ANTHROPIC_API_KEY` (+ `LLM_MODEL`). Called only for queries the rules cannot read (and for
Banglish words the dictionary does not know). Its output is validated against the query text, results are cached, and
after 3 failures it pauses for 60 s. Without a key the system works fully on tiers 1 and 3.

## Tuning / extending
* Add Banglish words, product groups: JSON file via `LEXICON_FILE` (`{"aliases": {"jama": "shirt"}, "groups": {...}}`).
* Pre-fill the cache with common queries: `WARM_FILE=data/common_queries.txt`.
* Ranking weights are fields of `Settings` (`ALPHA`, `GROUP_TIER`, `PRICE_TIER`, `HARD_PRICE`, ...), all env-overridable.
* `USE_RERANK=true` adds the cross-encoder from the notebook (more accurate, slower, +~90 MB RAM).
* Memory: ≈ 1.5 KB per product for 384-d vectors (50k products ≈ 80 MB). Measured on 50k products (hashing embedder):
  5-9 ms per uncached-result query, ~300 MB process RSS.

## Layout
`catalog.py` data + sparse BM25 + persisted embeddings · `parser.py` rules · `lexicon.py` Banglish/groups ·
`llm.py` optional API · `cache.py` Redis/memory · `engine.py` pipeline · `server.py` API · `tests/`
