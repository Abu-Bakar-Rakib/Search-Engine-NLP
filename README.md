# lite_search

<div align="center">

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Friendly-EE4C2C.svg?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**Lightweight, CPU-first hybrid product search for e-commerce backends.**  
Runs efficiently on commodity hardware without requiring expensive GPUs or massive LLMs.
</div>

---

## ⚡ Highlights

* **Pure CPU Execution**: Optimized for standard cloud instances or PCs (~300 MB process RSS, no GPU needed).
* **3-Tier Hybrid Pipeline**:
  1. **Rules (Microseconds)**: Deterministic parsing for price bounds, brands, colors, categories, materials, and sizes.
  2. **BM25 + Dense Embeddings**: Dual lexical + semantic ranking over product vectors.
  3. **Optional LLM Fallback**: Engages only when queries have no rule match (cached with circuit-breaker protection).
* **Banglish & Bangla Support**: Built-in dictionary and normalizer handling phonetic spelling (e.g., *juta*, *panjabi*, *jama*).
* **Multi-Level Caching**: Works seamlessly with **Redis** or auto-degrades to an internal **in-memory LRU cache**.
* **Ready-to-Use HTTP API**: Built with FastAPI, including interactive Swagger docs (`/docs`) and a built-in test Web UI (`/`).
* **Pluggable Data Sources**: Connect to CSV files or relational databases (PostgreSQL, MySQL, SQLite, etc.) via SQLAlchemy.

---

## 🔍 How a Query is Answered

```text
Query ─► Result Cache ─► Normalise (Banglish dictionary / phonetic rules)
       ─► 1. RULES       Price, size, gender, category, brand, colour, material (microseconds)
       ─► 2. LLM         Fallback only if rules find nothing & API key configured (cached, 4s timeout)
       ─► 3. EMBEDDINGS  BM25 + SentenceTransformer vectors (ranks catalog matches)
       ─► Tiers Ranking  Product Group > Attribute Matches > Soft Price > Audience
       ─► Cache Store ─► Response JSON
```

---

## 🚀 Quick Start

### 1. Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Abu-Bakar-Rakib/Lite-search.git
cd Lite-search

pip install -r requirements.txt
pip install -e .
```

### 2. Configure Environment

Copy `.env.example` to `.env`:

```bash
cp .env.example .env    # On Windows: copy .env.example .env
```

*(By default, `.env` points to `data/demo_products.csv` and uses in-memory caching.)*

### 3. Verify Catalog & Data Mapping

Run the diagnostic tool to inspect your catalog fill rates and warnings:

```bash
python -m lite_search --check
```

### 4. Run Search Queries in Terminal

* **Single Query**:
  ```bash
  python -m lite_search "red nike shoes under 5000"
  ```
* **Interactive Terminal Prompt (REPL)**:
  ```bash
  python -m lite_search --repl
  ```
* **Fast Mode (Instant execution without deep model download)**:
  ```bash
  # Linux/macOS:
  EMBEDDER=hashing python -m lite_search "red nike shoes"

  # Windows PowerShell:
  $env:EMBEDDER="hashing"; python -m lite_search "red nike shoes"
  ```

---

## 🌐 HTTP API Server

Start the API server:

```bash
python -m lite_search --serve --host 0.0.0.0 --port 8000
# or with uvicorn:
uvicorn lite_search.server:app --host 0.0.0.0 --port 8000
```

- **Interactive API Documentation (Swagger)**: `http://localhost:8000/docs`
- **Interactive Web Search UI**: `http://localhost:8000/`

### Docker Support

Run with Redis and Search API containerized:

```bash
docker compose up -d
```

---

## 📡 API Reference

### 1. Search Products
`GET /search?q={query}&k={limit}` or `POST /search`

```bash
curl "http://localhost:8000/search?q=nike%20shoes%20under%205000&k=3"
```

**Response**:
```json
{
  "query": "nike shoes under 5000",
  "normalized": "nike shoes under 5000",
  "intent": {
    "brand": "nike",
    "max_price": 5000
  },
  "source": "rules",
  "notes": [
    "product group: footwear"
  ],
  "results": [
    {
      "id": "P064",
      "title": "Nike Air Zoom Winflo Running Shoe",
      "category": "running shoes",
      "brand": "Nike",
      "color": "black",
      "material": "mesh",
      "gender": "men",
      "sizes": "7,8,9,10",
      "price": 4600.0,
      "rank": 1,
      "score": 72.58,
      "cosine": 0.369,
      "bm25": 6.964,
      "match": "intent",
      "matched": "brand+price"
    }
  ],
  "took_ms": 45.89,
  "cached": false
}
```

### 2. Autocomplete Suggestions
`GET /suggest?q={prefix}&k={limit}`

```bash
curl "http://localhost:8000/suggest?q=nik&k=5"
```

**Response**:
```json
{
  "prefix": "nik",
  "suggestions": [
    "nike",
    "nikon",
    "nike club hoodie",
    "nike baseball cap",
    "nike gym duffel bag"
  ]
}
```

### 3. Similar Products ("More Like This")
`GET /similar/{product_id}?k={limit}`

```bash
curl "http://localhost:8000/similar/P001?k=3"
```

**Response**:
```json
{
  "id": "P001",
  "results": [
    {
      "id": "P064",
      "title": "Nike Air Zoom Winflo Running Shoe",
      "category": "running shoes",
      "brand": "Nike",
      "price": 4600.0,
      "rank": 1,
      "similarity": 0.833
    }
  ]
}
```

### 4. Health & Engine Status
`GET /health`

```bash
curl "http://localhost:8000/health"
```

**Response**:
```json
{
  "status": "ok",
  "products": 886,
  "catalog_version": "78d7c0c4e7",
  "embedder": "hashing-256",
  "llm": "disabled",
  "cache": {
    "backend": "memory",
    "items": 32,
    "hit_rate": 0.25
  }
}
```

### 5. Hot Reload Catalog
`POST /reload`

Re-indexes the database or CSV without server downtime. Unchanged products reuse existing vectors.

---

## 🐍 Python SDK Usage

```python
from lite_search import build_engine
from lite_search.config import Settings

# Initialize engine from .env configuration
cfg = Settings.from_env()
engine = build_engine(cfg)

# 1. Search
response = engine.search("black panjabi under 2000", k=5)
for item in response["results"]:
    print(f"{item['rank']}. {item['title']} - {item['price']} BDT")

# 2. Autocomplete
suggestions = engine.suggest("panj", k=5)
print(suggestions["suggestions"])

# 3. Similar products
similar = engine.similar("P001", k=3)
print(similar["results"])
```

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `CATALOG_CSV` | `data/demo_products.csv` | Path to CSV catalog. |
| `DATABASE_URL` | *None* | SQLAlchemy DB connection string (PostgreSQL, MySQL, etc.). |
| `SQL_QUERY` | *None* | SQL query selecting catalog items. |
| `COLUMN_MAP` | *None* | Map DB column names to standard fields (`col:title,id_col:id`). |
| `REDIS_URL` | *None* | Redis connection URL (`redis://localhost:6379/0`). If unset, uses memory LRU. |
| `CACHE_PREFIX` | `ls` | Prefix for Redis keys. |
| `TTL_RESULT` | `600` | Search result cache TTL in seconds. |
| `EMBEDDER` | `sentence-transformers` | `sentence-transformers` or `hashing` (zero-download mode). |
| `EMB_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual model for English & Bangla. |
| `INDEX_DIR` | `.lite_search_index` | Directory for persisted vector cache. |
| `LLM_PROVIDER` | `none` | Optional LLM tier (`none` or `anthropic`). |
| `ANTHROPIC_API_KEY` | *None* | Anthropic API key for edge cases. |
| `SEARCH_API_KEY` | *None* | Optional shared secret header (`X-API-Key`) for API security. |
| `RELOAD_MINUTES` | `0` | Auto reload interval in minutes (0 = disabled). |

---

## 📁 Repository Structure

```text
├── data/
│   ├── demo_products.csv       # Sample catalog with 880+ e-commerce items
│   └── common_queries.txt      # Cache pre-warming queries
├── lite_search/
│   ├── __init__.py             # Exports build_engine
│   ├── __main__.py             # CLI entrypoint (--check, --repl, --serve)
│   ├── cache.py                # Redis & in-memory LRU cache abstraction
│   ├── catalog.py              # Catalog loading, BM25, and index management
│   ├── config.py               # Pydantic configuration loaded from .env
│   ├── diagnose.py             # Schema and column fill-rate check
│   ├── embedder.py             # SentenceTransformer & Hashing embedders
│   ├── engine.py               # Core 3-tier hybrid pipeline
│   ├── lexicon.py              # Banglish dictionary & product category mappings
│   ├── llm.py                  # Optional LLM fallback logic
│   ├── parser.py               # Deterministic rule parser
│   ├── server.py               # FastAPI application
│   ├── text.py                 # Normalization and token utilities
│   └── web_ui.py               # Self-contained HTML/CSS/JS frontend
├── tests/
│   └── test_engine.py          # Engine unit tests
├── .env.example                # Example configuration
├── .gitignore                  # Git exclusions for secrets and index files
├── Dockerfile                  # Container definition
├── docker-compose.yml          # Compose file for API + Redis
├── pyproject.toml              # Build & package metadata
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 📄 License

MIT License. Feel free to use in your own projects!
