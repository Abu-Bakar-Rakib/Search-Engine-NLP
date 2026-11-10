# lite_search ⚡

<div align="center">

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg?logo=fastapi&logoColor=white)
![SentenceTransformers](https://img.shields.io/badge/SentenceTransformers-Multilingual-yellow.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20Friendly-EE4C2C.svg?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**High-Performance Multilingual Hybrid Search Engine for E-Commerce Backends**  
*Dual Sparse BM25 + Dense Semantic Vectors + Reciprocal Rank Fusion (RRF) + Neural Cross-Encoder Reranking*  
*Full Native Support for Bangla (বাংলা), Banglish Phonetics, and English*
</div>

---

## ⚡ Highlights

* **True Multilingual Understanding**:
  * Native Bengali Unicode script (`\u0980-\u09FF`) support alongside Banglish phonetic romanization and English.
  * Bengali numerals (`০`-`৯`), multi-word expressions, clothing synonyms, and localized price phrases (`৩০০০ টাকার নিচে`, `২০০০ টাকার মধ্যে`, `১০০০ টাকার কম`).
* **Persona & Occasion Intent Recognition**:
  * Automatically identifies recipients (`বাবা`/`father`, `মা`/`mother`, `বাচ্চা`/`kids`, `ছাত্র`/`student`) and occasions (`উপহার`/`gift`, `online class`, `gym`, `office`).
  * Injects catalog affinity boosts so queries like *"baba ke gift debo"* automatically surface panjabis, leather wallets, and wristwatches without explicit keyword matches.
* **Reciprocal Rank Fusion (RRF)**:
  * Merges rankings from **BM25 lexical search**, **multilingual dense vector similarity**, and **structured intent compatibility** into a unified, balanced score.
* **Neural Cross-Encoder Reranking**:
  * Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` with **Query-Document Pair Caching** for deep semantic relevance and low latency.
* **Zero Dead-Ends (Waterfall Relaxation)**:
  * Strict constraints (budget, size, gender) are applied first; if too few products match, filters relax gracefully (`size` $\rightarrow$ `gender` $\rightarrow$ `price`) with clear explanation notes.
* **Enterprise Connectors & Profiler**:
  * Ingest data from **CSV files**, **SQLite**, or **PostgreSQL / MySQL / Oracle** via SQLAlchemy. Built-in schema validator and dataset profiler.
* **Multi-Level Caching**:
  * Zero-config in-memory LRU cache with automatic Redis fallback for multi-worker production deployments.
* **Ready-to-Use HTTP REST API & Web UI**:
  * Built with FastAPI, including interactive Swagger docs (`/docs`), web UI (`/` and `/ui`), live benchmarking (`/benchmark`), and intent extraction (`/intent/parse`).

---

## 🔍 Search Architecture & Methodology

```text
User Query: English, Bangla (বাংলা), or Banglish
   │
   ├─► 0. Cache Check (Redis / In-Process LRU) ──► Instant Return (<1ms) on hit
   │
   ├─► 1. NLP & Intent Engine (lite_search.nlp)
   │      ├─ Convert Bengali digits (৩০০০ -> 3000) & handle multipliers (3k -> 3000)
   │      ├─ Multi-word & dictionary normalizer (পানির বোতল -> water bottle)
   │      ├─ Canonical attribute resolution (কালো -> black, সুতি -> cotton)
   │      ├─ Persona detection (বাবা/baba -> father, gender: men)
   │      ├─ Intent detection (উপহার -> gift) & category affinities
   │      └─ Constraint extraction (max_price, min_price, size, audience)
   │
   ├─► 2. Dual Retrieval (Sparse + Dense + Intent Alignment)
   │      ├─ BM25 Lexical Score: Inverted token index across Latin + Bengali Unicode
   │      ├─ Dense Vector Cosine Sim: Multilingual MiniLM 384-d semantic embeddings
   │      └─ Intent Compatibility Score: Affinity scoring against catalog categories
   │
   ├─► 3. Reciprocal Rank Fusion (RRF)
   │      Score = w_dense/(60 + r_dense) + w_bm25/(60 + r_bm25) + w_intent/(60 + r_intent)
   │
   ├─► 4. Hard Constraints & Waterfall Relaxation
   │      Apply hard masks (price, size, gender) -> Relax (size -> gender -> price) if < min_results
   │
   ├─► 5. Soft Attribute & Tier Boosts
   │      Calibrated boosts for Brand (+14), Category (+10), Color (+7), Material (+6), Price Proximity
   │
   ├─► 6. Neural Cross-Encoder Reranking
   │      ms-marco-MiniLM-L-6-v2 evaluates top 20-30 candidate pool with Pair Caching
   │
   └─► 7. Response Formatting & Cache Store
          Enriched with scores, BM25, cosine, matched flags, and decision explanations
```

---

## 🧠 AI Models Used

| Model Role | Model Identifier | Architecture | Size | Details |
| :--- | :--- | :--- | :--- | :--- |
| **Dense Embedder** | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 12-layer Bi-Encoder | ~470 MB | 384-dimensional vectors; native Bangla, English, and 50+ languages. |
| **Neural Reranker** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 6-layer Cross-Encoder | ~80 MB | Joint self-attention over query-product pairs with caching. |
| **Fast Fallback Embedder** | `hashing-256` | CRC32 Bag-of-Words | 0 MB | Zero-download instant mode for tests and low-memory environments. |

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

Key environment configurations in `.env`:
```env
CATALOG_CSV=data/demo_products.csv
EMBEDDER=sentence-transformers
EMB_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
USE_RERANK=true
CE_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 3. Run Pre-Built Tests & Demos

* **Multilingual Regression Tests** (Bangla, Banglish, and English):
  ```bash
  python tests/test_multilingual.py
  ```
* **Bangla & Banglish Eligibility Suite**:
  ```bash
  python tests/test_bangla_eligibility.py
  ```
* **Typo & Incorrect Word Intent Recovery Suite**:
  ```bash
  python tests/test_typo_correction.py
  ```
* **REST API Client Test**:
  ```bash
  python tests/test_api_client.py
  ```
* **Full Benchmark Suite (120+ ground-truth labeled queries & latency benchmarks)**:
  ```bash
  python benchmarks/run_benchmark.py
  ```
* **Interactive Live Demo (9 real-world scenarios)**:
  ```bash
  python benchmarks/run_live_demo.py
  ```
* **Interactive Terminal Search (REPL)**:
  ```bash
  python -m lite_search --repl
  ```

---

## 🌐 Running the HTTP API Server

Start the production server:

```bash
python -m lite_search --serve --host 0.0.0.0 --port 8000
# or with uvicorn:
uvicorn lite_search.server:app --host 0.0.0.0 --port 8000
```

* **Interactive Web Search UI**: [`http://localhost:8000/`](http://localhost:8000/) or [`http://localhost:8000/ui`](http://localhost:8000/ui)
* **Interactive Swagger API Docs**: [`http://localhost:8000/docs`](http://localhost:8000/docs)
* **Alternative API Docs (ReDoc)**: [`http://localhost:8000/redoc`](http://localhost:8000/redoc)

---

## 📡 API Reference

### 1. Search Products
`GET /search?q={query}&limit={k}` or `POST /search`

```bash
curl "http://localhost:8000/search?q=বাবার%20জন্য%20উপহার%20৩০০০%20টাকার%20নিচে&limit=3"
```

**Response**:
```json
{
  "query": "বাবার জন্য উপহার ৩০০০ টাকার নিচে",
  "normalized": "father gift",
  "intent": {
    "gender": "men",
    "max_price": 3000,
    "recipient": "father",
    "intents": ["gift"],
    "affinities": ["leather wallet", "leather belt", "panjabi", "wristwatch", "ceramic coffee mug"]
  },
  "source": "rules",
  "notes": [
    "Bangla/Banglish query detected → translated: 'father gift'",
    "persona: father",
    "intent: gift"
  ],
  "results": [
    {
      "id": "P0599",
      "product_id": "P0599",
      "title": "Aarong Khadi Handloom Casual Panjabi",
      "category": "panjabi",
      "brand": "Aarong",
      "color": "navy blue",
      "material": "cotton",
      "gender": "men",
      "sizes": "38,40,42,44",
      "price": 2400.0,
      "currency": "BDT",
      "rank": 1,
      "score": 7.754,
      "cosine": 0.0,
      "bm25": 0.0,
      "match": "intent",
      "matched": "gender+price",
      "decision_explanation": "Rank 1 via RRF hybrid fusion (intent)"
    }
  ],
  "latency_ms": 94.2,
  "is_cache_hit": false
}
```

### 2. Parse Query Intent
`POST /intent/parse`

Extracts structured intent, demographics, budget, affinities, and detected language without executing a full product search:

```bash
curl -X POST "http://localhost:8000/intent/parse" \
     -H "Content-Type: application/json" \
     -d '{"query": "কালো জুতো সাইজ 42"}'
```

**Response**:
```json
{
  "query": "কালো জুতো সাইজ 42",
  "language": "bn",
  "normalized": "black shoes সাইজ 42",
  "retrieval_query": "black shoes",
  "color": "black",
  "size": "42",
  "demographics": {
    "recipient": null,
    "target_audience": null,
    "target_gender": null
  },
  "budget_constraints": {
    "min_price": null,
    "max_price": null,
    "currency": "BDT"
  },
  "confidence": 0.95
}
```

### 3. Catalog Distribution & Statistics
`GET /catalog/stats`

```bash
curl "http://localhost:8000/catalog/stats"
```

**Response**:
```json
{
  "domain_name": "E-Commerce Multilingual Retail",
  "total_products": 886,
  "catalog_version": "78d7c0c4e7",
  "unique_categories": 150,
  "categories": ["t-shirt", "shirt", "panjabi", "running shoes", "saree", "laptop", "watch"],
  "price_summary": {
    "min_price": 350.0,
    "max_price": 395000.0,
    "currency": "BDT"
  }
}
```

### 4. Live Benchmark
`GET /benchmark`

Measures cold vs. cached query latencies and pipeline speedup factors:

```bash
curl "http://localhost:8000/benchmark"
```

### 5. Invalidate Query Cache
`POST /cache/invalidate`

Purges the query cache when catalog items change.

### 6. Autocomplete Suggestions
`GET /suggest?q={prefix}&k={limit}`

Fast prefix-based suggestions from catalog categories, brands, and titles.

### 7. Similar Products ("More Like This")
`GET /similar/{product_id}?k={limit}`

Nearest neighbors in embedding space.

---

## 🐍 Python SDK Usage

```python
from lite_search import build_engine, Settings, QueryParser, DomainPack

# 1. Initialize search engine
cfg = Settings.from_env()
engine = build_engine(cfg)

# 2. Search in Bangla, Banglish, or English
res = engine.search("মায়ের জন্য শাড়ি ২০০০ টাকার মধ্যে", k=5)
for item in res["results"]:
    print(f"#{item['rank']}: {item['title']} - BDT {item['price']} ({item['category']})")

# 3. Direct NLP intent parsing
parser = QueryParser(DomainPack.default())
intent = parser.parse("laptop for student under 50000")
print("Recipient:", intent.recipient)        # student
print("Budget Cap:", intent.max_price)       # 50000.0
print("Affinities:", intent.affinities)      # ['backpack', 'notebook', 'pen set', ...]
```

---

## 🔌 Connecting External Databases

Use the built-in connectors to ingest data from relational databases or data lakes:

```python
from lite_search.connectors import SQLAlchemyConnector, SQLiteConnector, CSVConnector
from lite_search.schema import SchemaProfiler

# Option A: Connect to PostgreSQL or MySQL via SQLAlchemy
schema_map = {"product_name": "title", "cost": "price", "prod_id": "id"}
conn = SQLAlchemyConnector("postgresql://user:pass@localhost:5432/ecommerce", "products", schema_map)
products = conn.load_products()

# Option B: Profile dataset for missing values, outliers, and duplicates
profile = conn.profile_schema()
print(profile)
```

---

## ⚙️ Configuration Reference

| Environment Variable | Default | Description |
|---|---|---|
| `CATALOG_CSV` | `data/demo_products.csv` | Path to CSV catalog. |
| `DATABASE_URL` | *None* | SQLAlchemy DB connection string (PostgreSQL, MySQL, SQLite). |
| `SQL_QUERY` | `SELECT * FROM products` | SQL query selecting catalog items. |
| `COLUMN_MAP` | *None* | JSON/dict mapping DB columns to standard fields. |
| `REDIS_URL` | *None* | Redis connection URL (`redis://localhost:6379/0`). If unset, uses memory LRU. |
| `CACHE_PREFIX` | `ls` | Prefix for Redis cache keys. |
| `TTL_RESULT` | `600` | Search result cache TTL in seconds. |
| `EMBEDDER` | `sentence-transformers` | `sentence-transformers` or `hashing` (fast zero-download mode). |
| `EMB_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Dense multilingual model for English & Bangla. |
| `USE_RERANK` | `true` | Enables neural cross-encoder reranking. |
| `CE_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural cross-encoder reranking model. |
| `INDEX_DIR` | `.lite_search_index` | Directory for persisted embeddings. |
| `SEARCH_API_KEY` | *None* | Optional API key header (`X-API-Key`) for authenticated calls. |

---

## 📁 Repository Structure

```text
├── config/
│   ├── domain_pack_ecommerce.json # E-Commerce retail pack (taxonomy, Bangla dictionary, personas)
│   └── domain_pack_second.json    # Portability domain pack
├── data/
│   ├── all_products.csv           # 1,536 items (unified active catalog combining all datasets)
│   ├── demo_products.csv          # 886 retail items
│   ├── synthetic_catalog_1.csv    # 500-item evaluator benchmark catalog
│   ├── synthetic_catalog_1.sqlite # Read-only benchmark SQLite catalog
│   ├── synthetic_catalog_2.csv    # 150-item industrial tools catalog
│   ├── synthetic_catalog_2.sqlite # Read-only industrial SQLite catalog
│   └── common_queries.txt         # Cache pre-warming queries
├── docs/
│   ├── COMPANY_ONBOARDING.md      # 16-step enterprise integration & read-only DB connector guide
│   ├── LOCAL_SETUP.md             # Local environment and dependency setup
│   └── TECHNICAL_READINESS_REPORT.md # Performance, ablation findings & production SLA report
├── notebooks/
│   ├── hybrid_updated_3.ipynb     # Development & experimental retrieval research notebook
│   └── intent_aware_hybrid_search.ipynb # Interactive exploration and visualization
├── benchmarks/
│   ├── run_benchmark.py           # 120+ query evaluation runner with latency benchmarking
│   └── run_live_demo.py           # Terminal demo covering 9 diverse query types
├── tests/
│   ├── test_multilingual.py       # Live test suite across Bangla, Banglish & English queries
│   ├── test_bangla_eligibility.py # Capability verification for Bengali Unicode & numerals
│   ├── test_typo_correction.py    # Typo tolerance & incorrect word intent recovery suite
│   ├── test_api_client.py         # HTTP REST API integration test suite
│   └── test_engine.py             # Unit tests for core hybrid search engine
├── lite_search/
│   ├── __init__.py                # Package exports (SearchEngine, DomainPack, Connectors, etc.)
│   ├── __main__.py                # CLI runner (--check, --repl, --serve)
│   ├── benchmark.py               # Latency and throughput benchmarking
│   ├── cache.py                   # Multi-level Redis & in-process thread-safe LRU cache
│   ├── catalog.py                 # Fast vectorized BM25 token index and embeddings
│   ├── catalog_generator.py       # Synthetic multilingual catalog generator
│   ├── config.py                  # Pydantic Settings loaded from .env
│   ├── connectors.py              # SQLite, CSV, and SQLAlchemy database connectors
│   ├── diagnose.py                # Catalog diagnostic fill-rate inspector
│   ├── domain_pack.json           # Embedded default domain configuration
│   ├── embedder.py                # SentenceTransformer & Hashing embedders
│   ├── engine.py                  # 7-stage search orchestrator with RRF fusion
│   ├── evaluator.py               # 120+ labeled ground-truth evaluation queries & metrics
│   ├── lexicon.py                 # Keyword mappings and groups
│   ├── llm.py                     # Optional LLM fallback logic
│   ├── nlp.py                     # Multilingual parser, Bangla digits, persona & intent
│   ├── parser.py                  # Deterministic rule parser
│   ├── reranker.py                # Neural CrossEncoder with pair caching
│   ├── schema.py                  # CanonicalProduct dataclass, SchemaValidator, SchemaProfiler
│   ├── server.py                  # FastAPI REST application
│   ├── text.py                    # Unicode Bengali regex, tokenization, and stopwords
│   └── web_ui.py                  # Embedded interactive HTML/CSS/JS frontend
├── Dockerfile                     # Docker container definition
├── docker-compose.yml             # Containerized Search API + Redis
├── pyproject.toml                 # Package metadata
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation
```

---

## 📄 License

MIT License. Free to use in commercial and open-source projects!
