# Company Server Integration & Onboarding Guide

This document defines the strict, non-destructive **16-step onboarding procedure** for deploying this search engine on a company's server PC using its existing product database.

---

## 1. Architectural Philosophy: Generic Engine + Domain Pack

The search engine's core code (`lite_search/`) is completely generic and domain-agnostic. It contains **no hardcoded company strings, categories, brands, or business rules**.

All company-specific data and behaviors reside strictly in:
1. **The Database Connector**: Reads from the company's database without altering schemas or tables.
2. **The Domain Pack (`config/domain_pack_<company>.json`)**: Configurable JSON/YAML definitions for schemas, taxonomies, brand aliases, buyer personas, and business rules.

---

## 2. The 16-Step Server Onboarding Procedure

```text
 Stage 1: Assessment & Read-Only Access
   [Step 1: Read-Only DB Connection] ──► [Step 2: Column Profiling] ──► [Step 3: Confirm Schema Map]
                                                                                │
 Stage 2: Normalization & Taxonomy                                              │
   [Step 6: Build Domain Pack] ◄── [Step 5: Taxonomy Mapping] ◄── [Step 4: Validate Catalog]
        │
        ▼
   [Step 7: Configure Lexicon & Aliases] ──► [Step 8: Persona/Intent Mapping]
                                                       │
 Stage 3: Indexing & Storage                           │
   [Step 11: Seed Redis Stores] ◄── [Step 10: Build BM25 Index] ◄── [Step 9: Vector DB Adapter]
        │
        ▼
 Stage 4: Verification & Shadow Rollout
   [Step 12: Assemble Test Queries] ──► [Step 13: Baseline vs Engine Benchmark]
                                                    │
   [Step 16: Controlled Rollout] ◄── [Step 15: Shadow Search Mode] ◄── [Step 14: Calibrate Cutoffs]
```

### Step 1: Connect via Read-Only Credentials
Obtain read-only database credentials (`SELECT` permissions only).
```bash
# In .env (never hardcode credentials in code)
COMPANY_DB_URI="postgresql://ro_user:secret_pass@192.168.1.100:5432/company_catalog"
```
Use `SQLAlchemyConnector`:
```python
from lite_search.connectors import SQLAlchemyConnector
connector = SQLAlchemyConnector(connection_string=COMPANY_DB_URI, table_name="products", schema_map={})
```

### Step 2: Profile the Schema
Run `SchemaProfiler` to automatically inspect data types, null rates, duplicate IDs, and price distributions:
```python
profile = connector.profile_schema()
print("Suggested Schema Map:", profile["suggested_schema_map"])
```

### Step 3: Confirm Proposed `schema_map`
Review and verify the mapping between company column names and the canonical 20-field schema:
```json
{
  "product_id": "item_id",
  "title": "item_title",
  "price": "retail_price",
  "category_leaf": "category_name",
  "category_path": "category_tree",
  "brand": "vendor_name",
  "stock_qty": "inventory_units"
}
```

### Step 4: Normalize and Validate Product Data
Use `SchemaValidator` to transform database rows into canonical `CanonicalProduct` objects. Missing fields are recorded with provenance `"missing"` or `"derived"`.

### Step 5: Map Company Taxonomy
Inspect company categories and group them into logical divisions in the Domain Pack:
```json
"taxonomy_map": {
  "footwear": ["sneakers", "running shoes", "sandals"],
  "apparel": ["shirts", "polo", "jeans"]
}
```

### Step 6: Build the Company Domain Pack
Save the configuration file in `config/domain_pack_<company>.json`.

### Step 7: Configure Lexicon and Brand Aliases
Add company brand misspellings, customer slang, and Bengali / Banglish nicknames:
```json
"lexicon": {
  "aliases": { "juta": "shoes", "panjabi": "punjabi" },
  "brand_aliases": { "samsang": "Samsung", "apel": "Apple" }
}
```

### Step 8: Configure Persona and Intent Affinities
Map buyer personas (father, mother, toddler, student) to the company's actual category names:
```json
"persona_config": {
  "father": {
    "target_audience": "adult",
    "target_gender": "men",
    "affinities": ["leather wallet", "panjabi", "wristwatch", "belt"]
  }
}
```

### Step 9: Existing Vector Database Adapter
- **If the company already has a vector database** (e.g. Qdrant, Milvus, pgvector): Implement `VectorDBAdapterInterface` to query existing vector collections without re-indexing.
- **If no vector database exists**: The engine's pre-computed dense embedding matrix (`ProductCatalogIndexer`) serves queries locally in milliseconds.

### Step 10: Build or Update BM25 Index
Build lexical token inverted indexes from product titles, tags, and attributes.

### Step 11: Seed Redis Stores
Populate the Redis Term Store (Store A) with catalog category names and brand variants for vector similarity lookups.

### Step 12: Generate & Review Company Test Queries
Assemble 50–100 representative queries from company search logs across English, Bangla, and Banglish.

### Step 13: Benchmark Baseline vs Proposed Pipeline
Quantify precision, recall, and NDCG gains over the company's existing search service using `BenchmarkEvaluator`.

### Step 14: Calibrate Thresholds
Tune `min_cosine_threshold`, `rrf_k`, and reranking weights based on empirical validation results.

### Step 15: Enable Shadow Search (Dry-Run Mode)
Run the new search engine in the background alongside the existing search service:
- Log top 5 results from both systems without affecting live customer searches.
- Verify zero latency spikes, zero memory leaks, and 99.9% uptime.

### Step 16: Controlled Production Rollout
Incrementally switch live user traffic:
1. 10% Canary traffic for 48 hours
2. 50% A/B testing
3. 100% Full production deployment
