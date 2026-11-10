# Local Development Setup Guide

This guide details how to set up, run, and develop the **Intent-Aware Hybrid E-Commerce Search Engine** on your local machine (Windows or Linux).

---

## 1. Prerequisites and Python Version

- **Recommended Python**: Python 3.10, 3.11, or 3.12 (64-bit)
- **Operating Systems**: Windows 10/11 or Linux (Ubuntu 20.04+)
- **Hardware Acceleration**: CPU-only execution is fully supported; NVIDIA GPU (CUDA) is automatically used if available (e.g. Colab/Kaggle T4).

---

## 2. Virtual Environment Setup

### Windows (PowerShell)
```powershell
# Navigate to the project root
cd "c:\Users\hp\Desktop\SEARCH ENGINE"

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
.venv\Scripts\Activate.ps1
```

### Linux / macOS (Bash)
```bash
cd /path/to/SEARCH\ ENGINE
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Dependency Installation

Install all required dependencies:
```bash
pip install torch sentence-transformers rank-bm25 redis pandas numpy streamlit sqlalchemy
```

### Dependency Breakdown:
| Package | Version | Purpose |
| :--- | :--- | :--- |
| `torch` | >= 2.0.0 | Deep learning runtime for neural encoders |
| `sentence-transformers` | >= 2.5.0 | Multilingual sentence & cross-encoder reranking models |
| `rank-bm25` | >= 0.2.2 | BM25 lexical token indexer |
| `redis` | >= 5.0.0 | High-performance cache client (optional fallback included) |
| `pandas` & `numpy` | Latest | High-speed vectorized filtering, RRF fusion, and profiling |
| `streamlit` | >= 1.30.0 | Interactive web search UI and decision inspector |
| `sqlalchemy` | >= 2.0.0 | Generic relational database connector for server integration |

---

## 4. Redis Cache Configuration & Fallback

The search engine features a **3-Store Cache System**:
1. **Store A (Term / Phrase Store)**: Catalog terms & alias vector index.
2. **Store B (Query Cache)**: Normalized queries mapped to product IDs with authoritative stock/price revalidation.
3. **Store C (Model Output Cache)**: Translation and intent caches with TTL.

### Option A: Running with Redis (Recommended for Production / Shared Multi-Worker)
If Redis is installed locally:
- **Windows**: Install Redis via WSL or Memurai (`net start memurai`).
- **Linux**: `sudo systemctl start redis-server`
- Default connection: `127.0.0.1:6379`, DB `0`.

### Option B: Running CPU-Only / In-Process LRU Fallback (Zero Setup)
**Redis is completely optional.** When no Redis server is detected on `127.0.0.1:6379`, the search engine automatically and seamlessly activates its **thread-safe in-process LRU cache** with bounded memory. No configuration changes are required!

---

## 5. Model Download & Cache Management

The engine uses two compact, high-efficiency models:
- **Dense Multilingual Embedder**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (~470MB)
  - Supports English, Bangla (বাংলা), and 50+ languages natively.
- **Neural Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (~80MB)
  - Calibrated for product relevance score adjustment.

Models are automatically downloaded on the first run and cached locally in `~/.cache/huggingface/hub/`.

---

## 6. How to Run Locally

### Interactive Terminal CLI
```bash
python cli.py
```
- Type any query: `"gift for my father under 3000"`, `"কালো জুতা"`, `"panjabi 2000 er niche"`.
- Type `'k 10'` to change top results count.
- Type `'portability'` to run second catalog verification.
- Type `'exit'` to quit.

### Interactive Web Search Application (Streamlit)
```bash
streamlit run app.py
```
Opens an interactive dashboard in your browser (`http://localhost:8501`) with:
- Live search bar and pre-loaded sample queries
- Language detection badge and structured intent breakdown
- Decision log explanation for every retrieved product
- Sidebar controls to test ablations and switch catalogs

### Full Evaluation & Benchmark Suite
```bash
python run_benchmark.py
```
Runs all 120+ ground-truth labeled queries across 6 ablation modes, measures latency, and tests domain portability.

### Jupyter Notebook Execution
Open `intent_aware_hybrid_search.ipynb` or `hybrid_updated_3.ipynb` in VS Code, JupyterLab, Google Colab, or Kaggle:
```bash
jupyter notebook intent_aware_hybrid_search.ipynb
```

---

## 7. Windows-Specific Notes
- **Terminal Encoding**: All scripts configure UTF-8 output (`sys.stdout.reconfigure(encoding='utf-8')`) so Bengali script displays cleanly in PowerShell.
- **Path Separators**: Use forward slashes `/` or `os.path.join` for cross-platform portability.
