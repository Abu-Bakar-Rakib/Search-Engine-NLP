"""HTTP API.   Run:  uvicorn lite_search.server:app --host 0.0.0.0 --port 8000
Configuration comes from environment variables (see .env.example)."""
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .config import Settings
from .engine import build_engine
from .web_ui import INDEX_HTML

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
STATE = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = Settings.from_env()
    STATE["engine"] = build_engine(cfg)
    stop = threading.Event()
    if cfg.reload_minutes:                       # keep the index in sync with the company DB
        def loop():
            while not stop.wait(cfg.reload_minutes * 60):
                try:
                    STATE["engine"] = build_engine(Settings.from_env())   # atomic swap; unchanged products reuse vectors
                except Exception:
                    logging.exception("scheduled reload failed - keeping the previous index")
        threading.Thread(target=loop, daemon=True).start()
    yield
    stop.set()


app = FastAPI(title="lite_search", version="0.1.0", lifespan=lifespan)
_cors = os.environ.get("CORS_ORIGINS")
if _cors:
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in _cors.split(",")],
                       allow_methods=["GET", "POST"], allow_headers=["*"])
API_KEY = os.environ.get("SEARCH_API_KEY")          # optional shared secret for backend-to-backend calls


def _auth(key: Optional[str]):
    if API_KEY and key != API_KEY:
        raise HTTPException(401, "invalid api key")


class SearchRequest(BaseModel):
    query: str
    k: int = Field(10, ge=1, le=100)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(content=INDEX_HTML)


@app.get("/search")
def search_get(q: str = Query(..., min_length=1), k: int = Query(10, ge=1, le=100),
               x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return STATE["engine"].search(q, k=k)


@app.post("/search")
def search_post(req: SearchRequest, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return STATE["engine"].search(req.query, k=req.k)


@app.get("/suggest")
def suggest(q: str = Query(..., min_length=1), k: int = Query(8, ge=1, le=20), x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    return STATE["engine"].suggest(q, k=k)


@app.get("/similar/{product_id}")
def similar(product_id: str, k: int = Query(10, ge=1, le=50), x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    out = STATE["engine"].similar(product_id, k=k)
    if out.get("error"):
        raise HTTPException(404, out["error"])
    return out


@app.get("/health")
def health():
    return {"status": "ok", **STATE["engine"].info()}


@app.post("/reload")
def reload_catalog(x_api_key: Optional[str] = Header(None)):
    """Re-read the product DB/CSV (call after products change). Unchanged products are not re-embedded."""
    _auth(x_api_key)
    STATE["engine"] = build_engine(Settings.from_env())
    return STATE["engine"].info()
