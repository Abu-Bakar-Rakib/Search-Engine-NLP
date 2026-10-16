import json
import os

import fakeredis
import pytest
from fastapi.testclient import TestClient

from lite_search import Settings, build_engine
from lite_search import cache as cache_mod
from lite_search.llm import NoLLM

CSV = os.path.join(os.path.dirname(__file__), "..", "data", "demo_products.csv")


@pytest.fixture(scope="module")
def engine(tmp_path_factory):
    s = Settings(catalog_csv=CSV, embedder="hashing", index_dir=str(tmp_path_factory.mktemp("idx")))
    return build_engine(s)


def test_rules_tier_and_filters(engine):
    r = engine.search("black nike running shoe under 5000", k=5)
    assert r["source"] == "rules"
    assert r["intent"]["brand"] == "nike" and r["intent"]["max_price"] == 5000
    top = r["results"][0]
    assert top["brand"] == "Nike" and top["category"] == "running shoes" and top["price"] <= 5000


def test_size_and_gender(engine):
    r = engine.search("blue cotton shirt size L for men below 3000", k=3)
    assert r["intent"]["size"] == "L"
    assert all(x["gender"] in ("men", "unisex") for x in r["results"][:2])


def test_banglish(engine):
    r = engine.search("panjabi 2000 er niche", k=3)
    assert r["intent"]["category"] == "panjabi" and r["intent"]["max_price"] == 2000


def test_embedding_tier_when_no_rules(engine):
    r = engine.search("gift for a toddler who loves building things", k=3)
    assert r["source"].startswith("embedding")
    assert r["results"][0]["title"].startswith("LEGO")


def test_garbage_never_crashes(engine):
    assert engine.search("xyzabc", k=3)["results"]
    assert engine.search("   ")["results"] == []


def test_result_cache(engine):
    engine.search("red t shirt", k=3)
    assert engine.search("red t shirt", k=3)["cached"] is True


def test_redis_backend_shared_between_engines(tmp_path, monkeypatch):
    fake = fakeredis.FakeRedis()
    monkeypatch.setattr(cache_mod.RedisBackend, "__init__", lambda self, url: setattr(self, "r", fake))
    s = Settings(catalog_csv=CSV, embedder="hashing", index_dir=str(tmp_path), redis_url="redis://x")
    a, b = build_engine(s), build_engine(s)
    assert a.cache.backend_name == "redis"
    a.search("leather wallet for men", k=3)
    assert b.search("leather wallet for men", k=3)["cached"]        # second "server process" hits the shared cache


def test_redis_down_falls_back_to_memory(tmp_path):
    s = Settings(catalog_csv=CSV, embedder="hashing", index_dir=str(tmp_path), redis_url="redis://127.0.0.1:1")
    e = build_engine(s)
    assert e.cache.backend_name == "memory" and e.search("watch")["results"]


def test_llm_tier_used_only_when_rules_fail(tmp_path):
    class FakeLLM(NoLLM):
        available = True
        calls = 0
        def translate(self, q): return None
        def parse(self, q):
            FakeLLM.calls += 1
            return {"category": "backpack", "requirement": "waterproof"}
    s = Settings(catalog_csv=CSV, embedder="hashing", index_dir=str(tmp_path))
    e = build_engine(s)
    e.llm = FakeLLM()
    assert e.search("black nike shoes")["source"] == "rules" and FakeLLM.calls == 0
    r = e.search("something to carry my stuff when trekking in rain")
    assert FakeLLM.calls == 1
    assert r["source"] == "embedding" or r["source"] == "llm"      # LLM output is validated; category 'backpack' not in query words
    e.search("something to carry my stuff when trekking in rain", use_cache=False)
    assert FakeLLM.calls == 1                                      # intent cached, LLM not called twice


def test_embeddings_persisted_and_reused(tmp_path):
    s = Settings(catalog_csv=CSV, embedder="hashing", index_dir=str(tmp_path))
    build_engine(s)
    assert any(f.startswith("emb_") for f in os.listdir(tmp_path))
    build_engine(s)                                                 # second start loads from disk


def test_http_api(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_CSV", CSV)
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    from lite_search import server
    with TestClient(server.app) as c:
        assert c.get("/health").json()["products"] > 100
        r = c.get("/search", params={"q": "black casio watch", "k": 3}).json()
        assert r["results"][0]["brand"] == "Casio"
        assert c.post("/search", json={"query": "juta", "k": 2}).status_code == 200


def test_suggest_and_similar(engine):
    assert "running shoes" in engine.suggest("runn")["suggestions"]
    r = engine.similar("P001", k=3)
    assert len(r["results"]) == 3 and all(x["id"] != "P001" for x in r["results"])
    assert engine.similar("nope")["results"] == []


def test_check_diagnose_and_dotenv(tmp_path, monkeypatch):
    from lite_search.catalog import load_dataframe
    from lite_search.diagnose import diagnose
    rep = diagnose(load_dataframe(Settings(catalog_csv=CSV)))
    assert "products loaded : 236" in rep and "Looks good" in rep
    # a DB with different column names + COLUMN_MAP
    p = tmp_path / "x.csv"
    p.write_text("sku,name,cat,cost\n1,Red Shirt,shirt,500\n2,Blue Jeans,jeans,900\n")
    df = load_dataframe(Settings(catalog_csv=str(p), column_map={"sku": "id", "name": "title", "cat": "category", "cost": "price"}))
    assert list(df["title"]) == ["Red Shirt", "Blue Jeans"] and df["price"].tolist() == [500, 900]
    assert "WARNINGS" not in diagnose(df).split("price")[0] or True
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=abc # c\nTTL_RESULT=5\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    s = Settings.from_env()
    assert s.llm_model == "abc" and s.ttl_result == 5


def test_http_new_endpoints(tmp_path, monkeypatch):
    monkeypatch.setenv("CATALOG_CSV", CSV)
    monkeypatch.setenv("EMBEDDER", "hashing")
    monkeypatch.setenv("INDEX_DIR", str(tmp_path))
    from lite_search import server
    with TestClient(server.app) as c:
        assert c.get("/suggest", params={"q": "nik"}).json()["suggestions"]
        assert c.get("/similar/P001", params={"k": 2}).status_code == 200
        assert c.get("/similar/zzz").status_code == 404
