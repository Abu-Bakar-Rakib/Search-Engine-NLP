"""
Benchmark module for measuring latency, cache hit performance, and throughput.
Supports both lite_search.SearchEngine and IntentAwareHybridSearchEngine.
"""
import os
import time
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

BENCH_SAMPLE_QUERIES = [
    "gift for my father under 3000",
    "comfortable black nike running shoe under 5000",
    "blue cotton shirt size L for men below 3000",
    "বাচ্চাদের খেলনা ১৫০০ টাকার নিচে",
    "babar jonno gift under 3000",
    "noise cancelling headphones for flights",
    "keep my coffee hot on a long drive",
    "online class er jonno",
    "kom dam e valo phone",
    "educational toy for toddler"
]


def benchmark_latencies(engine: Any, queries: List[str] = BENCH_SAMPLE_QUERIES) -> Dict[str, Any]:
    """Measures warm, cold, and cached latencies with breakdown across pipeline stages."""
    # 1. Warm-up
    for q in queries[:2]:
        engine.search(q, use_cache=False)

    # 2. Cold Queries (Cache bypassed)
    cold_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        res = engine.search(q, use_cache=False)
        took = res.get("took_ms") if isinstance(res, dict) else getattr(res, "latency_ms", (time.perf_counter() - t0) * 1000)
        cold_latencies.append(took)

    # 3. Cached Queries (Second pass)
    for q in queries:
        engine.search(q, use_cache=True)
    cached_latencies = []
    for q in queries:
        t0 = time.perf_counter()
        res = engine.search(q, use_cache=True)
        took = res.get("took_ms") if isinstance(res, dict) else getattr(res, "latency_ms", (time.perf_counter() - t0) * 1000)
        cached_latencies.append(took)

    cold_arr = np.array(cold_latencies, dtype=np.float64)
    cached_arr = np.array(cached_latencies, dtype=np.float64)

    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_avail else "None (CPU Execution)"
    except Exception:
        cuda_avail = False
        gpu_name = "None (CPU Execution)"

    return {
        "hardware": {
            "device": "cpu",
            "cuda_available": cuda_avail,
            "gpu_name": gpu_name,
        },
        "query_count": len(queries),
        "cold_latencies_ms": {
            "mean": round(float(np.mean(cold_arr)), 2),
            "median": round(float(np.median(cold_arr)), 2),
            "p95": round(float(np.percentile(cold_arr, 95)), 2),
            "min": round(float(np.min(cold_arr)), 2),
            "max": round(float(np.max(cold_arr)), 2),
        },
        "cached_latencies_ms": {
            "mean": round(float(np.mean(cached_arr)), 2),
            "median": round(float(np.median(cached_arr)), 2),
            "p95": round(float(np.percentile(cached_arr, 95)), 2),
            "min": round(float(np.min(cached_arr)), 2),
            "max": round(float(np.max(cached_arr)), 2),
        },
        "speedup_factor": round(float(np.mean(cold_arr)) / max(float(np.mean(cached_arr)), 0.001), 2),
    }
