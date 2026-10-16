"""Optional LLM fallback over plain HTTPS (no SDK needed). Used ONLY when rules find nothing.
Fails soft: timeouts/errors return None and a circuit breaker pauses calls after repeated failures."""
import json
import logging
import os
import time
import urllib.request
from typing import Dict, Optional

from . import lexicon as L
from .parser import SCHEMA_KEYS, parse_json

log = logging.getLogger("lite_search.llm")

PARSE_SYSTEM = f"""Extract product search attributes from the query. Return ONLY one JSON object.
Allowed keys: {SCHEMA_KEYS}.
Include ONLY the keys that are stated in the query and omit all others. Never guess.
- category: product type, singular, lowercase.
- size: as written (uppercase for clothing sizes).
- gender: men, women, boys, girls or unisex.
- requirement: quality or use-case wanted (never a price).
- min_price / max_price: numbers only. "under X" -> max_price, "over X" -> min_price, "between X and Y" -> both."""
PARSE_SHOTS = [
    ("red cotton M size shirt for men under $30",
     {"category": "shirt", "size": "M", "color": "red", "material": "cotton", "gender": "men", "max_price": 30}),
    ("something warm for a cold winter", {"requirement": "warm, for cold winter"}),
]
TRANS_SYSTEM = ("Translate this e-commerce search query into a short English product search query. "
                "The input may be Bangla, Banglish, Hindi, Urdu or misspelled English. "
                "Return only the English query. Keep brand names, sizes and numbers.")


class NoLLM:
    available = False

    def parse(self, query): return None
    def translate(self, query): return None


class AnthropicLLM:
    URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model: str, timeout: float = 4.0, api_key: Optional[str] = None):
        self.model, self.timeout = model, timeout
        self.key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.available = bool(self.key)
        self._fails, self._paused_until = 0, 0.0

    def _call(self, system: str, shots, user: str, max_tokens: int) -> Optional[str]:
        if not self.available or time.time() < self._paused_until:
            return None
        msgs = []
        for q, a in shots:
            msgs += [{"role": "user", "content": q},
                     {"role": "assistant", "content": a if isinstance(a, str) else json.dumps(a)}]
        msgs.append({"role": "user", "content": user})
        body = json.dumps({"model": self.model, "max_tokens": max_tokens, "system": system, "messages": msgs}).encode()
        req = urllib.request.Request(self.URL, body, {"x-api-key": self.key, "anthropic-version": "2023-06-01",
                                                      "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                data = json.load(r)
            self._fails = 0
            return "".join(b.get("text", "") for b in data.get("content", []))
        except Exception as e:
            self._fails += 1
            log.warning("LLM call failed (%s)", e)
            if self._fails >= 3:                       # circuit breaker: stop hammering a broken API for 60 s
                self._paused_until, self._fails = time.time() + 60, 0
            return None

    def parse(self, query: str) -> Optional[Dict]:
        text = self._call(PARSE_SYSTEM, PARSE_SHOTS, query, 120)
        return parse_json(text) if text else None

    def translate(self, query: str) -> Optional[str]:
        text = self._call(TRANS_SYSTEM, L.TRANS_SHOTS, query, 32)
        return text.strip().split("\n")[0].strip(" \"'.").lower() if text else None


def make_llm(settings):
    if settings.llm_provider == "anthropic":
        llm = AnthropicLLM(settings.llm_model, settings.llm_timeout)
        if not llm.available:
            log.warning("LLM_PROVIDER=anthropic but ANTHROPIC_API_KEY is not set - LLM tier disabled")
        return llm
    return NoLLM()
