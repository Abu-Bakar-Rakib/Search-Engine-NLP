# lite_search — Intent, Gap Analysis and Improvement Plan

Status: steps 1, 2a, 6a done (see Progress). Updated 2026-11-25.
Companion files: `docs/search_gaps.md` (manual query log), `benchmarks/relevance_eval.py` (benchmark),
`benchmarks/queries/*.jsonl` (labeled queries), `benchmarks/results/*_baseline.json` (baseline numbers).

## 1. Intent (what we are building)

A light, self-hosted product search engine for Bangla, English and Banglish shoppers. It should give
the same core features as OpenSearch / Typesense (typo tolerance, synonyms, filters, facets, hybrid
lexical + vector search, pinning, business ranking rules), but:

- **Robust core.** The core gives good results on any catalog. It must not depend on per-product hacks.
- **Business rules are config, not code.** Filters (hard or soft), boosts, synonyms, pins, exclusions
  and fallback policy live in one validated profile file. A client changes a few lines, not Python.
- **Light and fast.** CPU only. 100k products in under 150 ms p95. Small RAM.
- **Measured.** Every change runs the relevance benchmark. A change ships only if nDCG@10 does not drop.

### 1.1 Rule zero: no per-query tuning

The first version failed because it was tuned to specific queries and products (about 25 hard-coded rules that
worked on the curated set and broke on the real catalog). We do not repeat that.

- A failing query is **evidence, not a target**. It becomes a labeled test case. It never gets its own code path.
- A fix must name a **general cause** ("the category text is too vague", "the meaning score is outweighed by
  bonuses"). It must work for any catalog and any language, not only for the query that exposed it.
- Comments describe the **rule**, never example queries or products.
- No product, brand or query word lists in the core. Language data (stopwords, Banglish map) and business data
  (synonyms, boosts) live in config, not in ranking code.
- Global parameters (weights, thresholds) may be tuned on the tuning set. They must hold on the **holdout**
  (`benchmarks/queries/holdout.jsonl`, 100 queries, totals only, never read per query).
- Example of a breach: Step 2b (gift-query fixes, reverted). It helped the queries it was tuned on and did
  nothing on unseen ones.

### 1.2 Alignment check (written before every step)

Before each step, answer in the Progress log:
1. Which intent bullet does it serve?
2. Is the change general (any catalog, any query), or does it target specific queries or products?
3. Does it move logic into config, or add more code-side rules?
4. Gate: tuning set up, curated within 0.01, holdout not down.

## 1.3 Original architecture and business invariants (reviewed 2026-11-25)

**Original design** (first commit, Oct 2026; engine 324 lines) was a clean tiered pipeline:
`cache → normalize (Banglish→EN aliases) → rule parser (catalog-built vocabulary: price, size, gender, category, brand,
color, material) → LLM fallback only when rules find nothing → hybrid ranking (BM25 + embedding) with explainable tiers
(same product group first, inside price range, attribute bonuses) → relax waterfall (size → gender → price) → rerank → cache`.
Strengths to keep: catalog-built vocabulary, explainable tiers and notes, no dead ends, connectors, settings from env.
Later additions bolted on a second parser (`nlp.py`: Bangla/Banglish, persona, occasion, domain pack) and about 25
per-product rules inside `_search`. The engine grew to 1,200 lines. That growth, not the original design, caused most root causes.

**Business invariants: every change must keep these.** The `tests/` suite (original) is the contract.
1. API: same endpoints (`/search` GET/POST, `/intent/parse`, `/products`, `/suggest`, `/similar`, `/reload`,
   `/cache/invalidate`, `/health`, `/catalog/stats`, `/benchmark`) and response fields (`query, normalized, corrections,
   intent, parsed_intent, source, notes, results[id, product_id, title, category, brand, price, currency, image_url, rank,
   score, match, matched, decision_explanation …], took_ms, cached`).
2. Filters stated in the query are honoured: price (min/max, also Bangla/Banglish phrases), brand, size, gender,
   color/material when the catalog has them. Inside-range items rank first. The relax waterfall explains itself in `notes`.
   `HARD_PRICE` keeps prices strict.
3. Negations ("not samsung", "without X") are never violated.
4. No dead ends: closest results with a note (`NEVER_EMPTY`), unless the business turns it off.
5. Persona and occasion queries (gift for father, office, gym) return items that fit the need.
6. Multilingual input: Bangla digits, Bangla/Banglish price phrases, Banglish words.
7. Cache, reload, suggest and similar keep working.

**Foundational issues and the planned fix** (each is justified by the root causes above and done inside a roadmap step):
| # | Issue | Fix (step) |
|---|---|---|
| F1 | Two parsers merged ad hoc | One query-understanding module (roadmap 3) |
| F2 | Scores on mixed scales (rank fusion ~0.02 vs tiers ~50) | Partly fixed (Step 7). Normalize every signal to 0–1 with one weight each (roadmap 4) |
| F3 | 800-line `_search` with per-product rules | Retire by rule family with ablation (roadmap 3) |
| F4 | Bangla understanding is dictionary-only | Transliteration + catalog-built lexicon (roadmap 2) |
| F5 | Cache keys lack code/config version (R11) | Add version + settings hash (with Step 4 profile) |
| F6 | Original test suite not runnable (no pytest; 3 env failures) | Make it run and pass; run it before every commit |

## 2. Baseline (before any change)

| Dataset | Products | Queries | P@10 | Hit@1 | nDCG@10 | Zero-result | p50 / p95 latency |
|---|---|---|---|---|---|---|---|
| curated (`data/all_products.csv`) | 1,886 | 50 | 0.864 | 0.96 | **0.919** | 0% | 198 / 278 ms |
| live (scraped products DB) | 100,420 | 65 | 0.608 | 0.68 | **0.624** | 6.2% | 442 / 659 ms |

Live by query type: plain 0.84, brand 0.99, price 0.68, typo 0.55, vague 0.73, **gap queries 0.10**.
Live by language: en 0.62, banglish 0.70, bn 0.58.

### Progress

| Step | Live nDCG@10 | Live Zero% | Curated nDCG@10 | Live p50 | What changed |
|---|---|---|---|---|---|
| baseline | 0.624 | 6.2% | 0.919 | 442 ms | — |
| step1 | **0.669** | **0%** | **0.939** | 436 ms | One price source with direction words first, no spec units, 5k support (R5). Never-empty fallback, `NEVER_EMPTY` setting (R2). A brand word must be the brand in at least 15% of titles that use it (R1). |

Wider reference (185 live queries, `--sets live,live_extra`, step1): nDCG@10 **0.569**, Zero 0%, p50 446 ms.
By type: plain 0.69, price 0.71, brand 0.68, gap 0.41, vague 0.45, **typo 0.36**. By language: en 0.70, banglish 0.49, **bn 0.39**.
Step 2a (generic product-term gate + title field weight 3x, no product-type list): **0.592** (+0.023). en 0.73, banglish 0.51, bn 0.39; typo 0.41. Curated 0.941.
Fixed: baby diaper 0→1.0, perfum for men 0→0.73, churi 0.10→0.59. Worse: smart wach 0.37→0.18 (typo, Step 3).
Holdout (30 unseen queries): step1 0.484 · step2a 0.488 · step2b 0.474. Step 2b was tuned around manually tried queries
(gift for father) and did not generalize: unseen vague queries stayed ~0, so it was reverted (60a6772).
Note: holdout is small (30, only 3 vague) — one query moves it by ~0.015. Grow it before trusting small deltas.
Holdout grown to 100 (vague 20, typo 20, gap 18, plain 15, price 15, brand 12; en 40, bn 30, banglish 30).
Reference on step2a: **holdout nDCG@10 0.465** — en 0.70, banglish 0.41, **bn 0.20**; price 0.83, brand 0.61, plain 0.50, gap 0.40, **typo 0.30, vague 0.29**.
Accept rule from here: tuning set (185) up AND holdout (100) not down. Only holdout totals are read, never per-query rows.
Step 6 (moved up) — embedding model A/B:
| Model | dense-only nDCG (185) | bn | en | engine nDCG (185) | holdout (100) |
|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 | 0.316 | **0.012** | 0.50 | 0.592 | 0.465 |
| multilingual-e5-small (query = raw + translated) | **0.500** | 0.146 | 0.72 | **0.602** | **0.476** |
The old model barely understood Bangla. e5 alone (0.50) is close to the full engine; inside the engine the gain is small, so rules/bonuses outweigh the dense signal. Adopted e5 (EMB_MODEL, EMB_QUERY=both).
Step 7 — meaning-score weight. Alignment check:
1. Intent: *Robust core* (works on any catalog) and *Business rules are config* (weight read from settings).
2. General: one global weight for the meaning score on the same scale as the other bonuses. No query, product or word list.
3. Config, not code: `DENSE_TIER` setting (later in the profile file). It shifts ranking power from hand rules to the model.
4. Gate: tuning up, curated within 0.01, holdout not down.
Cause: the keyword score added up to 15 bonus points, the meaning score only ~0.02 (via rank fusion), so rules decided ranking.
Sweep (tuning / curated): w=0 0.602/0.941 · 10 0.622/0.942 · 15 0.625/0.939 · **25 0.641/0.936** · 40 0.635/0.927 (fails curated).
Holdout with 25: 0.476 → **0.492** (typo 0.34→0.37, vague 0.31→0.33, banglish 0.42→0.45). Adopted `DENSE_TIER=25`.

Step 3 — typo and partial-word matching (R4). Alignment check:
1. Intent: *same core features as Typesense* (typo tolerance, prefix search) and *robust core* (any catalog, any language).
2. General: the vocabulary and word frequencies come from the catalog itself. Typo rule: a query word also matches catalog words
   1 edit away (2 for words of 7+ letters, Typesense defaults) that are at least 10x more common. Prefix rule: a rare or unknown LAST word also
   matches the most common catalog words that start with it. Works the same for English, Banglish and Bangla script.
   No word lists and no per-query maps (the old `_TYPO_MAP` is left as is for now and removed in a later cleanup).
3. Config: `TYPO_EXPAND` on/off. Edit distances and weights are constants for now (to the profile in Step 4).
4. Gate: tuning up, curated within 0.01, holdout not down. Also compare typo and partial-word groups with Typesense.
Result: tuning 0.641 → **0.647** (typo 0.54 → 0.58), curated 0.936 (no change), holdout 0.492 → **0.508** (typo 0.37 → 0.46,
banglish 0.45 → 0.48, bn 0.24 → 0.26). Index: 31k words, 0.3 s build, ~90 MB RAM, <0.1 ms per query. Adopted `TYPO_EXPAND=true`.
Known limit: Bangla-script misspellings rarely fix, because most catalog titles are English (the right Bangla spelling is not in the vocabulary).
This is the number later steps must beat. `holdout.jsonl` (30) is checked only at the end of each step.

The engine scores well on the clean catalog it was tuned on, and poorly on the real catalog. Most of the gap comes from the design, not from the models.

## 3. Root causes (where, why, what to do)

| # | Root cause | Where | Proof queries (live) | Fix |
|---|---|---|---|---|
| R1 | **Two parsers, merged ad hoc.** `RuleParser` takes any catalog brand word as a brand, so "water", "top", "bulb" and "hot" all become brands. | `parser.py:231`, merge in `engine.py:355-384` | tank top (brand=top), something to keep my coffee hot (brand=hot), ac dc light bulb (brand=bulb) | One parser. It tags entities with confidence. A brand counts only if the word is mostly used as a brand in the catalog and is not a dictionary word. |
| R2 | **Hard filters with no recovery.** Group mask, brand, color, storage, model generation and the final relevance gate all set the score to -1e9. | `engine.py:778, 810-828, 953, 995-1003, 1005-1033` | hand glovs, condom cheap, smart wach, ac dc light bulb (0 results) | Filters become scored features. A relaxation ladder comes from config. The result list is never empty unless the profile says strict. |
| R3 | **No head-noun priority.** Persona or intent affinities and group boosts overwrite the product word. | `engine.py:466-509, 642`, `_intent_compat_scores` `engine.py:254`, `nlp.py:873-881` | baby diaper → teddy bears, perfum for men → shirts, men underwear → shirts, রান্নার কড়াই → kettles, ghori detergent → watches | Find the product type (head noun) first. It is the strongest signal. Persona and intent only re-rank inside that type. |
| R4 | **The typo and alias layer ignores context.** Per-token maps, fuzzy fixing picks the first match at min distance, a typo that appears in the catalog is "known", and words under 5 letters are never fixed. | `nlp.py:640-716`, `_TYPO_MAP` `nlp.py:306`, `"ac"` map `nlp.py:479` | hand glovs → gloss, face creame, smart wach, ac dc light, churi | Frequency-weighted fuzzy fixing (SymSpell-style) over catalog terms. Score candidates by doc frequency. Accept an alias only if the other words agree. |
| R5 | **Price parsing.** "max" is read as a price cap. The plain "N taka" pattern runs before "above". There are two price extractors. | `nlp.py:781-791`, `parser.py:28` | iphone … pro max 128gb, মায়ের জন্য শাড়ি ২০০০ টাকার উপরে (max instead of min) | One price parser. Direction words are found first. "max" needs a currency or number context. Tested by unit tests. |
| R6 | **Catalog model does not fit the real data.** The live SQL has no color, gender, material or size columns, so gender is always unisex. There are 2,687 noisy shop categories, the taxonomy has 90 leaves, and the product type is only in the title. | `.env SQL_QUERY`, `catalog.py:60-75`, `domain_pack.json taxonomy_map` | red cotton t-shirt … for boys, football (category "footballs" → cake toppers) | Enrich at ingestion: derive `product_type`, canonical category, color and gender from the title. This runs once, offline. |
| R7 | **About 25 hard-coded per-product rules** (iPhone, Sony, Pegasus, AC, programming laptops, mom gift, phone brand lists). They were tuned to the curated set. | `engine.py:600-770` | curated 0.92 vs live 0.62 | Delete them. The general product-type match and config boost rules replace them. |
| R8 | **Models.** The reranker (ms-marco) is English-only but is used on Bangla. The embedding uses only the translated query. The product text includes a 1,000-char description that the 128-token model cuts off. | `config.py:42-45`, `catalog.py:78-95`, `engine.py:610, 905` | bn nDCG 0.58 | Benchmark a multilingual embedder and reranker. Use title-centred embedding text. Use ONNX int8 on CPU. |
| R9 | **Latency.** Per-query Python loops over all 100k rows (title scans, regex per row, list comprehensions). | `engine.py:265-273, 535-596, 673-679, 684-747, 873, 1019` | p50 442 ms on 100k vs 198 ms on 1.9k | Precompute token→row maps at index time. Score only the top-N candidates (BM25 ∪ dense). Vectorize. |
| R10 | **Config is ignored.** `retrieval_config` and `threshold_config` in the domain pack are never read. The weights are hard-coded constants. `config/domain_pack_*.json` are unused. | `engine.py:27-44`, `nlp.py:95-101` | — | One profile file (YAML), validated at startup. The engine reads every weight and threshold from it. |
| R11 | **Cache and index correctness.** Intent and normalization cache keys have no catalog version. Result keys have no config hash. A shared index dir drops the other catalog's vectors. | `engine.py:74, 89, 305`, `catalog.py:138` | (silent stale results with Redis) | Put the catalog version and the profile hash in every cache key. Use one index dir per catalog. |
| R12 | **Fake benchmark.** "Returned anything" counted as a hit. | `benchmarks/run_benchmark.py:58-66` | — | Replaced by `benchmarks/relevance_eval.py`. |
| R13 | **Code shape.** `_search` is a 775-line method. Lexicons are spread over 5 places. `reranker.py` is unused. | `engine.py:321-1095`, `lexicon.py`, `nlp.py` | — | Split the code into pipeline stages (section 4). One lexicon source. |

## 4. Target architecture

```
offline:  ingest → enrich (product_type, canonical category, color, gender) → index (BM25 + vectors + token maps)
online:   normalize (NFC, digits, Banglish→EN) → understand (typo, entities with spans + confidence, head noun, price)
          → retrieve (BM25 top-N ∪ dense top-N) → features → rank (weights from profile)
          → business rules (filters / boosts / pins / exclusions from profile) → relax ladder → rerank (optional)
          → explain (why each result is here)
```

Proposed package layout:

```
lite_search/
  core/        query.py (understand) · retrieve.py · rank.py · relax.py · explain.py
  index/       ingest.py · enrich.py · bm25.py · vectors.py
  lang/        normalize.py · banglish.py · typo.py · lexicon/ (json data only)
  rules/       profile.py (pydantic schema) · apply.py
  api/         server.py · web_ui.py
profiles/      default.yaml · strict.yaml · marketplace.yaml
```

What goes in the profile (business) and what stays in code (core):

| Profile (config) | Core (code, tested) |
|---|---|
| fallback on/off, min_results, relax order | tokenizing, normalizing, typo engine |
| hard/soft per filter (price, brand, gender, category) | entity tagging and head-noun detection |
| boosts (in-stock, sponsored, margin), pins, exclusions | BM25, dense retrieval, fusion |
| synonyms, Banglish map, brand stoplist, typo overrides | price-direction parsing |
| ranking weights, model names, rerank on/off | relaxation mechanics, explain output |

## 5. Plan (each step gated by the benchmark)

1. **Safety fixes.** R5 price, R2 never-empty fallback, R1 brand confidence. Expected: zero-result rate goes to 0%, gap queries improve.
2. **Head noun and product_type.** R3 + R6 enrichment. Delete the R7 hacks. Expected: biggest nDCG gain on live.
3. **Typo engine.** R4: frequency-weighted, catalog-built.
4. **Profile config.** R10: pydantic and YAML, plus R11 cache keys.
5. **Speed.** R9: candidate-only scoring. Target p95 under 150 ms at 100k.
6. **Models.** R8: A/B test multilingual embedder and reranker with the benchmark. Keep a model only if it wins.
7. **Restructure.** R13: move the code into the layout above, done in step with the steps above (not a big-bang rewrite).

Rule for every step: run `benchmarks/relevance_eval.py live` and `curated`. Accept the step if nDCG@10 on live goes up and does not drop on curated by more than 0.01. Save results as `benchmarks/results/<dataset>_<step>.json`.

## 6. Benchmark protocol

- Query sets: `benchmarks/queries/live.jsonl` (66) and `curated.jsonl` (50). They cover en, bn and Banglish, plus typo, price, brand, vague and gap types.
- Relevance: a title regex, plus an optional exclude regex and price range. It is deterministic and cheap. Queries with 0 relevant items in the catalog are skipped and listed.
- Metrics: P@10, Hit@1, nDCG@10, zero-result rate, p50/p95 latency (cold, cache off).
- Limit: regex labels are a proxy. Grow the set over time (target 200+). Add a held-out set so we do not tune to these queries.

## 7. Investigation (2026-11-25): what limits quality now

Priority change (user, 2026-11-25): **relevance first, speed later.** Step 5 (speed) is postponed.
Method: tuning set only (holdout untouched). Per-query signal diagnosis, a plain-hybrid ablation, and an LLM label audit.

| Finding | Evidence | Meaning |
|---|---|---|
| Labels undercount | LLM judge on 674 results of the 74 weakest queries: 139 right results marked wrong, 11 wrong ones marked right. True P@10 0.36 vs measured 0.19 (bn 0.43 vs 0.19). | Scores read low, Bangla most. The yardstick needs fixing before we can see real gains. |
| Bangla/Banglish words not understood | 67% of bn queries keep untranslated Bangla words after normalization. ~60% of the 40 truly bad queries are this (শ্যামপু, চারজার, istri, chhat fan, gas chula, pani er filter). Only 7.4% of titles are Bangla. | Main cause is the query side (no translation/transliteration), not missing Bangla product text. |
| Old rule layer misfires | Plain hybrid (BM25+typo, meaning, price) scores 0.625 vs full engine 0.647: rules add +0.02 net, but are worse on gap queries (0.56 vs 0.65); 13 queries better without rules, 21 worse. ~25% of bad queries are rule misfires (product group +50 on toothpaste, "wear" in underwear, "ac" phrase map, "case" → phone case). | Keep the rule families that help (price, brand, Bangla dictionary, needs), remove or soften the rest. |
| Vague needs | 5 of the 40 bad queries ("sleep better", "keep coffee hot"). | Needs a need → product-type layer that holds for any query. |
| English-only reranker | ms-marco cross-encoder runs on Bangla too. | A/B a multilingual reranker. |

## 8. Roadmap (quality first; every step: alignment check + gate)

1. **Judged labels (measurement).** Pool the top 10 from Light Search and Typesense for all 285 queries. An LLM judges each result, and a human spot-checks 10%. This becomes the main score, and the regex score stays as a second number. It changes the yardstick, not the engine.
2. **Bangla/Banglish understanding (biggest gain).** (a) Transliterate Bangla script to Latin, then run the catalog typo matcher (handles every English loanword: শ্যামপু → shampoo). (b) A bilingual lexicon built **from catalog vocabulary**, never from benchmark queries: the top product nouns get Bangla script and Banglish spellings (LLM-drafted, human-checked). It is stored as config data.
3. **Retire the rule layer by family.** Ablate each family (product groups, phrase maps, category hacks, per-product rules). A family is kept only if the holdout says it helps. The rest is replaced by soft, data-driven features.
4. **Learned global weights.** Run a coordinate search over the ~6 ranking weights on the tuning set, then check on the holdout.
5. **Multilingual reranker A/B.** Keep it only if it wins.
6. **Need → product layer for vague queries** (config data + meaning model).
Postponed: speed (Step 5).
