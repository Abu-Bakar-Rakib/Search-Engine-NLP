---
name: search-bench
description: Run the lite_search relevance benchmark (live 100k + curated catalogs) and compare against a previous run. Use after any change to lite_search ranking, parsing, typo, price or model code, or when the user says "benchmark", "search-bench", or "did it improve".
---

# search-bench

Measures search quality with labeled queries and blocks changes that make it worse.

## Usage
`/search-bench <tag> [baseline_tag]` — `tag` names this run (e.g. `step2`). `baseline_tag` defaults to the latest earlier tag in `benchmarks/results/`, else `baseline`.

## Steps
1. Unit checks first (fast, fail early):
   `.venv/bin/python tests/test_price_parse.py`
2. Run both datasets (cache is off inside the script; curated uses its own index dir). Live takes several minutes, so run it in the background:
   `.venv/bin/python benchmarks/relevance_eval.py curated --tag <tag>`
   `.venv/bin/python benchmarks/relevance_eval.py live --tag <tag>`
   Wider set (200+ queries, slower): add `--sets live,live_extra` (result tag becomes `<tag>__live+live_extra`). Holdout: `--sets holdout` only at the end of a step.
3. Compare:
   `.venv/bin/python benchmarks/compare.py curated <baseline_tag> <tag>`
   `.venv/bin/python benchmarks/compare.py live <baseline_tag> <tag>`
4. Gate (from `docs/SEARCH_ENGINE_INTENT.md` §5): accept only if live nDCG@10 (`--sets live,live_extra`) goes up, curated drops by no more than 0.01, AND holdout (`--sets holdout`, 100 q) does not drop (`compare.py` exits 1 on a drop). Read only holdout totals, never its per-query rows.
5. For every query in the "worse" list, open its `top` list in `benchmarks/results/<ds>_<tag>.json` and give the cause in one line.
6. Report in a short table: dataset · nDCG@10 · P@10 · Zero% · p95, old → new, then PASS/FAIL, then the worse-query causes.
7. Add the new row to the baseline table section in `docs/SEARCH_ENGINE_INTENT.md` ("Progress" table).

## Rules
- Never edit `benchmarks/queries/*.jsonl` labels to make a run pass. A label fix needs its own commit and a reason.
- Keep `benchmarks/queries/holdout.jsonl` out of tuning: check it only at the end of a step.
- Queries with 0 relevant items in the catalog are skipped automatically and listed.
