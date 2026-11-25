# lite_search — working rules

Read `docs/SEARCH_ENGINE_INTENT.md` first. It holds the intent, the root causes, the plan and the progress log.

- **Rule zero (§1.1): no per-query tuning.** A failing query becomes a labeled test case in `benchmarks/queries/`,
  never its own code path. Fixes name a general cause. Comments describe rules, not example queries.
- **Before every step,** write the alignment check (§1.2) in the Progress log: which intent bullet, general or specific,
  config or code.
- **Gate:** `/search-bench` — tuning set (`--sets live,live_extra`) up, curated within 0.01, holdout (`--sets holdout`)
  not down. Read holdout totals only.
- **After a code change,** restart the server and clear the Redis `ls:result:*` keys (cache keys have no code version yet, R11).
- The Typesense comparison lives in `../Typesense` (separate repo). Never commit it here.
