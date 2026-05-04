# Week 1 Retrospective

## What was built

- **Day 1** — Project skeleton: FastAPI app factory, CORS middleware, `/health` endpoint, Pydantic v2 schemas (`Source`, `ResearchRequest`, `ResearchResponse`, `HealthResponse`), stub `POST /research`. Verified by `test_health.py` (3 tests).
- **Day 2** — Production config: `pydantic-settings` `Settings` with `SecretStr` API keys, `lru_cache` singleton via `get_settings()`, 422 validation on `ResearchRequest` (min length 3, must contain a word character). Verified by schema validation tests.
- **Day 3** — Next.js 16 frontend: typed API client (`lib/api.ts`) with `ResearchResponse` interface, React state machine (query / loading / result / error), 30-second `AbortController` timeout, Tailwind v4 styling with source cards.
- **Day 4** — Real LLM client: Groq `llama-3.3-70b-versatile` as primary, Gemini `gemini-2.5-flash-lite` as fallback, sanitized `LLMUnavailableError`, structured telemetry fields (`provider`, `model`, `latency_ms`, `tokens_in`, `tokens_out`). Verified by `test_llm_client.py` (4 tests).
- **Day 5** — Tavily search integration: async `search()` with URL deduplication and 15s timeout, `SearchUnavailableError` sanitization, context-formatted prompt with `[1]`/`[2]` citations, `sources` array in response. Verified by `test_search.py` (3 tests) and `test_research_baseline.py` (5 tests).
- **Day 6** — LangGraph wrapping: `ResearchState` TypedDict, single-node compiled graph (`research_node`), `POST /research/graph` endpoint, Baseline/Graph toggle in the frontend. Verified by `test_graph.py` (4 tests) and `test_research_graph_endpoint.py` (4 tests). Total: 23 tests.
- **Day 7** — Documentation: `docs/architecture.md` (sequence diagram, flowchart, state schema table, component descriptions, telemetry examples, decision log), `docs/week1-retro.md` (this file).

---

## What worked

**`SecretStr` + `lru_cache` for settings and clients.** This pattern required almost no maintenance across the week. API keys never appeared in log output or test stdout even during debugging sessions where state was printed liberally, because `SecretStr.__repr__` returns `'**********'` rather than the value. The `lru_cache` on `get_settings()`, `get_llm_client()`, and `get_search_client()` kept initialization trivial — no dependency injection framework, no application-level global state that tests need to carefully clean up. Tests that need a clean LLM client just call `get_llm_client.cache_clear()` in a fixture and move on.

**The async fallback chain in `LLMClient`.** The "fail over immediately, never retry the same provider" rule eliminated a whole class of design questions before they arose: no retry count, no exponential backoff, no circuit breaker. The resulting `complete()` method fits in one screen and has a single obvious control flow. The sanitized `LLMUnavailableError` that names only exception types (never raw provider messages) means there is no information leakage in 503 responses — something that would have required a separate sanitization pass if raw errors had been propagated first.

**Structured log lines from Day 4 onward.** Adding `search_latency_ms` as a distinct field (separate from `total_elapsed_ms`) paid off on the first day of real queries. It was immediately obvious that Tavily was consuming roughly two thirds of the total response time — not obvious from the JSON response, but unmistakable in the log. Having `provider`, `model`, `tokens_in`, and `tokens_out` as separate key=value fields also means the Week 4 eval harness can parse them with a simple line split rather than a regex.

**Type-safe API client in TypeScript.** The `ResearchResponse` interface caught a schema mismatch on Day 5 when `mode` was briefly absent from the backend response — TypeScript flagged it before any browser test ran. Adding the Graph mode toggle on Day 6 was mechanical: add a `mode` parameter to `runResearch()`, append `/graph` to the URL, update the interface — no implicit string wrangling. This is a small thing but the friction difference between typed and untyped API clients compounds over multiple days.

---

## What was surprising / harder than expected

**Python 3.14 wheel availability.** The initial setup used a Python 3.14 preview build because it was available in the system PATH. None of the required packages — `pydantic`, `groq`, `google-genai`, `tavily-python` — had binary wheels for 3.14 yet, so pip attempted to compile C extensions from source and failed on multiple packages. Downgrading to 3.13 resolved everything in under five minutes. The lesson: for a project that depends on third-party C-extension packages, check wheel availability on PyPI before picking the Python version. `python-requires` constraints on package metadata are not enforced at `pip install` time until the build fails.

**Gemini free-tier "limit: 0" behavior.** During Day 4 testing, the Gemini fallback was validated by temporarily disabling Groq. The first call returned HTTP 429 with `"quota limit is 0"` — not a standard rate limit, but a model that was not available on the free tier at all under the chosen project settings. This is harder to diagnose than a normal rate limit because the error message looks like a misconfigured quota rather than a missing entitlement. The mock-only test suite would not have caught this. The lesson: test fallback paths against the real fallback provider early — mocks verify the code path, not that the provider will actually respond.

**Tavily latency dominates the response time budget.** Average Tavily response: ~2 seconds. Average Groq generation: under 1 second. Total response time is therefore bounded by search, not generation — which means optimizing the LLM (smaller model, lower temperature) does almost nothing for perceived latency. This was not surprising in retrospect, but it wasn't planned for; the 30-second timeout was set conservatively without any profiling. The lesson: instrument external calls separately from the start (hence `search_latency_ms` as a distinct log field), and when a pipeline has a single dominant cost center, name it explicitly before optimizing anything else.

**Turbopack and OneDrive path reparse points on Windows.** Running `next dev` with the project under an OneDrive-synced directory triggered a Turbopack workspace-root detection failure. Turbopack could not resolve the project root correctly through the OneDrive reparse point, and the dev server started with broken module resolution. Switching to `next dev --no-turbopack` resolved it immediately. Webpack handled the same path without issues. The lesson: Turbopack has edge cases on Windows path configurations (symlinks, reparse points, junction points) that webpack 5 does not; when developing on Windows with synced directories, disable Turbopack until this is resolved upstream.

---

## Risks for Week 2

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Multi-agent orchestration adds latency that exceeds user patience (5 nodes × ~1s LLM call each = 5s+ before a response) | High | Parallelize Searcher sub-queries where possible; target <10s total wall-clock; add incremental progress indication in the frontend |
| Critic → Writer feedback loop infinite-loops if `revision_count` guard is missed or bypassed by a LangGraph edge bug | Medium | Hard-cap at 2 revisions with an explicit LangGraph conditional edge; add a panic test that forces the Critic to always reject and asserts the loop terminates |
| Token costs scale with agent count — 5 nodes means ~5× prompt tokens per query compared to single-pass RAG | High | Factor system prompts across nodes; summarize search context before passing to Writer rather than forwarding the full context block; add `tokens_in` per-node telemetry to identify the biggest consumers |
| Eval harness deferred to Week 4 — no signal on whether multi-agent beats baseline until three weeks after it ships | High | Run a lightweight smoke eval in Week 3 on 10–20 HotpotQA examples to get an early directional signal before committing to the full eval design |
| Gemini fallback path was never exercised against production traffic in Week 1 — first real trigger in Week 2 may surface parsing or auth bugs | Medium | Add a Week 2 end-to-end test that forces Groq to fail (mock at the SDK level) and exercises the full Gemini path including response parsing; verify the 503 message format matches what the frontend expects |

---

## Open questions

1. **Does multi-agent actually outperform baseline on HotpotQA?** Hypothesis: yes for multi-hop questions requiring synthesis across evidence from multiple documents; no for simple single-hop factual lookups where single-pass RAG is already correct. The eval harness in Week 4 is the only way to find out at scale.

2. **What is the right `revision_count` cap?** The current plan is 2. It is not clear whether 2 revisions is enough for the Critic to converge on most query types, or whether some queries routinely need a third pass. This should be observable in the Week 3 smoke eval.

3. **Should the Planner produce a structured output or a freeform plan?** A structured list of sub-questions is more predictable and easier for the Searcher to execute mechanically. A freeform reasoning trace may handle novel or ambiguous query types better but is harder to parse downstream. The answer likely depends on the query distribution.

4. **Does Tavily's `score` field correlate with grounding usefulness?** The current pipeline passes all retrieved results to the LLM in order of Tavily's ranking. If the `score` field does not correlate with how useful a source is for LLM grounding (as opposed to general web relevance), a re-ranking step — even a simple one — might improve citation quality without adding an extra LLM call.

---

## Metrics so far

| Metric | Value |
|--------|-------|
| Average baseline latency (`POST /research`) | ~3s (Tavily ~2s, Groq ~0.9s) |
| Average graph latency (`POST /research/graph`) | ~3s (same pipeline; LangGraph overhead negligible) |
| Tokens per query — input (Groq) | ~1,300 |
| Tokens per query — output (Groq) | ~50 |
| Approximate cost per query at Groq published rates | ~$0.0006 (effectively free for development) |
| Test count | 23 |
| Test suite runtime | ~1.7s |
| Python LOC (backend, excluding venv and generated files) | ~600 |
