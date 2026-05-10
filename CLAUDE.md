# Agentic Research Assistant — CLAUDE.md

## Project overview

Multi-agent research assistant. FastAPI backend + LangGraph agent graph + Next.js frontend (Day 3+).

## Repo layout

```
backend/        FastAPI app + LangGraph graph + tests
frontend/       Next.js (TBD Day 3)
docs/           Architecture docs and diagrams
```

## Backend conventions

- Python 3.11+
- Line length: 100 (Black)
- All settings loaded via `app.config.Settings` (pydantic-settings); never hard-code keys
- Tests live in `backend/tests/`; run with `pytest` from inside `backend/`
- `asyncio_mode = auto` — no need for `@pytest.mark.asyncio`

## Key files

| File | Purpose |
|------|---------|
| `backend/app/main.py` | FastAPI app factory, CORS, health endpoint |
| `backend/app/config.py` | Settings / env loading |
| `backend/app/schemas.py` | Shared Pydantic models |
| `backend/app/api/research.py` | POST /research entry point |
| `backend/app/graph/state.py` | LangGraph `ResearchState` TypedDict |
| `backend/app/graph/workflow.py` | LangGraph compiled graph |

## Current status

Week 2 complete. Multi-agent graph: Planner → Searcher → FactChecker → Writer → Critic, with conditional feedback loop capped at 2 revisions. Searcher parallelized via `asyncio.gather`. 69 pytest tests passing. Smoke eval at `backend/eval/reports/smoke-eval-2026-05-08.md` graded with graph wins on 4/8 questions, ties on 3/8, baseline wins on 1/8. Average latencies: baseline 4.0s, graph 12.4s (~3.1x cost). Critic revision rate ~25% across live tests. Week 2 retro at `docs/week2-retro.md`. Week 3 begins HotpotQA eval harness.

### Week 3 — Eval pipeline

Week 3 Day 15 complete — HotpotQA loader at `backend/eval/hotpot/loader.py`. Downloads + caches the dev set (~7,400 questions in distractor setting), exposes a `HotpotQuestion` dataclass, deterministic seeded sampling with type/level filters, and a stats helper. Cached JSON is gitignored. ~76 pytest tests passing (69 from Week 2 + 7 new loader tests).

Week 3 Day 16 complete — HotpotQA runner at `backend/eval/hotpot/runner.py`. Sequential calls to baseline and graph endpoints, JSONL output for crash-safety, resume-on-rerun, configurable delay between questions. Smoke run on 5 questions takes ~2 minutes; production runs of 100-200 questions are 30-50 minutes. ~84 tests passing (76 from Day 15 + 8 new runner tests).

**Eval vs tests:** The eval harness in `backend/eval/` is intentionally NOT in the pytest suite. It is an integration-only artifact that requires real APIs (Tavily, Groq) and a running server (`uvicorn`). Pattern: `backend/tests/` for code correctness (mocked, fast, CI-safe); `backend/eval/` for behavioral quality (live APIs, human-graded, run manually before retros).

## Architecture

Current graph (Day 11): **5 nodes** — `planner` decomposes query into 2-4 sub-questions; `searcher` runs Tavily per sub-question, deduplicates URLs globally (cap 8); `fact_checker` extracts verified {claim, sources} pairs; `writer` synthesizes final answer (revision-aware: uses different system prompt + previous draft + critique on revision passes); `critic` evaluates draft against verified facts and routes: approve → END, revise (revision_count < 2) → writer, revise (revision_count ≥ 2) → END (hard cap). `revision_count` is incremented ONLY by Critic on "revise" verdicts.

Current baseline (locked after Day 5): **Search (Tavily) → Generate (Groq w/ Gemini fallback)** — a single-pass RAG system, not multi-agent yet.  Served by POST /research; must not change.

Week 2 target (multi-agent, for eval comparison): **5 nodes** — DEPLOYED Day 11.  Days 12-14 are tuning, parallelization, smoke eval, and Week 2 retro.  Served by POST /research/graph.

## Architecture decisions worth defending

- **Loader does not include HotpotQA's context paragraphs.** Our system retrieves its own evidence via Tavily, so giving it the dataset's own source passages would defeat the eval. The `HotpotQuestion` dataclass stores titles only — not passage text.
- **JSONL output format chosen over single-JSON for resumability.** Each question is one self-contained line, atomically appendable with a flush after each write. A crash mid-run leaves a valid partial file. The scorer (Day 17) reads with line-by-line `json.loads`; re-writing a full JSON array on every append would risk corruption on crash.

## Known issues / limitations

- Token counting in Gemini fallback path may show None for tokens_in/tokens_out depending on SDK version — non-blocking, eval harness in Week 4 will use prompt_token_count from native API responses.
- Graph latency variance: 3–26s observed across smoke eval queries; no defined p95 budget before deployment.
- Soft hallucinations possible when retrieved sources contain speculative claims — Critic evaluates the draft against retrieved facts, not ground truth, so a misleading source that passes Searcher can persist to the final answer.
- Critic does not currently use a source-quality signal; all Tavily results are treated as equally credible.
- HotpotQA's distractor setting includes adversarial paragraphs; our system uses Tavily, not the dataset's context, so we evaluate against open-web retrieval rather than the closed-book setting common in published HotpotQA scores. Numbers won't be directly comparable to published baselines but trends will be interpretable.

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
