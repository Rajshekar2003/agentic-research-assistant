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

Week 2 Day 13 complete — smoke eval harness in `backend/eval/`. Runs 8-10 curated questions through both `/research` and `/research/graph` endpoints, writes a markdown report with paired answers ready for human scoring on a 4-criterion rubric (groundedness, citation accuracy, completeness, hallucination). First eval report committed at `eval/reports/smoke-eval-{date}.md`. ~69 pytest tests still passing (eval is not unit-tested — it's a thin script with no logic to test).

**Eval vs tests:** The eval harness in `backend/eval/` is intentionally NOT in the pytest suite. It is an integration-only artifact that requires real APIs (Tavily, Groq) and a running server (`uvicorn`). Pattern: `backend/tests/` for code correctness (mocked, fast, CI-safe); `backend/eval/` for behavioral quality (live APIs, human-graded, run manually before retros).

## Architecture

Current graph (Day 11): **5 nodes** — `planner` decomposes query into 2-4 sub-questions; `searcher` runs Tavily per sub-question, deduplicates URLs globally (cap 8); `fact_checker` extracts verified {claim, sources} pairs; `writer` synthesizes final answer (revision-aware: uses different system prompt + previous draft + critique on revision passes); `critic` evaluates draft against verified facts and routes: approve → END, revise (revision_count < 2) → writer, revise (revision_count ≥ 2) → END (hard cap). `revision_count` is incremented ONLY by Critic on "revise" verdicts.

Current baseline (locked after Day 5): **Search (Tavily) → Generate (Groq w/ Gemini fallback)** — a single-pass RAG system, not multi-agent yet.  Served by POST /research; must not change.

Week 2 target (multi-agent, for eval comparison): **5 nodes** — DEPLOYED Day 11.  Days 12-14 are tuning, parallelization, smoke eval, and Week 2 retro.  Served by POST /research/graph.

## Known issues

- Token counting in Gemini fallback path may show None for tokens_in/tokens_out depending on SDK version — non-blocking, eval harness in Week 4 will use prompt_token_count from native API responses.

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
