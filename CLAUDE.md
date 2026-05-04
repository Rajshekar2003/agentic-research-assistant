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

Week 1 Day 6 complete. POST /research/graph endpoint runs the same RAG pipeline through a single-node LangGraph state machine. POST /research (baseline) untouched — kept as permanent eval comparison point. Frontend has Baseline/Graph toggle. 19+ pytest tests passing. Ready for Day 7 (architecture diagram + Week 1 retro). Week 2 will replace the single graph node with planner → searcher → fact_checker → writer → critic flow.

## Architecture

Current graph (Day 6): **1 node** — `research_node` = search + write, same pipeline as the baseline.  The `mode` field in ResearchResponse (`"baseline"` | `"graph"`) lets the Week 4 eval harness compare both paths on the same HotpotQA queries.

Current baseline (locked after Day 5): **Search (Tavily) → Generate (Groq w/ Gemini fallback)** — a single-pass RAG system, not multi-agent yet.  Served by POST /research; must not change.

Week 2 target (multi-agent, for eval comparison): **5 nodes** in a LangGraph state machine: Planner → Searcher → Fact-checker → Writer → Critic.  Will be served by POST /research/graph (replacing the single-node graph).

## Known issues

- Token counting in Gemini fallback path may show None for tokens_in/tokens_out depending on SDK version — non-blocking, eval harness in Week 4 will use prompt_token_count from native API responses.

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
