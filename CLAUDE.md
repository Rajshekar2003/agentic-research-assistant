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

Week 2 Day 8 complete. Multi-agent foundation: Planner → Searcher graph (2 nodes). Planner decomposes the query into 2-4 sub-questions; Searcher runs Tavily per sub-question with global URL dedup and an 8-result cap, then synthesizes a grounded answer. Baseline endpoint untouched. ~30 pytest tests passing. Day 9 will add FactChecker between Searcher and Writer.

## Architecture

Current graph (Day 8): **2 nodes** — `planner` decomposes the query into 2-4 sub-questions; `searcher` runs Tavily per sub-question, deduplicates URLs globally (cap 8), and calls the LLM to synthesize a grounded answer.  The `mode` field in ResearchResponse (`"baseline"` | `"graph"`) lets the Week 4 eval harness compare both paths on the same HotpotQA queries.

Current baseline (locked after Day 5): **Search (Tavily) → Generate (Groq w/ Gemini fallback)** — a single-pass RAG system, not multi-agent yet.  Served by POST /research; must not change.

Week 2 target (multi-agent, for eval comparison): **5 nodes** in a LangGraph state machine: Planner → Searcher → FactChecker → Writer → Critic.  Days 9-11 will add FactChecker, a dedicated Writer node (splitting the LLM call out of Searcher), and Critic with conditional feedback loop.  Served by POST /research/graph.

## Known issues

- Token counting in Gemini fallback path may show None for tokens_in/tokens_out depending on SDK version — non-blocking, eval harness in Week 4 will use prompt_token_count from native API responses.

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
