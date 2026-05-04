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

Week 1 Day 5 complete. Tavily search integrated; /research endpoint is now full RAG (search → context-formatted prompt → grounded answer with [1]/[2] citations → sources array). Frontend renders source cards. 12+ pytest tests passing including search mocks. This is the locked baseline — Week 2 multi-agent graph will be evaluated against it.

## Architecture

Current baseline (locked after Day 5): **Search (Tavily) → Generate (Groq w/ Gemini fallback)** — a single-pass RAG system, not multi-agent yet.

Week 2 target (multi-agent, for eval comparison): Agents in a LangGraph state machine: Planner → Searcher → Fact-checker → Writer → Critic.

## Known issues

- Token counting in Gemini fallback path may show None for tokens_in/tokens_out depending on SDK version — non-blocking, eval harness in Week 4 will use prompt_token_count from native API responses.

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
