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

Week 1 Day 3 complete. Next.js 16 + React 19 + Tailwind v4 frontend scaffolded. Single-page research interface (`frontend/app/page.tsx`) with type-safe API client (`frontend/lib/api.ts`). No UI libraries; useState only; 30s timeout; plain fetch. Production build verified. Ready for Day 4 (LangGraph agent wiring).

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
