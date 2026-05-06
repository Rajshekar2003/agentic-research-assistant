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

Week 2 Day 10 complete. Multi-agent graph: Planner → Searcher → FactChecker → Writer (4 nodes). FactChecker is now verification-only (extracts {claim, sources} pairs); Writer synthesizes the final answer from verified facts. When facts are empty, Writer skips the LLM and returns an honest "cannot verify" message. ~45 pytest tests passing. Day 11 adds Critic with conditional edge back to Writer.

## Architecture

Current graph (Day 10): **4 nodes** — `planner` decomposes the query into 2-4 sub-questions; `searcher` runs Tavily per sub-question, deduplicates URLs globally (cap 8), retrieval-only; `fact_checker` extracts verified {claim, sources} pairs from search results; `writer` synthesizes the final answer from verified facts only (skips LLM when facts are empty).  The `mode` field in ResearchResponse (`"baseline"` | `"graph"`) lets the Week 4 eval harness compare both paths on the same HotpotQA queries.  The `facts` array exposes each verified claim with its supporting source IDs.

Current baseline (locked after Day 5): **Search (Tavily) → Generate (Groq w/ Gemini fallback)** — a single-pass RAG system, not multi-agent yet.  Served by POST /research; must not change.

Week 2 target (multi-agent, for eval comparison): **5 nodes** in a LangGraph state machine: Planner → Searcher → FactChecker → Writer → Critic.  Day 11 adds Critic with conditional feedback loop (5 nodes).  Served by POST /research/graph.

## Known issues

- Token counting in Gemini fallback path may show None for tokens_in/tokens_out depending on SDK version — non-blocking, eval harness in Week 4 will use prompt_token_count from native API responses.

## Dev workflow

```powershell
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```
