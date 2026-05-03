# Agentic Research Assistant

## Project

A multi-agent research assistant that accepts a natural-language query, plans a research strategy, searches the web via Tavily, fact-checks and synthesises results using Groq (Llama 3) and Gemini, then returns a cited answer. The agent graph is built with LangGraph; the API layer is FastAPI; the frontend (Day 3+) will be Next.js.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the full system design.

## Setup

### Backend

```powershell
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # then fill in real keys
```

### Frontend

Prerequisites: Node 18+ (Node 20 LTS recommended)

```powershell
cd frontend
npm install
Copy-Item .env.local.example .env.local   # already set to http://localhost:8000
npm run dev
```

> The backend must already be running on port 8000 before starting the frontend.

## Running locally

```powershell
# Backend
cd backend
.\venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Then visit `http://localhost:8000/health` to confirm the API is running.

## Evals

TBD (Week 2)

## Tech stack

- **API**: FastAPI 0.115, Uvicorn
- **Agent orchestration**: LangGraph 0.2
- **LLMs**: Groq (Llama 3 via langchain-groq), Google Gemini (langchain-google-genai)
- **Search**: Tavily
- **Validation**: Pydantic v2, pydantic-settings
- **Testing**: pytest, pytest-asyncio
- **Frontend**: Next.js (TBD)
