# Backend Deployment Runbook

## Required environment variables

| Variable | Description |
|---|---|
| `GROQ_API_KEY` | Groq LLM inference key |
| `GOOGLE_API_KEY` | Google Gemini fallback key |
| `TAVILY_API_KEY` | Tavily web search key |
| `APP_ENV` | Set to `production` on Render |
| `LOG_LEVEL` | `INFO` for production; `DEBUG` for diagnosing issues |

## Deploying to Render

1. Push this repo to GitHub (or connect it in the Render dashboard).
2. In the Render dashboard, choose **New → Blueprint** and point it at this repo. Render will detect `backend/render.yaml` and create the service automatically.
3. After the service is created, go to **Environment** for the service and add the three secret keys (`GROQ_API_KEY`, `GOOGLE_API_KEY`, `TAVILY_API_KEY`). These are marked `sync: false` in `render.yaml` and will not deploy correctly until they are set.
4. Trigger a manual deploy or wait for the next push to `main`.

## Testing a deployed instance

Replace `<HOST>` with the Render service URL (e.g. `https://agentic-research-assistant-backend.onrender.com`).

```bash
# liveness check
curl https://<HOST>/health

# baseline research endpoint
curl -s -X POST https://<HOST>/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}' | python -m json.tool

# multi-agent graph endpoint
curl -s -X POST https://<HOST>/research/graph \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}' | python -m json.tool
```

## Known constraint — free tier spin-down

Render free tier suspends the service after 15 minutes of inactivity. The first request after a cold start takes approximately 50 seconds. Options:

- **Keep-alive pinger**: a cron job (external or Render cron service) that hits `/health` every 10 minutes.
- **Paid tier**: Render's Starter plan ($7/month) keeps the instance always-on.

For a public demo, the pinger is the cheapest fix; for anything production-facing, use a paid tier.

## Rolling back

In the Render dashboard, navigate to the service → **Events** → find the last known-good deploy → click **Rollback to this deploy**. Render re-deploys from that commit without any git revert required.
