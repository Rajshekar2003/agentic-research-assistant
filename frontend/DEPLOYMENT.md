# Frontend Deployment Runbook

## Required environment variables

| Variable | Description |
|---|---|
| `NEXT_PUBLIC_API_URL` | Base URL of the backend API. Must be set to the live Render URL for production. |

## Deploying to Vercel

1. Push the repo to GitHub (already done).
2. In the Vercel dashboard, choose **Add New → Project** and import the GitHub repo.
3. Set the root directory to `frontend/` — the repo has both backend and frontend in subdirectories and Vercel needs to know where to look.
4. Vercel auto-detects Next.js. No build command override is needed.
5. In **Environment Variables**, add `NEXT_PUBLIC_API_URL` with the value `https://agentic-research-assistant-backend.onrender.com` scoped to the **Production** environment.
6. Click **Deploy**.

## Testing

Once deployed, visit the Vercel URL. Submit a sample query in the UI. Confirm:

- A successful response renders with an answer and source cards (if Tavily quota is available).
- OR a 503 error renders gracefully if Tavily quota is exhausted — this is expected behavior; see backend known limitations.

## Known constraints

- **Render free-tier sleep**: the backend suspends after 15 minutes of inactivity. The first request after a cold start takes approximately 50 seconds. The client timeout is set to 60 seconds to absorb this.
- **Tavily quota**: the free-tier allowance (1000 credits/month) is shared with eval runs. When exhausted, `/research` returns 503 until the monthly reset.

## Rolling back

Vercel dashboard → **Deployments** tab → find a previous deploy → **Promote to Production**.
