# Week 4 Close

## State at end of Week 4

**What shipped:**
- 5-node multi-agent graph (Planner → Searcher → FactChecker → Writer → Critic) with Critic→Writer feedback loop capped at 2 revisions, served at `POST /research/graph`
- Baseline endpoint (`POST /research`) locked since Day 5 as a clean eval comparison point
- Eval infrastructure: loader, runner, scorer, reporter (`backend/eval/hotpot/`)
- 134 pytest tests passing, all mocked, CI-safe
- Two eval runs: Day 19 smoke N=25 (stratified sample), Day 21 pilot N=32 effective out of 200 attempted

**Headline result (Day 21, N=32):**

| | Baseline | Graph |
|---|---|---|
| F1 | 0.055 | 0.074 |
| EM | 0.000 | 0.000 |
| Refusal rate | 41% (13/32) | 22% (7/32) |

EM is 0 for both systems — an artifact of HotpotQA expecting terse single-word answers against verbose prose outputs. Refusal tracking compensates partially; the graph refuses less and leads on F1.

**Counts:** 134 tests, 5 doc files in `docs/` (architecture, week2-retro, week4-retro, week4-close, plus diagrams placeholder).

---

## What enters Week 5

- **Deployment:** backend to Railway, frontend to Vercel
- **Rate limiting and auth:** basic, enough to survive public exposure without exhausting provider quotas on a live URL
- **Open decision:** whether to upgrade Tavily/Groq tier for a larger second eval run, or accept the Day 21 N=32 pilot as the final eval numbers for the resume artifact

---

## Risks for Week 5

- Railway free-tier cold-start latency will worsen the graph's already-wide P95 (57s observed in Day 21). A cold-instance response may feel broken to a user with no context on what the graph does internally.
- Rate limiting and auth are new code paths that need their own tests before going live; shipping untested auth against a public endpoint is a known risk.
- The portfolio narrative relies on the eval numbers being defensible. They are, but N=32 is small — the graph's F1 lead (0.074 vs 0.055) is directionally consistent with the Day 19 smoke run, not statistically conclusive.
