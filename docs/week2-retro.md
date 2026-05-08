# Week 2 Retrospective — Multi-Agent Graph

## What shipped

Week 2 started from the single-node LangGraph wrapper established in Day 6 — a trivially wrapped copy of the baseline RAG pipeline whose only purpose was to prove the LangGraph plumbing worked. It ended with a five-node graph: Planner → Searcher → FactChecker → Writer → Critic, with a conditional Critic→Writer feedback loop capped at two revisions. Each node is a standalone async function with a single responsibility. Latency went from ~3.5s (baseline) to ~6–12s (graph, median across the smoke eval), in exchange for decomposed sub-question planning, parallel multi-source retrieval, verified-claim extraction, source-grounded synthesis, and self-critique with targeted revision. The Searcher's Tavily calls were parallelized on Day 12, cutting the search phase from ~N×2s to ~2s regardless of plan size. A smoke eval on Day 13 graded 8 curated questions and found the graph wins on 4, ties on 3, and loses on 1 (a simple factual lookup where the baseline's richer context outweighed the graph's verified-only constraint).

---

## Day-by-day summary

| Day | Commit | What landed | Tests at end of day |
|-----|--------|-------------|---------------------|
| Day 8 | `43a7418` | Planner → Searcher graph foundation; Planner decomposes query into 2–4 sub-questions via LLM; Searcher runs Tavily per sub-question with global URL dedup (cap 8) | 33 |
| Day 9 | `0bcf0fb` | FactChecker added; extracts `{claim, sources}` pairs from search results; Searcher made retrieval-only | 40 |
| Day 10 | `c8cbd8e` | FactChecker → Writer split; FactChecker keeps claim extraction; Writer gets dedicated synthesis prompt; each node has one job | 47 |
| Day 11 | `fb767e8` | Critic + conditional feedback loop: approve → END, revise (revision_count < 2) → Writer, revise (revision_count ≥ 2) → END | 64 |
| Day 12 | `0e47f49` | Searcher Tavily calls parallelized via `asyncio.gather`; explicit per-layer timeouts added; 504 vs 503 distinction documented | 69 |
| Day 13 | `4dedbe7` | Smoke eval harness: 8 curated questions, paired baseline + graph answers, markdown report for human scoring on a 4-criterion rubric | 69 (no new tests — eval harness is not unit-tested) |

---

## Architecture decisions worth defending

- **Agent-as-function pattern (no classes, no inheritance).** Each agent is a module with a single `async def run(state)` function. No base classes, no shared state, no framework inheritance. This makes each agent independently testable with a plain dict fixture, and the only coupling between agents is the `ResearchState` TypedDict, which is documented in one place.

- **Searcher became retrieval-only on Day 9.** The original Day 8 design had Searcher responsible for both retrieval and synthesis. Moving synthesis to a dedicated node separated concerns: a Searcher failure now unambiguously means "Tavily is down" — no LLM involved in the diagnosis.

- **FactChecker → Writer split on Day 10.** Originally, FactChecker both extracted claims and synthesized an answer in one prompt. Splitting into two nodes gave each a tighter single-responsibility prompt and made it possible to attribute quality failures to the right step in telemetry.

- **Critic falls back to `approve` on any JSON parse failure.** Three layers of validation: JSON parse, schema check (`verdict` must be `"approve"` or `"revise"`), consistency check (`revise` with an empty critique forces `approve`). On any failure the Critic approves and ships the Writer's current draft. Skipping verification is no worse than having no Critic at all; an infinite loop would be catastrophically worse.

- **Hard cap at 2 revisions.** The `route_after_critic` function in `workflow.py` enforces this unconditionally: `revision_count >= 2` routes to END regardless of the Critic's verdict. This prevents a prompt calibration mistake from turning a single query into an infinite loop.

- **504 vs 503 distinction.** 503 = a downstream provider (Tavily, Groq, Gemini) is unavailable. 504 = our own pipeline exceeded its time budget. This distinction matters for on-call routing: a 503 spike means "check the provider status page"; a 504 spike means "check query complexity or add a faster path."

- **Mocked tests only (no integration tests in CI).** All 69 tests mock external services at the module-attribute level via `monkeypatch`. The suite runs in ~1.5s with no network and no secrets. The accepted tradeoff — mocks cannot catch SDK parsing bugs or real provider behavior — is covered by the smoke eval harness for behavioral quality.

- **Smoke eval uses human grading, not LLM-as-judge.** Using the same LLM that produced the answers to grade them would introduce self-evaluation bias. The Day 13 report is a markdown file with filled score tables graded by a human (with mentor input). Week 4's LLM-as-judge approach will be calibrated against this human baseline.

---

## What worked

- **Verification protocol: test in correct mode → paste logs → commit.** This discipline caught two real bugs during the week. On Day 9, running the test suite in baseline mode masked a missing Searcher node — the logs revealed this immediately when pasted. On Day 10, code was written but never committed — `git status` caught it before moving on. Formalizing "test-paste-commit" as the session rule made both recoveries fast.

- **LangGraph's `add_conditional_edges` API.** The `route_after_critic` pattern — a plain function that reads `state` and returns a node name or `END` — was cleaner than anticipated. No framework magic, no callbacks; the routing logic fits in 10 lines and is testable in isolation.

- **`asyncio.gather` parallelization landed with no debugging.** The change from sequential Tavily calls to `asyncio.gather` on Day 12 was a self-contained diff: replace the `for` loop with `gather`, wrap in `asyncio.wait_for`, handle `TimeoutError`. No race conditions, no shared mutable state, no test failures.

- **Critic prompt calibration hit the target band without manual tuning.** The goal was a 15–50% revision rate — low enough to avoid thrashing, high enough to prove the loop is active. The smoke eval saw 2 revision triggers out of 8 queries (25%), both on complex multi-step questions (q04 multi-hop, q05 current events). The "be calibrated, not a perfectionist" framing in the system prompt did the work.

- **The `"I couldn't verify any claims"` fallback held up on every unanswerable query.** The Writer short-circuits with a graceful fallback when `facts` is empty, without calling the LLM. The smoke eval's q06 (NeurIPS 2026 keynote slide) and q08 (Brazilian 2022 World Cup false premise) both triggered honest refusals without fabricating claims.

---

## What didn't work / what I'd change

- **The "ok next" pattern.** Several times during the week, a prompt ended with "ok next" before the verification log was pasted. Each time, the next session had to re-establish context — checking whether the prior commit landed, re-running tests, or discovering that a step was only half done. The test-paste-commit rule now formalizes what "done" looks like before moving on.

- **Smoke eval scope reduced due to time pressure.** The Day 13 eval was run and graded in one session with mentor input rather than a fully independent read-and-score. The grading is directionally correct but not a controlled evaluation. Full independent grading is deferred to the Week 4 HotpotQA eval, which will use a rubric-driven approach with enough examples to get statistical signal.

- **Soft hallucination on q03 (Tailwind v3 vs v4 comparison).** The baseline produced a "v4 was rewritten in Rust, v3 was JavaScript" framing — misleading, but traceable to a speculative source in Tavily's results. The Critic did not flag it because the cited source did say it. This is structural: both endpoints trust all retrieved sources equally. A source-quality signal would help penalize speculative content relative to official documentation.

- **Graph latency variance is too wide for confident deployment.** q05 (AWS re:Invent 2024) took 25.6s — the longest query in the eval. q07 (React popularity) took 15.8s for no quality gain over baseline. The range across 8 queries was roughly 3–26s. Before the graph endpoint is deployed to users, there needs to be a defined latency budget and a way to short-circuit for queries the graph handles no better than baseline.

---

## Metrics

| Metric | Value |
|--------|-------|
| Tests — start of Week 2 (Week 1 end state) | 23 |
| Tests — end of Day 11 | 64 |
| Tests — end of Week 2 (Day 12) | 69 |
| Tests added by eval harness (Day 13) | 0 (intentional — eval harness requires live APIs) |
| Test suite runtime | ~1.5s |
| Smoke eval — baseline total score | 58 / 64 |
| Smoke eval — graph total score | 63 / 64 |
| Graph wins | 4 questions (q02 RAFT, q03 Tailwind, q04 multi-hop, q05 AWS re:Invent) |
| Baseline wins | 1 question (q01 Mongolia — baseline's richer context outweighed graph's verified-only constraint) |
| Ties | 3 questions (q06 unanswerable, q07 React, q08 false premise) |
| Average latency — baseline | 3985ms |
| Average latency — graph | 12443ms (~3.1× cost) |
| Graph latency range (smoke eval, 8 queries) | ~3s (factual lookups) to ~25.6s (q05 current events) |
| Critic revision rate (live eval, 8 queries) | 2/8 queries triggered at least one revision (~25%) |

---

## Decisions deferred to later weeks

- **LLM-as-judge evaluation** — Week 4. The smoke eval used human grading deliberately; Week 4 will compare LLM-as-judge ratings against the human baseline from Day 13.
- **HotpotQA benchmark run** — Week 4. The smoke eval gave directional signal; HotpotQA will provide statistical significance across 100+ multi-hop questions.
- **Frontend display of `facts` array and `revisions` count** — Week 5 (deployment polish pass). The API already returns these fields; the frontend does not surface them yet.
- **Source-quality signal to penalize speculative sources** — Week 6 stretch goal. The q03 hallucination demonstrated the system trusts all retrieved sources equally; a re-ranking step or source credibility heuristic would help.
- **Streaming responses** — out of scope for this project; would require SSE or WebSocket changes to both backend and frontend.
- **Caching and rate-limiting on the API** — Week 5, before public deployment.

---

## What's next

Week 3 builds the evaluation infrastructure for HotpotQA — a multi-hop question-answering benchmark that stress-tests exactly the kind of decomposition and fact-grounding the graph was designed for. Week 4 runs that benchmark on both endpoints and produces the quantitative comparison the smoke eval previewed qualitatively. Week 5 deploys the full system to Railway (backend) and Vercel (frontend), with a final polish pass on the frontend to surface the `facts` array and revision count from the graph path. Week 6 is the final writeup and demo.
