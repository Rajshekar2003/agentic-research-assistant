# Week 4 Retrospective — Scale Hardening

## What shipped

Day 20 hardened the runner for scale. Default `--delay-seconds` bumped to 3.0. A `--max-retries` flag (default 3) added exponential backoff capped at 30s for 429, 503, 504, and network timeouts; retry counts surface in the per-run summary. A `--target` flag lets the runner keep sampling from the same seeded pool until N questions succeed — a safety net for when transient failures leave a run short. Six new retry tests brought the total to 134.

Day 21 ran 200 stratified HotpotQA questions. The runner executed cleanly and absorbed 1,010 retries without crashing. The limiting factor was provider quotas: Groq (LLM) and Tavily (search) both returned 503s mid-run. Questions 1–32 completed both endpoints successfully; the remaining 168 exhausted retries. The scorer skips failed entries, so headline numbers reflect 32 valid pairs, not 200.

## What the numbers say

On the 32 completed pairs: graph F1 0.074, baseline F1 0.055 (graph +35% relative). EM is 0.000 for both — expected, since HotpotQA ground truths are terse single words and both systems produce verbose prose. Refusal rate: baseline 41% (13/32), graph 22% (7/32). Win/loss/tie: graph 14, baseline 10, tied 8.

Type breakdown follows the same direction as Day 19 — bridge F1 baseline 0.058 / graph 0.076, comparison baseline 0.036 / graph 0.060 — but the comparison subset is only 4 questions, too small to interpret separately.

Latency: baseline mean 3.6s (P95 6.4s), graph mean 9.5s (P95 57.2s). The P95 tail is the most actionable figure.

One concrete failure mode worth naming: Q2 (Hawker Hurricane / No. 1455 Flight), where the planner resolved "No. 1455 Flight" to Southwest Airlines flight 1455 instead of an RAF unit. That zeroed the graph on a question where baseline scored F1 0.162. Planner entity-resolution errors propagate through the whole pipeline — wrong sub-entities, wrong search, wrong facts, no downstream recovery point.

N=32 is too small to draw strong conclusions. The direction is consistent with Day 19 (graph attempts more, refuses less, leads on F1), but the margin is narrow enough that noise could account for it.

## What I'd change

Free-tier quota is a hard ceiling for sequential 200-question runs. Real options: upgrade Tavily and Groq tiers, batch the run with quota waits between segments, or accept that N=32 is the realistic ceiling at free tier. Trying to recover the 168 failed questions from the same run would require both providers to reset quota simultaneously.

Graph P95 latency of 57 seconds warrants investigation. The most likely culprits are Searcher (parallel Tavily calls on multi-hop plans) and Critic revision loops (each revise verdict adds a full Writer+Critic round-trip). Per-node wall-time instrumentation would confirm which dominates.

The scorer correctly skips failed JSONL entries, but scores.json has no `successful_n` field — a reader has to count manually to see that "scored 32" means "32 out of 200 attempted." That gap should be explicit in the artifact.
