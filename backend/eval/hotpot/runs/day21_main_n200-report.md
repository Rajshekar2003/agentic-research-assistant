# HotpotQA Eval Report — 2026-05-15

**System:** 5-node multi-agent graph (Planner → Searcher → FactChecker → Writer → Critic) vs single-node RAG baseline.
**Dataset:** HotpotQA dev set (distractor setting), 200 questions sampled.
**Metrics:** Exact Match (EM) and token-level F1, computed by direct reproduction of HotpotQA's official hotpot_evaluate_v1.py.

---

## Headline numbers

|              | Baseline   | Graph      |
|--------------|------------|------------|
| Exact Match  | 0.000      | 0.000      |
| F1           | 0.055      | 0.074      |

200 questions scored. 32 baseline OK, 32 graph OK.

## By question type

|                  | Comparison (30) | Bridge (170) |
|------------------|-----------------|--------------|
| Baseline EM      | 0.000           | 0.000        |
| Baseline F1      | 0.036           | 0.058        |
| Graph EM         | 0.000           | 0.000        |
| Graph F1         | 0.060           | 0.076        |

## Latency

| Endpoint  | Mean (ms) | Median (ms) | P95 (ms) |
|-----------|-----------|-------------|----------|
| Baseline  | 3599      | 3363        | 6372     |
| Graph     | 9539      | 5352        | 57238    |

## Honest refusals

The HotpotQA EM/F1 metrics do not distinguish between confident-but-wrong answers and honest "I don't know" refusals — both score 0. Identifying refusals separately is therefore important for interpreting the headline numbers.

A prediction is considered an "honest refusal" if it matches one of these patterns (case-insensitive):

- "couldn't verify"
- "could not verify"
- "do not contain information"
- "do not contain enough information"
- "no information available"
- "cannot be determined"
- "I don't have"
- "I do not have"

**Refusal counts:**
- Baseline refusals: 13 / 32
- Graph refusals: 7 / 32

**Refusal examples:**

**Example 1:**
- **Question:** Isabella Kelly was born at a ruined castle characterized as one of the most isolated fortifications in Britain by who?
- **Ground truth:** `The Changing Scottish Landscape`
- **Graph answer:** I couldn't verify any claims from the available sources for this question.

**Example 2:**
- **Question:** What is the name of the Australian specialist electronic music magazine that features avant-rock, experimental sound art, and experimental hip hop?
- **Ground truth:** `Cyclic Defrost`
- **Graph answer:** I couldn't verify any claims from the available sources for this question.

**Example 3:**
- **Question:** What is the population of the city that Munsonville is in the northwest corner of?
- **Ground truth:** `729 at the 2010 census`
- **Graph answer:** The provided facts do not include population data for the town of Nelson, where Munsonville is located in the northwest corner [1, 2]. Therefore, the population of the town where Munsonville is situated cannot be determined based on the available information. The verified facts only confirm the location of Munsonville within the town of Nelson [1, 2]. The population information is not available based on the given data.

## Sample question pairs

For qualitative inspection, here are 4 questions chosen to span the score distribution: 1 where graph clearly beat baseline by F1, 1 where baseline beat graph, 1 where both scored well, 1 where both scored 0 (refusal or genuine failure).

### Q1: The Prussian General Carl von Clausewitz is associated with what type of realism?

**Ground truth:** `Modern thinkers associated with classical realism are Carl von Clausewitz`

**Baseline (3834ms):** Carl von Clausewitz is considered to be a traditional (old) realist [1]. This classification is based on his viewpoint of warfare, which echoed the ideas of the Greek Thucydides [1]. Additionally, his focus on the political aspects of war, as seen in his famous aphorism "War is the continuation of policy with other means" [2], further supports his association with traditional realism.

- EM: 0 | F1: 0.149

**Graph (3311ms, plan_size=N/A, facts=3, verdict=approve, revisions=0):** Carl von Clausewitz is associated with classical realism [1] and is considered a traditional (old) realist [3], with his work also being explored through the lens of Existential Imperative Realism [2].

- EM: 0 | F1: 0.359

_Graph wins by F1; FactChecker extracted 3 verified fact(s)._

### Q2: What where both Hawker Hurricane and No. 1455 Flight apart of?

**Ground truth:** `Royal Air Force`

**Baseline (3462ms):** The Hawker Hurricane was part of the Royal Air Force (RAF) [3], and No. 1455 Flight was located in Tangmere [1]. Additionally, the Hawker Hurricane is also part of the Battle of Britain Memorial Flight [3] and [5].

- EM: 0 | F1: 0.162

**Graph (5648ms, plan_size=N/A, facts=3, verdict=approve, revisions=1):** Hawker Hurricane was part of World War II [2]. Southwest Airlines flight 1455 was part of a 3-day flight sequence [3] that consisted of five flights [3]. Both were part of distinct events, with Hawker Hurricane involved in a major conflict [2] and Southwest Airlines flight 1455 being part of a specific flight sequence [3].

- EM: 0 | F1: 0.000

_Baseline wins by F1._

### Q3: what language did the ethnic group which Torstein Ellingsen was its drumer  speaks 

**Ground truth:** `Norwegian language`

**Baseline (2293ms):** The ethnic group that Torstein Ellingsen was a part of is Norwegian, as he is a Norwegian drummer and music producer [1]. The language spoken by Norwegians is Norwegian.

- EM: 0 | F1: 0.148

**Graph (4596ms, plan_size=N/A, facts=1, verdict=approve, revisions=0):** The language spoken by the ethnic group of Torstein Ellingsen's band is Norwegian [1].

- EM: 0 | F1: 0.267

_Graph wins by F1; FactChecker extracted 1 verified fact(s)._

### Q4: In what year was the university where Sergei Aleksandrovich Tokarev was a professor founded?

**Ground truth:** `1755`

**Baseline (2111ms):** The sources provided do not contain enough information to answer the question about the founding year of the university where Sergei Aleksandrovich Tokarev was a professor. While [1] and [2] mention that Tokarev was a professor at Moscow State University, they do not provide the founding year of the university.

- EM: 0 | F1: 0.000

**Graph (3488ms, plan_size=N/A, facts=2, verdict=approve, revisions=0):** The founding year of Moscow University, where Sergei Aleksandrovich Tokarev was a professor [1, 2], is not explicitly mentioned in the provided sources [1, 2].

- EM: 0 | F1: 0.000

_Baseline refused; graph attempted an answer._

## Limitations

- HotpotQA's EM and F1 do not credit honest refusals; refer to the dedicated refusal section above.
- Open-web retrieval via Tavily differs from the closed-book setting common in published HotpotQA scores. Trends are interpretable but absolute numbers are not directly comparable to the original paper's baselines.
- Sample size for this run: 200 questions. Statistical significance is limited below ~50 questions.
- Smoke runs may include questions with sparse or speculative source coverage; bridge questions especially may suffer when the second hop requires entity resolution from the first hop's output.

## Run metadata

- Source JSONL: `eval\hotpot\runs\day21_main_n200.jsonl`
- Source scores: `eval\hotpot\runs\day21_main_n200-scores.json`
- Generated: `2026-05-15T14:18:45Z`
