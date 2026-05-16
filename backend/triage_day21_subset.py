import json
lines = open("eval/hotpot/runs/day21_main_n200.jsonl", encoding="utf-8").readlines()
rows = [json.loads(l) for l in lines]

# Filter to both-ok only
ok_rows = [r for r in rows if r["baseline"]["status"] == "ok" and r["graph"]["status"] == "ok"]
print(f"Both-ok subset: {len(ok_rows)} questions")

# Load scores.json to find per-question F1
scores = json.load(open("eval/hotpot/runs/day21_main_n200-scores.json", encoding="utf-8"))
per_q = {p["question_id"]: p for p in scores["per_question"]}

ok_ids = {r["question_id"] for r in ok_rows}
ok_scored = [per_q[qid] for qid in ok_ids if qid in per_q]

baseline_f1 = [p["baseline_f1"] for p in ok_scored if p.get("baseline_f1") is not None]
graph_f1 = [p["graph_f1"] for p in ok_scored if p.get("graph_f1") is not None]

print(f"Mean baseline F1 over 32 successful: {sum(baseline_f1)/len(baseline_f1):.3f}")
print(f"Mean graph F1 over 32 successful:    {sum(graph_f1)/len(graph_f1):.3f}")
print(f"Baseline > Graph: {sum(1 for p in ok_scored if p['baseline_f1'] > p['graph_f1'])}")
print(f"Graph > Baseline: {sum(1 for p in ok_scored if p['graph_f1'] > p['baseline_f1'])}")
print(f"Tied:             {sum(1 for p in ok_scored if p['baseline_f1'] == p['graph_f1'])}")
