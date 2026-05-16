import json
lines = open("eval/hotpot/runs/day21_main_n200.jsonl", encoding="utf-8").readlines()
rows = [json.loads(l) for l in lines]
ok = [(i, r["question_id"], r["question_type"]) for i, r in enumerate(rows, 1)
      if r["baseline"]["status"] == "ok" and r["graph"]["status"] == "ok"]
print(f"Successful (both ok): {len(ok)}/200")
print(f"First 10 positions: {[i for i, _, _ in ok[:10]]}")
print(f"Last 10 positions: {[i for i, _, _ in ok[-10:]]}")
bridge = sum(1 for _, _, t in ok if t == "bridge")
comparison = sum(1 for _, _, t in ok if t == "comparison")
print(f"Type breakdown: bridge={bridge}, comparison={comparison}")
