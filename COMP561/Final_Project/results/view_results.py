import json
import numpy as np

with open("results/final_blast_results.json", "r") as js:
    results = json.load(js)

avg_score = []
match_lens = []
p_vals = []
for hsp, content in results.items():
    avg_score.append(content["score"])
    match_lens.append(len(content["seqs"][1]))
    p_vals.append(content["p-value"])

print(f"Num matches={len(avg_score)}")
print(f"Average score={np.average(avg_score)}")
print(f"Avg len={np.average(match_lens)}")
print(f"Avg p-value={np.average(p_vals)}")
