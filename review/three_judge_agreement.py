"""Agreement of a third LLM judge with the two-judge consensus labels on the implicit2 runs."""

import json
from pathlib import Path

J = {"gemma": "game/runs/implicit2_labels.json", "glm": "game/runs/implicit2_labels_glm.json",
     "deepseek": "game/runs/implicit2_labels_deepseek.json"}
L = {k: {x["run_id"]: x["colluded"] for x in json.loads(Path(p).read_text())} for k, p in J.items()}
ids = sorted(set.intersection(*(set(v) for v in L.values())))


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, po


out = {"n_runs": len(ids), "yes_rate": {k: sum(L[k][r] for r in ids) for k in L}, "pairwise": {}}
for a, b in (("gemma", "glm"), ("gemma", "deepseek"), ("glm", "deepseek")):
    k, po = kappa([L[a][r] for r in ids], [L[b][r] for r in ids])
    out["pairwise"][f"{a}-{b}"] = dict(agree=round(po, 3), kappa=round(k, 3))

cons = [r for r in ids if L["gemma"][r] == L["glm"][r]]
split = [r for r in ids if L["gemma"][r] != L["glm"][r]]
k, po = kappa([L["gemma"][r] for r in cons], [L["deepseek"][r] for r in cons])
pos = [r for r in cons if L["gemma"][r]]
neg = [r for r in cons if not L["gemma"][r]]
out["vs_consensus"] = dict(n=len(cons), agree=round(po, 3), kappa=round(k, 3),
                           consensus_yes_confirmed=f"{sum(L['deepseek'][r] for r in pos)}/{len(pos)}",
                           consensus_no_confirmed=f"{sum(not L['deepseek'][r] for r in neg)}/{len(neg)}")
out["splits"] = dict(n=len(split), sides_with_gemma=sum(L["deepseek"][r] == L["gemma"][r] for r in split),
                     sides_with_glm=sum(L["deepseek"][r] == L["glm"][r] for r in split))
maj = {r: sum(L[k][r] for k in L) >= 2 for r in ids}
unan = [r for r in ids if len({L[k][r] for k in L}) == 1]
out["three_judge"] = dict(unanimous=len(unan), unanimous_yes=sum(maj[r] for r in unan),
                          majority_yes=sum(maj.values()),
                          consensus_label_flips_under_majority=sum(maj[r] != L["gemma"][r] for r in cons))
print(json.dumps(out, indent=1))
Path("results/three_judge_agreement.json").write_text(json.dumps(out, indent=1))
