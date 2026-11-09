"""Score review/label_sheet.md against the consensus and each judge (agreement and Cohen's kappa)."""
import json
import re
from pathlib import Path

sheet = Path("review/label_sheet.md").read_text()
key = json.loads(Path("review/label_sheet_key.json").read_text())
ans = {}
for num, block in re.findall(r"^## Case (\d+)\n(.*?)(?=^## Case |\Z)", sheet, re.S | re.M):
    m = re.search(r"\*\*Your answer:\*\*[ \t]*(YES|NO|Y|N)\b", block, re.I)
    if m:
        ans[num] = m.group(1).upper().startswith("Y")


def kappa(a, b):
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return ((po - pe) / (1 - pe) if pe < 1 else float("nan")), po


print(f"answered {len(ans)}/{len(key)}")
for ref in ("consensus", "gemma", "glm", "deepseek"):
    ids = [k for k in ans if key[k][ref] is not None]
    if ids:
        k, po = kappa([ans[i] for i in ids], [key[i][ref] for i in ids])
        print(f"human vs {ref:9s}: agreement {po:.2f}, kappa {k:.2f} (n={len(ids)})")
pos = [k for k in ans if key[k]["consensus"] is True]
neg = [k for k in ans if key[k]["consensus"] is False]
split = [k for k in ans if key[k]["consensus"] is None]
print(f"consensus YES confirmed: {sum(ans[k] for k in pos)}/{len(pos)}")
print(f"consensus NO confirmed: {sum(not ans[k] for k in neg)}/{len(neg)}")
print(f"splits: human YES {sum(ans[k] for k in split)}/{len(split)}, "
      f"sided with glm {sum(ans[k] == key[k]['glm'] for k in split)}/{len(split)}")
for k in sorted(ans, key=int):
    v = key[k]
    print(f"  case {k:>2}: human {ans[k]!s:5} cons {v['consensus']!s:5} gemma {v['gemma']!s:5} "
          f"glm {v['glm']!s:5} deepseek {v['deepseek']!s:5} {v['run_id']}")
