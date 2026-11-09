"""AUROC of the black-box text monitor on the same tiers as the probes, plus paired
differences against probe methods where per-run probe scores were dumped."""
import glob
import json
import sys

import numpy as np

sys.path.insert(0, "group-detector")
from within_scenario import REFUSAL  # noqa: E402

M = {k: v["score"] for k, v in json.load(open("game/runs/monitor_deepseek.json")).items()}
L1 = {x["run_id"]: x["colluded"] for x in json.load(open("game/runs/implicit_labels.json"))}
L2 = {x["run_id"]: x["colluded"] for x in json.load(open("game/runs/implicit2_consensus.json"))}
ref = {}
for d in glob.glob("gen/implicit_runs/*/"):
    r = json.load(open(d + "results.json"))
    ref[json.load(open(d + "run_config.json"))["run_id"]] = any(k in x.lower() for x in r["private_messages"] for k in REFUSAL)


def auc(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    return float(((p[:, None] > q[None]) + 0.5 * (p[:, None] == q[None])).mean())


def mode(r):
    return r.rsplit("__", 1)[-1]


tiers = {
    "transfer": {r: mode(r) == "collusion" for r in M if mode(r) in ("collusion", "control")},
    "implicit": {r: mode(r) == "implicit" for r in M if mode(r) in ("implicit", "control")},
    "implicit_coord": {r: v for r, v in L1.items() if v is not None},
    "implicit2_coord": {r: v for r, v in L2.items() if v is not None},
}
rng = np.random.default_rng(0)
dumps = {"27b": {"implicit2_coord": "results/scores_implicit_{}.json", "implicit_coord": "results/scores_implicit_{}.json",
                 "transfer": "results/scores_27b_transfer_{}.json", "implicit": "results/scores_27b_transfer_{}.json"},
         "gemma": {t: "results/gemma_scores_{}.json" for t in tiers}}
out = {}
for t, lab in tiers.items():
    ids = sorted(lab)
    y = np.array([lab[r] for r in ids]); s = np.array([M[r] for r in ids], float)
    pi, ni = np.where(y)[0], np.where(~y)[0]
    b = [auc(s[rng.choice(pi, len(pi))], s[rng.choice(ni, len(ni))]) for _ in range(2000)]
    line = f"{t:16s} n={len(ids):3d} pos={y.sum():3d}  monitor AUROC {auc(s[y], s[~y]):.3f} [{np.percentile(b, 2.5):.2f}, {np.percentile(b, 97.5):.2f}]"
    if t == "implicit2_coord":
        rf = np.array([ref.get(r, False) for r in ids])
        line += f"  no-refusal {auc(s[y & ~rf], s[~y & ~rf]):.3f}"
    print(line)
    out[t] = dict(auroc=auc(s[y], s[~y]), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))])
    for model, dd in dumps.items():
        if t not in dd:
            continue
        for ph in ("discussion", "final"):
            D = json.load(open(dd[t].format(ph))).get(t)
            if D is None:
                continue
            idx = {r: i for i, r in enumerate(D["run_ids"])}
            keep = [r for r in ids if r in idx]
            yy = np.array([lab[r] for r in keep]); ms = np.array([M[r] for r in keep], float)
            p, q = np.where(yy)[0], np.where(~yy)[0]
            for meth in ("peak_suspicion", "pair_max"):
                ps = [np.asarray(sc[meth])[[idx[r] for r in keep]] for sc in D["scores"]]
                def lm(a, b_):
                    return np.mean([auc(x[a], x[b_]) for x in ps])
                diffs = []
                for _ in range(1000):
                    a_, b_ = rng.choice(p, len(p)), rng.choice(q, len(q))
                    diffs.append(auc(ms[a_], ms[b_]) - lm(a_, b_))
                d0 = auc(ms[p], ms[q]) - lm(p, q)
                lo, hi = np.percentile(diffs, [2.5, 97.5])
                print(f"    monitor - {model} {ph[:4]} {meth:15s} probe {lm(p, q):.3f}  diff {d0:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
json.dump(out, open("results/monitor_eval.json", "w"), indent=1)
