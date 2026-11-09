"""Within-scenario AUROC for same-setup implicit runs, plus a refusal-excluded check.

Within-scenario AUROC compares coordinated and non-coordinated runs only when they come from
the same scenario, so scenario content cannot drive the result. Scores are layer means.

Usage: python -I within_scenario.py results/scores_implicit_<phase>.json gen/implicit_runs
"""

import glob
import json
import sys

import numpy as np

REFUSAL = ("i cannot", "i can't", "i can not", "i won't", "i will not")


def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0:
        return np.nan, 0
    s = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return s / (len(pos) * len(neg)), len(pos) * len(neg)


def main():
    scores_path, runs_dir = sys.argv[1], sys.argv[2]
    ref = {}
    for d in glob.glob(f"{runs_dir}/*/"):
        r = json.load(open(d + "results.json"))
        c = json.load(open(d + "run_config.json"))
        ref[c["run_id"]] = any(any(k in x.lower() for k in REFUSAL) for x in r["private_messages"])
    D = json.load(open(scores_path))["implicit2_coord"]
    ids, y = D["run_ids"], np.array(D["y"])
    S = {m: np.mean([sc[m] for sc in D["scores"]], 0) for m in D["scores"][0]}
    scen = np.array([r.rsplit("_s", 1)[0] for r in ids])
    rf = np.array([ref.get(r, False) for r in ids])
    print(f"pos {y.sum()} neg {(1 - y).sum()}; refusal runs: neg {rf[y == 0].sum()}, pos {rf[y == 1].sum()}")
    for m, s in S.items():
        num = den = 0
        for sc in set(scen):
            k = scen == sc
            a, n = auc(s[k & (y == 1)], s[k & (y == 0)])
            if n:
                num += a * n
                den += n
        a_nr, _ = auc(s[~rf & (y == 1)], s[~rf & (y == 0)])
        a_all, _ = auc(s[y == 1], s[y == 0])
        print(f"  {m:22s} all {a_all:.3f}   within-scenario {num / den:.3f} ({den} pairs)   no-refusal {a_nr:.3f}")


if __name__ == "__main__":
    main()


def boot_diff(scores_path, runs_dir, a="pair_max", b="peak_suspicion", n=2000, seed=0):
    """95% CI of AUROC(a) - AUROC(b) on non-refusal runs, stratified bootstrap over runs."""
    ref = {}
    for d in glob.glob(f"{runs_dir}/*/"):
        r = json.load(open(d + "results.json"))
        c = json.load(open(d + "run_config.json"))
        ref[c["run_id"]] = any(any(k in x.lower() for k in REFUSAL) for x in r["private_messages"])
    D = json.load(open(scores_path))["implicit2_coord"]
    y = np.array(D["y"])
    keep = ~np.array([ref.get(r, False) for r in D["run_ids"]])
    S = {m: np.mean([sc[m] for sc in D["scores"]], 0)[keep] for m in (a, b)}
    y = y[keep]
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    rng = np.random.default_rng(seed)
    out = {m: [] for m in (a, b, "diff")}
    for _ in range(n):
        p, q = rng.choice(pos, len(pos)), rng.choice(neg, len(neg))
        va, _ = auc(S[a][p], S[a][q])
        vb, _ = auc(S[b][p], S[b][q])
        out[a].append(va); out[b].append(vb); out["diff"].append(va - vb)
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in out.items()}
