"""AUROC difference vs Peak Suspicion, same metric as group_eval_v1 (mean over layers of per-layer
AUROC), stratified bootstrap over runs. Input: a --dump-scores file."""
import json
import sys

import numpy as np


def auc(p, n):
    return ((p[:, None] > n[None]) + 0.5 * (p[:, None] == n[None])).mean()


def lm_auc(L, m, p, q):
    return np.mean([auc(np.asarray(s[m])[p], np.asarray(s[m])[q]) for s in L])


for path in sys.argv[1:]:
    for t, d in json.load(open(path)).items():
        y = np.array(d["y"])
        pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
        rng = np.random.default_rng(0)
        for m in ("set_stats", "pair_max"):
            pt = lm_auc(d["scores"], m, pi, ni) - lm_auc(d["scores"], "peak_suspicion", pi, ni)
            b = []
            for _ in range(1000):
                p, q = rng.choice(pi, len(pi)), rng.choice(ni, len(ni))
                b.append(lm_auc(d["scores"], m, p, q) - lm_auc(d["scores"], "peak_suspicion", p, q))
            lo, hi = np.percentile(b, [2.5, 97.5])
            print(f"{path.split('/')[-1]:28s} {t:16s} {m:10s} - peak {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]{' *' if lo > 0 or hi < 0 else ''}")
