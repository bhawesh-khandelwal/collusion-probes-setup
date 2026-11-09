"""In-domain detectors on same-setup implicit runs: is there a coordination signal at all?

Train and test on the implicit2 runs (all have the same setup: two agents share an incentive
and a private channel). Positive run = both judges say the pair coordinated; negative = both
say it did not. Agent label = incentive-pair agent in a coordinated run.

Methods (fixed settings, fit inside each training fold on standardised 32-dim PCA features):
  peak            per-agent logistic regression, run score = max over the 4 agents
  pair_known      same per-agent model, mean over the two incentive-pair agents (oracle roles)
  set_stats       run-level regression on [mean, max, min, std] over agents
  pair_max        pair regression on [a*b, |a-b|, a+b], run score = max over the 6 pairs
  family_prior    base rate of the run's scenario family in the training folds
Scores are averaged over layers. Splits:
  scenario  GroupKFold by scenario (6 folds, 5 shuffles, out-of-fold scores averaged)
  family    leave one scenario family out (no family base rate can be learned)

Usage: python -I indomain_cv.py --acts DIR --labels FILE --runs DIR --layers ... --out FILE
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REFUSAL = ("i cannot", "i can't", "i can not", "i won't", "i will not")
K, C = 32, 0.1
PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def load(acts, labels, layers):
    meta = json.loads((acts / "metadata_gen.json").read_text())
    z = np.load(acts / "activations_gen.npz")
    lab = {x["run_id"]: x["colluded"] for x in json.loads(Path(labels).read_text())}
    runs = {}
    for i, m in enumerate(meta):
        if lab.get(m["run_id"]) is None:
            continue
        g = runs.setdefault(m["run_id"], dict(idx={}, role={}, fam=m["scenario_domain"],
                                               scen=m["run_id"].rsplit("_s", 1)[0], y=lab[m["run_id"]]))
        g["idx"][m["agent_name"]] = i
        g["role"][m["agent_name"]] = m["role"]
    ids = sorted(r for r, g in runs.items() if len(g["idx"]) == 4)
    X = {L: np.stack([z[f"layer_{L}"][[runs[r]["idx"][a] for a in sorted(runs[r]["idx"])]] for r in ids])
         for L in layers}
    pair = np.stack([[runs[r]["role"][a] == "implicit" for a in sorted(runs[r]["idx"])] for r in ids])
    meta_r = dict(y=np.array([runs[r]["y"] for r in ids], bool), fam=np.array([runs[r]["fam"] for r in ids]),
                  scen=np.array([runs[r]["scen"] for r in ids]), pair=pair)
    return ids, X, meta_r


def fit_predict(Xtr, Xte, ytr, pair_tr, pair_te, fam_tr, fam_te):
    n, _, d = Xtr.shape
    sc = StandardScaler().fit(Xtr.reshape(-1, d))
    pca = PCA(K, random_state=0).fit(sc.transform(Xtr.reshape(-1, d)))
    sd = pca.transform(sc.transform(Xtr.reshape(-1, d))).std(0) + 1e-6
    Ztr = (pca.transform(sc.transform(Xtr.reshape(-1, d))) / sd).reshape(n, 4, K)
    Zte = (pca.transform(sc.transform(Xte.reshape(-1, d))) / sd).reshape(len(Xte), 4, K)
    out = {}

    ya = (pair_tr & ytr[:, None]).reshape(-1)
    agent = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ztr.reshape(-1, K), ya)
    s = agent.decision_function(Zte.reshape(-1, K)).reshape(len(Zte), 4)
    out["peak"] = s.max(1)
    out["pair_known"] = np.array([s[r][pair_te[r]].mean() for r in range(len(s))])

    def stats(Z):
        return np.concatenate([Z.mean(1), Z.max(1), Z.min(1), Z.std(1)], -1)
    rl = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(stats(Ztr), ytr)
    out["set_stats"] = rl.decision_function(stats(Zte))

    def pf(Z):
        return np.stack([np.concatenate([Z[:, i] * Z[:, j], np.abs(Z[:, i] - Z[:, j]), Z[:, i] + Z[:, j]], -1)
                         for i, j in PAIRS], 1)
    Ptr = pf(Ztr)
    yp = np.stack([pair_tr[:, i] & pair_tr[:, j] & ytr for i, j in PAIRS], 1)
    pl = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Ptr.reshape(-1, 3 * K), yp.reshape(-1))
    out["pair_max"] = pl.decision_function(pf(Zte).reshape(-1, 3 * K)).reshape(len(Zte), len(PAIRS)).max(1)

    base = ytr.mean()
    out["family_prior"] = np.array([ytr[fam_tr == f].mean() if (fam_tr == f).any() else base for f in fam_te])
    return out


def oof(X, M, layers, splits):
    n = len(M["y"])
    acc, cnt = {}, np.zeros(n)
    for tr, te in splits:
        per_layer = [fit_predict(X[L][tr], X[L][te], M["y"][tr], M["pair"][tr], M["pair"][te],
                                 M["fam"][tr], M["fam"][te]) for L in layers]
        for m in per_layer[0]:
            v = np.mean([p[m] for p in per_layer], 0)
            if m != "family_prior":  # put fold scores on a common scale before pooling
                v = (v - v.mean()) / (v.std() + 1e-9)
            acc.setdefault(m, np.zeros(n))[te] += v
        cnt[te] += 1
    return {m: v / cnt for m, v in acc.items()}


def auc(p, q):
    p, q = np.asarray(p), np.asarray(q)
    if not len(p) or not len(q):
        return np.nan
    return float(((p[:, None] > q[None]) + 0.5 * (p[:, None] == q[None])).mean())


def within(s, y, grp):
    num = den = 0
    for g in set(grp):
        k = grp == g
        p, q = s[k & y], s[k & ~y]
        if len(p) and len(q):
            num += auc(p, q) * len(p) * len(q)
            den += len(p) * len(q)
    return num / den if den else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--runs", required=True, help="Run transcripts, for the refusal flag.")
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids, X, M = load(args.acts, args.labels, args.layers)
    ref = {}
    for d in glob.glob(f"{args.runs}/*/"):
        r = json.load(open(d + "results.json"))
        ref[json.load(open(d + "run_config.json"))["run_id"]] = any(
            k in x.lower() for x in r["private_messages"] for k in REFUSAL)
    rf = np.array([ref.get(r, False) for r in ids])
    y = M["y"]
    print(f"{len(ids)} runs, {y.sum()} coordinated; refusals among negatives {rf[~y].sum()}", flush=True)

    sc_splits = []
    for seed in range(5):
        groups = np.unique(M["scen"])
        perm = dict(zip(groups, np.random.default_rng(seed).permutation(len(groups))))
        g = np.array([perm[s] for s in M["scen"]])
        sc_splits += list(GroupKFold(6).split(np.zeros(len(y)), y, g))
    fam_splits = [(np.where(M["fam"] != f)[0], np.where(M["fam"] == f)[0]) for f in np.unique(M["fam"])]

    rng = np.random.default_rng(0)
    pi, ni = np.where(y)[0], np.where(~y)[0]
    out = {}
    for split_name, splits in (("scenario", sc_splits), ("family", fam_splits)):
        S = oof(X, M, args.layers, splits)
        print(f"\nsplit={split_name}\n{'method':14s}{'AUROC':>8s}  95% CI        {'within-family':>14s}{'no-refusal':>12s}"
              f"   minus peak (95% CI)")
        for m, s in S.items():
            a = auc(s[y], s[~y])
            boots = [(auc(s[p], s[q]), auc(s[p], s[q]) - auc(S["peak"][p], S["peak"][q]))
                     for p, q in ((rng.choice(pi, len(pi)), rng.choice(ni, len(ni))) for _ in range(1000))]
            b = np.array(boots)
            ci, dci = np.percentile(b[:, 0], [2.5, 97.5]), np.percentile(b[:, 1], [2.5, 97.5])
            wf, nr = within(s, y, M["fam"]), auc(s[~rf & y], s[~rf & ~y])
            d = a - auc(S["peak"][y], S["peak"][~y])
            print(f"{m:14s}{a:8.3f}  [{ci[0]:.2f}, {ci[1]:.2f}]  {wf:12.3f}{nr:12.3f}   "
                  f"{d:+.3f} [{dci[0]:+.3f}, {dci[1]:+.3f}]", flush=True)
            out[f"{split_name}|{m}"] = dict(auroc=a, ci=ci.tolist(), within_family=wf, no_refusal=nr,
                                            minus_peak=d, minus_peak_ci=dci.tolist())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
