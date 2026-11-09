"""Same-setup runs read at the private-channel turns: does coordination show up there?

Each implicit2 run has two private turns, one per incentive-pair member; the second speaker's
prompt already holds the first message, so its read covers the whole conversation.
Positive run = both judges say the pair coordinated; negative = both say it did not.

Methods (scores averaged over layers):
  core_probe_*   zero-shot: NARCBench deception probe trained on Core discussion turns
                 (colluder vs honest), applied to the two private reads; max, mean, second speaker
  act_*          in-domain logistic regression on standardised 32-dim PCA features of the
                 private reads: second speaker, pair mean, pair interaction [a*b, |a-b|, a+b]
  text_tfidf     in-domain logistic regression on TF-IDF of the private conversation text
  family_prior   base rate of the run's scenario family in the training folds
In-domain splits: scenario held out (GroupKFold 6, 5 shuffles) and family held out.

Usage: python -I private_read.py --acts DIR --core DIR --labels FILE --runs DIR --layers ... --out FILE
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

REFUSAL = ("i cannot", "i can't", "i can not", "i won't", "i will not")
K, C = 32, 0.1


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


def load(acts, labels, runs_dir, layers):
    meta = json.loads((acts / "metadata_gen.json").read_text())
    z = np.load(acts / "activations_gen.npz")
    lab = {x["run_id"]: x["colluded"] for x in json.loads(Path(labels).read_text())}
    rows = {}
    for i, m in enumerate(meta):
        if lab.get(m["run_id"]) is not None:
            rows.setdefault(m["run_id"], []).append(i)
    ids = sorted(r for r, v in rows.items() if len(v) == 2)
    X = {L: np.stack([z[f"layer_{L}"][rows[r]] for r in ids]) for L in layers}
    fam = np.array([meta[rows[r][0]]["scenario_domain"] for r in ids])
    text, ref = [], []
    for r in ids:
        msgs = json.loads((Path(runs_dir) / r / "results.json").read_text())["private_messages"]
        text.append("\n\n".join(msgs))
        ref.append(any(k in m.lower() for m in msgs for k in REFUSAL))
    return ids, X, dict(y=np.array([lab[r] for r in ids], bool), fam=fam,
                        scen=np.array([r.rsplit("_s", 1)[0] for r in ids]), text=text, ref=np.array(ref))


def core_probe(core, layers):
    """Per layer: (scaler, direction) of the NARCBench-style deception probe on Core discussion turns."""
    meta = json.loads((core / "metadata_gen.json").read_text())
    z = np.load(core / "activations_gen.npz")
    rows = [i for i, m in enumerate(meta) if m["mode"] == "collusion" and m["channel"] == "public"
            and m["phase"] == "discussion" and m["role"] in ("colluder", "honest")]
    y = np.array([meta[i]["role"] == "colluder" for i in rows])
    out = {}
    for L in layers:
        Xc = z[f"layer_{L}"][rows]
        sc = StandardScaler().fit(Xc)
        clf = LogisticRegression(C=25.0, max_iter=8, penalty="l2", solver="liblinear", random_state=0)
        out[L] = (sc, clf.fit(sc.transform(Xc), y))
    return out


def fit_predict(Xtr, Xte, ytr, fam_tr, fam_te):
    n, _, d = Xtr.shape
    sc = StandardScaler().fit(Xtr.reshape(-1, d))
    pca = PCA(K, random_state=0).fit(sc.transform(Xtr.reshape(-1, d)))
    sd = pca.transform(sc.transform(Xtr.reshape(-1, d))).std(0) + 1e-6
    Ztr = (pca.transform(sc.transform(Xtr.reshape(-1, d))) / sd).reshape(n, 2, K)
    Zte = (pca.transform(sc.transform(Xte.reshape(-1, d))) / sd).reshape(len(Xte), 2, K)
    feats = {"act_second": lambda Z: Z[:, 1], "act_mean": lambda Z: Z.mean(1),
             "act_pair": lambda Z: np.concatenate([Z[:, 0] * Z[:, 1], np.abs(Z[:, 0] - Z[:, 1]), Z[:, 0] + Z[:, 1]], -1)}
    out = {}
    for name, f in feats.items():
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(f(Ztr), ytr)
        out[name] = clf.decision_function(f(Zte))
    return out


def oof(X, M, layers, splits):
    n = len(M["y"])
    acc, cnt = {}, np.zeros(n)
    text = np.array(M["text"], dtype=object)
    for tr, te in splits:
        per_layer = [fit_predict(X[L][tr], X[L][te], M["y"][tr], M["fam"][tr], M["fam"][te]) for L in layers]
        fold = {m: np.mean([p[m] for p in per_layer], 0) for m in per_layer[0]}
        vec = TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True).fit(text[tr])
        tclf = LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced").fit(vec.transform(text[tr]), M["y"][tr])
        fold["text_tfidf"] = tclf.decision_function(vec.transform(text[te]))
        for m, v in fold.items():
            v = (v - v.mean()) / (v.std() + 1e-9)
            acc.setdefault(m, np.zeros(n))[te] += v
        base = M["y"][tr].mean()
        fp = np.array([M["y"][tr][M["fam"][tr] == f].mean() if (M["fam"][tr] == f).any() else base
                       for f in M["fam"][te]])
        acc.setdefault("family_prior", np.zeros(n))[te] += fp
        cnt[te] += 1
    return {m: v / cnt for m, v in acc.items()}


def report(S, M, rng, out, prefix):
    y, rf = M["y"], M["ref"]
    pi, ni = np.where(y)[0], np.where(~y)[0]
    draws = [(rng.choice(pi, len(pi)), rng.choice(ni, len(ni))) for _ in range(1000)]
    print(f"\n{prefix}\n{'method':16s}{'AUROC':>8s}  95% CI        within-family [95% CI]  no-refusal")
    for m, s in S.items():
        a = auc(s[y], s[~y])
        ci = np.percentile([auc(s[p], s[q]) for p, q in draws], [2.5, 97.5])
        wf, nr = within(s, y, M["fam"]), auc(s[~rf & y], s[~rf & ~y])
        wci = np.nanpercentile([within(s[np.r_[p, q]], y[np.r_[p, q]], M["fam"][np.r_[p, q]]) for p, q in draws],
                               [2.5, 97.5])
        print(f"{m:16s}{a:8.3f}  [{ci[0]:.2f}, {ci[1]:.2f}]  {wf:8.3f} [{wci[0]:.2f}, {wci[1]:.2f}]{nr:10.3f}",
              flush=True)
        out[f"{prefix}|{m}"] = dict(auroc=a, ci=ci.tolist(), within_family=wf, within_family_ci=wci.tolist(),
                                    no_refusal=nr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts", type=Path, required=True)
    ap.add_argument("--core", type=Path, required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--layers", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ids, X, M = load(args.acts, args.labels, args.runs, args.layers)
    y = M["y"]
    print(f"{len(ids)} runs, {y.sum()} coordinated; refusals among negatives {M['ref'][~y].sum()}", flush=True)
    rng = np.random.default_rng(0)
    out = {}

    probes = core_probe(args.core, args.layers)
    per = np.mean([probes[L][1].decision_function(probes[L][0].transform(X[L].reshape(-1, X[L].shape[-1])))
                   .reshape(len(ids), 2) for L in args.layers], 0)
    report({"core_probe_max": per.max(1), "core_probe_mean": per.mean(1), "core_probe_second": per[:, 1]},
           M, rng, out, "zero_shot")

    sc_splits = []
    for seed in range(5):
        groups = np.unique(M["scen"])
        perm = dict(zip(groups, np.random.default_rng(seed).permutation(len(groups))))
        g = np.array([perm[s] for s in M["scen"]])
        sc_splits += list(GroupKFold(6).split(np.zeros(len(y)), y, g))
    fam_splits = [(np.where(M["fam"] != f)[0], np.where(M["fam"] == f)[0]) for f in np.unique(M["fam"])]
    for name, splits in (("scenario", sc_splits), ("family", fam_splits)):
        report(oof(X, M, args.layers, splits), M, rng, out, name)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
