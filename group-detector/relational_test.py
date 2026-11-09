"""Is there partner signal in agent activations beyond per-agent suspicion?

For each collusion run r in a test tier, with pair score s(a, b):
  colluder gap  d_r = s(c1_r, c2_r) - mean s(c_r, c_r')     over colluders c_r' of other runs r'
  honest gap    h_r = s(h1_r, h2_r) - mean s(h_r, h_r')     over honest agents h_r' of other runs r'
  DiD_r = d_r - h_r
Cross-run "Frankenstein" colluder pairs look like real colluders one agent at a time, so
per-agent methods cannot tell them from real partners. The honest gap removes what any two
agents of the same conversation share (topic, transcript). DiD > 0 means real partners are
more related than same-conversation non-partners, beyond what each agent shows alone.
Cross-run partners come from all other runs ("any") or the same scenario family ("family").

Uncertainty: each bootstrap draw resamples runs and recomputes the cross-run partner pools
from the drawn runs only, so shared agents are not treated as independent.

Pair scores:
  cos_std       cosine of Core-standardised activations (training-free)
  cos_noprobe   same, with the deception-probe direction projected out (training-free)
  learned       logistic regression on [z_a * z_b, |z_a - z_b|] in a 32-dim PCA space (probe
                direction removed), trained on all of Core: positives = real colluder partners;
                negatives = cross-run colluder pairs, same-run honest pairs, same-run
                colluder-honest pairs. Fixed settings, no tuning. Reported on test tiers only,
                since on Core it would be scored on its own training pairs.

Usage: python -I relational_test.py <narcbench_repo> --core DIR --tier NAME=DIR ... --out FILE
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from group_eval_v1 import load_filtered, load_narcbench, probe_rows, probe_subspace  # noqa: E402

N_BOOT = 2000


def agents_by_role(g, scaler):
    """Collusion runs with two colluders and two honest agents: (C [n,2,d], H [n,2,d], families)."""
    C, H, fam = [], [], []
    for r in sorted(g):
        gr = g[r]
        if gr["mode"] != "collusion":
            continue
        names = sorted(gr["agents"])
        X = scaler.transform(np.stack([gr["agents"][a] for a in names]))
        roles = [gr["roles"][a] for a in names]
        c = [X[i] for i, ro in enumerate(roles) if ro == "colluder"]
        h = [X[i] for i, ro in enumerate(roles) if ro != "colluder"]
        if len(c) == 2 and len(h) == 2:
            C.append(c)
            H.append(h)
            fam.append(gr["scenario"] if gr["scenario"] != "deception" else gr["domain"])
    return np.array(C), np.array(H), np.array(fam)


def _unit(X):
    return X / (np.linalg.norm(X, axis=-1, keepdims=True) + 1e-8)


def cos_matrix(A, B):
    return _unit(A) @ _unit(B).T


class Learned:
    def __init__(self, w, k=32, C=0.01):
        self.w, self.k, self.C = w, k, C

    def _z(self, X):
        X = X - np.outer(X @ self.w, self.w)
        return self.pca.transform(X) / self.sd

    def _feats(self, za, zb):
        return np.concatenate([za * zb, np.abs(za - zb)], -1)

    def fit(self, C, H):
        d = C.shape[-1]
        allv = np.concatenate([C.reshape(-1, d), H.reshape(-1, d)])
        stripped = allv - np.outer(allv @ self.w, self.w)
        self.pca = PCA(self.k, random_state=0).fit(stripped)
        self.sd = self.pca.transform(stripped).std(0) + 1e-6
        n = len(C)
        zc, zh = self._z(C.reshape(-1, d)).reshape(n, 2, -1), self._z(H.reshape(-1, d)).reshape(n, 2, -1)
        F, y = [], []
        for r in range(n):
            F.append(self._feats(zc[r, 0], zc[r, 1])); y.append(1)
            F.append(self._feats(zh[r, 0], zh[r, 1])); y.append(0)
            for i in range(2):
                for j in range(2):
                    F.append(self._feats(zc[r, i], zh[r, j])); y.append(0)
        for r, r2 in itertools.combinations(range(n), 2):
            for i in range(2):
                for j in range(2):
                    F.append(self._feats(zc[r, i], zc[r2, j])); y.append(0)
        F = np.stack(F)
        self.fs = StandardScaler().fit(F)
        self.clf = LogisticRegression(C=self.C, max_iter=3000, class_weight="balanced").fit(
            self.fs.transform(F), np.array(y))
        return self

    def matrix(self, A, B):
        F = self._feats(self._z(A)[:, None, :], self._z(B)[None, :, :])
        n, m, f = F.shape
        return self.clf.decision_function(self.fs.transform(F.reshape(-1, f))).reshape(n, m)


def pair_stats(C, H, mat):
    """Real-pair scores per run and run-by-run mean cross scores, for colluders and honest agents."""
    n, d = len(C), C.shape[-1]
    real_c = np.array([mat(C[r, :1], C[r, 1:])[0, 0] for r in range(n)])
    real_h = np.array([mat(H[r, :1], H[r, 1:])[0, 0] for r in range(n)])
    Mc = mat(C.reshape(-1, d), C.reshape(-1, d)).reshape(n, 2, n, 2).mean((1, 3))
    Mh = mat(H.reshape(-1, d), H.reshape(-1, d)).reshape(n, 2, n, 2).mean((1, 3))
    return real_c, real_h, Mc, Mh


def did(real_c, real_h, Mc, Mh, fam, same_family, counts):
    """DiD over runs weighted by `counts` (bootstrap multiplicities); partners from drawn runs only."""
    n = len(real_c)
    allow = ~np.eye(n, dtype=bool)
    if same_family:
        allow &= fam[:, None] == fam[None, :]
    W = allow * counts[None, :]
    denom = W.sum(1)
    ok = (denom > 0) & (counts > 0)
    if not ok.any():
        return np.nan, np.nan, np.nan, 0
    fc = (W * Mc).sum(1)[ok] / denom[ok]
    fh = (W * Mh).sum(1)[ok] / denom[ok]
    wt = counts[ok]
    d = real_c[ok] - fc
    h = real_h[ok] - fh
    return (wt * d).sum() / wt.sum(), (wt * h).sum() / wt.sum(), (wt * (d - h)).sum() / wt.sum(), int(ok.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("narcbench", type=Path)
    ap.add_argument("--core", type=Path, default=None)
    ap.add_argument("--core-runs-from", type=Path, default=None)
    ap.add_argument("--tier", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--layers", type=int, nargs="+", default=[26, 27, 28, 29, 30])
    ap.add_argument("--core-phase", choices=["discussion", "final"], default="discussion",
                    help="Read position for Core agent vectors (the probe always trains on discussion).")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    R = load_narcbench(args.narcbench)
    run_ids = {m["run_id"] for m in R.load_data(args.core_runs_from)[0]} if args.core_runs_from else None
    core_meta, core_npz = load_filtered(R, args.core or R.CORE_DIR, run_ids)
    tier_dirs = dict(x.split("=", 1) for x in args.tier) or {"transfer": R.TRANSFER_DIR}
    tiers = {t: load_filtered(R, d) for t, d in tier_dirs.items()}

    acc, fams = {}, {}  # (tier, metric) -> list over layers of (real_c, real_h, Mc, Mh)
    for layer in args.layers:
        print(f"layer {layer}", flush=True)
        acts = R.get_layer(core_npz, len(core_meta), layer)
        rows, ys, doms = probe_rows(R, core_meta, "discussion")
        core_g = R.build_groups(core_meta, core_npz, layer, phase=args.core_phase,
                                avg_rounds=args.core_phase == "discussion")
        scaler, w, _ = probe_subspace(R, acts, rows, ys, doms, {g["domain"] for g in core_g.values()}, 1)
        cC, cH, cfam = agents_by_role(core_g, scaler)
        learned = Learned(w).fit(cC, cH)

        def strip(X, w=w):
            return X - np.outer(X @ w, w)

        metrics = {"cos_std": cos_matrix, "cos_noprobe": lambda A, B: cos_matrix(strip(A), strip(B)),
                   "learned": learned.matrix}
        data = {"core": (cC, cH, cfam)}
        for t, (m, z) in tiers.items():
            data[t] = agents_by_role(R.build_groups(m, z, layer, phase="final", avg_rounds=False), scaler)
        for t, (C, H, fam) in data.items():
            fams[t] = fam
            for mname, mat in metrics.items():
                if t == "core" and mname == "learned":
                    continue
                acc.setdefault((t, mname), []).append(pair_stats(C, H, mat))

    out = {}
    rng = np.random.default_rng(0)
    print(f"\n{'tier':16s}{'metric':13s}{'scope':8s}{'runs':>5s}{'colluder gap':>14s}{'honest gap':>12s}"
          f"{'DiD':>9s}  95% CI")
    for (t, mname), per_layer in acc.items():
        real_c, real_h, Mc, Mh = (np.mean([p[i] for p in per_layer], 0) for i in range(4))
        n = len(real_c)
        draws = [rng.multinomial(n, np.ones(n) / n) for _ in range(N_BOOT)]
        for scope in ("any", "family"):
            sf = scope == "family"
            dc, dh, dd, k = did(real_c, real_h, Mc, Mh, fams[t], sf, np.ones(n))
            if k == 0:
                continue
            boots = np.array([did(real_c, real_h, Mc, Mh, fams[t], sf, c)[2] for c in draws])
            ci = [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]
            out[f"{t}|{mname}|{scope}"] = dict(n_runs=k, colluder_gap=float(dc), honest_gap=float(dh),
                                                did=float(dd), ci=ci)
            print(f"{t:16s}{mname:13s}{scope:8s}{k:>5d}{dc:>14.4f}{dh:>12.4f}{dd:>9.4f}  "
                  f"[{ci[0]:+.4f}, {ci[1]:+.4f}]{'  *' if ci[0] > 0 else ''}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
