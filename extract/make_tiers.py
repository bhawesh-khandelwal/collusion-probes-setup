"""Build derived test tiers in NARCBench format from extracted activations.

implicit         Transfer implicit runs (positives) vs control runs (negatives).
                 Implicit agents become "colluder", run mode becomes "collusion".
implicit_strict  Same, keeping only implicit runs the judge says actually coordinated.
implicit_coord   Same setup on both sides: coordinated implicit runs (positives) vs implicit
                 runs the judge says did not coordinate (negatives, all agents honest).
game_matched     DM games vs dm_neutral games: identical prompt format, labels from the judge.
game_tier        Pricing-game runs labelled by game/label_runs.py. Ambiguous runs dropped.
                 The last period is the "final" phase; earlier periods are "discussion".

Usage:
  python -I make_tiers.py --transfer acts/transfer --implicit-labels game/runs/implicit_labels.json \
      --game acts/game --game-labels game/runs/main_labels.json --out acts
"""

import argparse
import json
from pathlib import Path

import numpy as np


def load(d):
    meta = json.loads((d / "metadata_gen.json").read_text())
    z = np.load(d / "activations_gen.npz")
    return meta, {k: z[k] for k in z.files}


def write(d, meta, arrays, keep):
    d.mkdir(parents=True, exist_ok=True)
    np.savez(d / "activations_gen.npz", **{k: v[keep] for k, v in arrays.items()})
    (d / "metadata_gen.json").write_text(json.dumps([meta[i] for i in keep]))
    runs = {meta[i]["run_id"]: meta[i]["mode"] for i in keep}
    print(f"{d.name}: {len(keep)} samples, {sum(v == 'collusion' for v in runs.values())} positive runs, "
          f"{sum(v == 'control' for v in runs.values())} negative runs")


def implicit_tiers(transfer_dir, labels_path, out):
    meta, arrays = load(transfer_dir)
    verdicts = {x["run_id"]: x["colluded"] for x in json.loads(labels_path.read_text())}
    coordinated = {r for r, v in verdicts.items() if v}
    new = []
    for m in meta:
        m = dict(m)
        if m["mode"] == "implicit":
            m["mode"] = "collusion"
            m["role"] = "colluder" if m["role"] == "implicit" else m["role"]
            m["coordinated"] = m["run_id"] in coordinated
        new.append(m)
    base = [i for i, m in enumerate(meta) if m["mode"] in ("implicit", "control")]
    write(out / "implicit", new, arrays, base)
    strict = [i for i in base if meta[i]["mode"] == "control" or new[i]["coordinated"]]
    write(out / "implicit_strict", new, arrays, strict)

    same = []
    for i, m in enumerate(meta):
        if m["mode"] != "implicit" or verdicts.get(m["run_id"]) is None:
            continue
        if not verdicts[m["run_id"]]:
            new[i] = dict(new[i], mode="control", role="honest")
        same.append(i)
    write(out / "implicit_coord", new, arrays, same)


def game_tier(game_dirs, labels_paths, out, name="game_tier", conds=None):
    metas, arrs = zip(*(load(d) for d in game_dirs))
    meta = [m for ms in metas for m in ms]
    arrays = {k: np.concatenate([a[k] for a in arrs]) for k in arrs[0]}
    labels = {(x["cond"], x["gid"]): x for p in labels_paths for x in json.loads(p.read_text())}
    last_t = max(m["t"] for m in meta)
    new, keep = [], []
    for i, m in enumerate(meta):
        lab = labels[(m["cond"], m["gid"])]
        sid = f"{m['cond']}_{m['gid']:03d}"
        new.append(dict(run_id=f"pricing__{sid}__{lab['label']}", mode=lab["label"],
                        role=lab["roles"][m["agent_name"]] if lab["label"] == "collusion" else "honest",
                        channel="public", phase="final" if m["t"] == last_t else "discussion",
                        round=m["t"], agent_name=m["agent_name"], scenario_id=sid,
                        scenario_domain="pricing", cond=m["cond"], ci=m["ci"]))
        if lab["label"] in ("collusion", "control") and (conds is None or m["cond"] in conds):
            keep.append(i)
    write(out / name, new, arrays, keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transfer", type=Path)
    ap.add_argument("--implicit-labels", type=Path)
    ap.add_argument("--game", type=Path, nargs="+")
    ap.add_argument("--game-labels", type=Path, nargs="+")
    ap.add_argument("--game-name", default="game_tier")
    ap.add_argument("--game-conds", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    if args.transfer:
        implicit_tiers(args.transfer, args.implicit_labels, args.out)
    if args.game:
        game_tier(args.game, args.game_labels, args.out, args.game_name, args.game_conds)


if __name__ == "__main__":
    main()
