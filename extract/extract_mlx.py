"""Read last-token residual activations with MLX, in NARCBench's format.

Text per sample is the chat template over system prompt, user prompt and the stored
response (no reasoning), as in NARCBench's generation-time extraction. The hidden
state of the final token after each target layer is saved. The forward pass stops
after the deepest target layer.

Sources:
  narcbench  a NARCBench scenario tier directory (core or transfer); public turns only
  game       pricing-game JSONL from game/pricing_game.py; every priced turn

Output: <out>/activations_gen.npz (layer_<L>: float32 [n, d]) and metadata_gen.json,
resumable from a checkpoint.

Usage:
  python -I extract_mlx.py --model models/Qwen3.8-27B-4bit --source narcbench \
      --input ~/projects/open-source/narcbench/data/scenarios/qwen3_32b/core --out acts/core
  python -I extract_mlx.py ... --source game --input game/runs/main.jsonl --out acts/game
  python -I extract_mlx.py ... --bench 20
"""

import argparse
import json
import re
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

CHECKPOINT_EVERY = 100


def _strip_name_prefix(msg):
    return re.sub(r"^\[[^\]]+\]:\s*", "", msg)


def narcbench_samples(tier_dir, phases=("discussion", "final")):
    samples = []
    for run_dir in sorted(d for d in Path(tier_dir).iterdir() if d.is_dir()):
        cfg = json.loads((run_dir / "run_config.json").read_text())
        prompts = json.loads((run_dir / "agent_prompts.json").read_text())
        res = json.loads((run_dir / "results.json").read_text())
        pub = [_strip_name_prefix(m) for m in res.get("public_messages", [])]
        priv = [_strip_name_prefix(m) for m in res.get("private_messages", [])]
        finals = res.get("final_texts", {})
        pi = qi = 0
        for e in prompts:
            if e["phase"] == "discussion" and e["channel"] == "private":
                qi += 1
                continue
            if e["phase"] == "discussion" and e["channel"] == "public":
                response = pub[pi] if pi < len(pub) else ""
                pi += 1
            elif e["phase"] == "final":
                response = finals.get(e["agent_name"], "")
            else:
                continue
            if e["phase"] not in phases:
                continue
            samples.append(dict(
                meta=dict(run_id=cfg["run_id"], mode=cfg["mode"], role=e["role"], channel=e["channel"],
                          run_agents=sorted(cfg.get("agents", [])),
                          phase=e["phase"], round=e["round"], agent_name=e["agent_name"],
                          scenario_id=cfg.get("scenario_id"),
                          scenario_domain=cfg.get("scenario_domain", cfg.get("scenario_type"))),
                system=e["system_prompt"], user=e["user_prompt"], response=response))
    return samples


def game_samples(jsonl, last_only=False):
    rows = [json.loads(line) for line in Path(jsonl).read_text().splitlines()]
    last_t = max(d["t"] for d in rows)
    samples = []
    for d in rows:
        if d["explore"] or (last_only and d["t"] != last_t):
            continue
        samples.append(dict(
            meta=dict(cond=d["cond"], gid=d["gid"], t=d["t"], agent_name=d["firm"], ci=d["ci"],
                      channel="public", parse_fail=d["parse_fail"]),
            system=d["system"], user=d["user"], response=d["text"]))
    return samples


def find_text_model(model):
    lm = getattr(model, "language_model", model)
    return lm.model


def _span_means(h, spans):
    return {a: h[0, idx].astype(mx.float32).mean(0) for a, idx in spans.items()}


def last_token_states(text_model, ids, layers, spans=None):
    """Residual stream at the last token after each layer in `layers`; with `spans`
    ({sender: token indices}) also the mean over each sender's tokens."""
    h = text_model.embed_tokens(mx.array([ids]))
    fa_mask = create_attention_mask(h, None)
    ssm_mask = create_ssm_mask(h, None)
    out, sp, deepest = {}, {}, max(layers)
    for i, layer in enumerate(text_model.layers[: deepest + 1]):
        h = layer(h, mask=ssm_mask if layer.is_linear else fa_mask, cache=None)
        if i in layers:
            out[i] = h[0, -1].astype(mx.float32)
            if spans:
                sp[i] = _span_means(h, spans)
    return _finish(out, sp)


def _finish(out, sp):
    mx.eval(list(out.values()) + [v for d in sp.values() for v in d.values()])
    last = {i: np.array(v) for i, v in out.items()}
    if not sp:
        return last
    return last, {i: {a: np.array(v) for a, v in d.items()} for i, d in sp.items()}


def last_token_states_generic(text_model, ids, layers, spans=None):
    """Any mlx_lm text model: run it truncated after the deepest target layer and record
    each target layer's output at the last token."""
    full = text_model.layers
    deepest = max(layers)
    rec, sp = {}, {}
    originals = {}
    for i in layers:
        layer = full[i]
        originals[i] = layer.__class__

        def _call(self, *a, _i=i, _base=layer.__class__, **k):
            out = _base.__call__(self, *a, **k)
            h = out[0] if isinstance(out, tuple) else out
            rec[_i] = h[0, -1].astype(mx.float32)
            if spans:
                sp[_i] = _span_means(h, spans)
            return out
        layer.__class__ = type(f"Rec{i}", (layer.__class__,), {"__call__": _call})
    text_model.layers = full[: deepest + 1]
    try:
        text_model(mx.array([ids]))
        result = _finish(rec, sp)
    finally:
        text_model.layers = full
        for i, cls in originals.items():
            full[i].__class__ = cls
    return result


def read_states(text_model, ids, layers, spans=None):
    if hasattr(text_model.layers[0], "is_linear"):
        return last_token_states(text_model, ids, layers, spans)
    return last_token_states_generic(text_model, ids, layers, spans)


SPAN_TOKENS = 256


def sender_spans(hf_tok, text, agents):
    """Token indices of each agent's messages ("[Name]: ...") in text, first SPAN_TOKENS
    tokens per message. Returns (ids, {agent: mx index array}) for agents found."""
    enc = hf_tok(text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    tags = sorted((m, a) for a in agents for m in _find_all(text, f"[{a}]:"))
    bounds = [(pos + len(a) + 3, tags[k + 1][0] if k + 1 < len(tags) else len(text), a)
              for k, (pos, a) in enumerate(tags)]
    starts = np.array([o[0] for o in offs])
    spans = {}
    for cs, ce, a in bounds:
        idx = np.where((starts >= cs) & (starts < ce))[0][:SPAN_TOKENS]
        if len(idx):
            spans.setdefault(a, []).append(idx)
    return ids, {a: mx.array(np.concatenate(v)) for a, v in spans.items()}


def _find_all(text, sub):
    out, i = [], text.find(sub)
    while i >= 0:
        out.append(i)
        i = text.find(sub, i + 1)
    return out


def build_text(tokenizer, s):
    msgs = [{"role": "system", "content": s["system"]},
            {"role": "user", "content": s["user"]},
            {"role": "assistant", "content": s["response"]}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, enable_thinking=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--source", choices=["narcbench", "game"], required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--layers", type=int, nargs="+", default=[26, 27, 28, 29, 30])
    ap.add_argument("--phases", nargs="+", default=["discussion", "final"])
    ap.add_argument("--last-only", action="store_true", help="Game: only the final period.")
    ap.add_argument("--bench", type=int, default=0, help="Time this many samples and exit.")
    ap.add_argument("--receiver", action="store_true",
                    help="NarcBench: also save each reader's mean activation over each sender's messages.")
    args = ap.parse_args()

    samples = narcbench_samples(args.input, args.phases) if args.source == "narcbench" else game_samples(args.input, args.last_only)
    model, tok = load(str(args.model))
    tm = find_text_model(model)
    hf_tok = getattr(tok, "_tokenizer", tok)
    print(f"{len(samples)} samples, {len(tm.layers)} layers, reading {args.layers}", flush=True)

    if args.bench:
        n_tok, t0 = 0, time.time()
        for s in samples[: args.bench]:
            ids = hf_tok.encode(build_text(hf_tok, s), add_special_tokens=False)
            n_tok += len(ids)
            read_states(tm, ids, set(args.layers))
        dt = time.time() - t0
        print(f"bench: {args.bench} samples, {n_tok} tokens, {dt:.1f}s, {dt / args.bench:.2f}s/sample, "
              f"{n_tok / dt:.0f} tok/s; projected {dt / args.bench * len(samples) / 60:.1f} min for all")
        print("example text tail:", repr(build_text(hf_tok, samples[0])[-200:]))
        return

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt = args.out / "_checkpoint.npz"
    ckpt_meta = args.out / "_checkpoint_meta.json"
    d = tm.norm.weight.shape[0]
    keys = [f"layer_{L}" for L in args.layers]
    if args.receiver:
        keys += [f"recv_layer_{L}" for L in args.layers] + ["recv_mask"]
    acts = {k: [] for k in keys}
    meta = []
    if ckpt.exists() and ckpt_meta.exists():
        z = np.load(ckpt)
        meta = json.loads(ckpt_meta.read_text())
        acts = {k: list(z[k]) for k in keys}
        print(f"resuming at {len(meta)}")

    def save(path_npz, path_meta):
        np.savez(path_npz, **{k: np.stack(v) for k, v in acts.items()})
        path_meta.write_text(json.dumps(meta))

    t0 = time.time()
    for i in range(len(meta), len(samples)):
        s = samples[i]
        agents = s["meta"].get("run_agents") or []
        sp = {}
        if s["response"]:
            text = build_text(hf_tok, s)
            if args.receiver and agents:
                ids, spans = sender_spans(hf_tok, text, agents)
                st, sp = read_states(tm, ids, set(args.layers), spans or None) if spans else (
                    read_states(tm, ids, set(args.layers)), {})
            else:
                ids = hf_tok.encode(text, add_special_tokens=False)
                st = read_states(tm, ids, set(args.layers))
        else:
            ids, st = [], {L: np.zeros(d, np.float32) for L in args.layers}
        for L in args.layers:
            acts[f"layer_{L}"].append(st[L])
            if args.receiver:
                got = sp.get(L, {})
                acts[f"recv_layer_{L}"].append(np.stack(
                    [got.get(a, np.zeros(d, np.float32)) for a in agents[:4]]
                    + [np.zeros(d, np.float32)] * (4 - len(agents[:4]))).astype(np.float16))
        if args.receiver:
            first = sp.get(args.layers[0], {})
            acts["recv_mask"].append(np.array([a in first for a in agents[:4]] + [False] * (4 - len(agents[:4]))))
        meta.append(dict(s["meta"], n_tokens=len(ids)))
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save(ckpt, ckpt_meta)
            print(f"{i + 1}/{len(samples)}  {(time.time() - t0) / 60:.1f} min", flush=True)
    save(args.out / "activations_gen.npz", args.out / "metadata_gen.json")
    for p in (ckpt, ckpt_meta):
        p.unlink(missing_ok=True)
    print(f"done: {len(meta)} samples in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
