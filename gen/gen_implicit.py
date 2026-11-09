"""Generate extra NARCBench Transfer "implicit" runs with Adaption-hosted qwen3.8-27b.

Reuses NARCBench's own generator (generation/transfer.py, MIT) unchanged except for the
model call: `call_vllm` is replaced by an Adaption Responses API call (reasoning off,
same temperature and token limit). Each scenario gets several seeds, so coordinated and
non-coordinated runs can be compared within the same scenario.

Output: <out>/<family>__<scenario>__implicit_s<k>/ with NARCBench's three JSON files; the
run_id in run_config.json carries the seed suffix so ids stay unique.

Usage: python -I gen_implicit.py <narcbench_repo> --seeds 4 --out gen/implicit_runs
"""

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adaption import Adaption
from dotenv import load_dotenv

MODEL = "qwen3.8-27b"
SEED_OFFSET = 5000


class Budget:
    def __init__(self, cap):
        self.cap, self.spent, self.lock = cap, 0.0, threading.Lock()

    def add(self, c):
        with self.lock:
            self.spent += c or 0.0
            if self.spent > self.cap:
                raise RuntimeError(f"cost cap reached: ${self.spent:.2f}")


def make_call(client, T, budget):
    def call(system, user):
        for attempt in range(4):
            try:
                r = client.responses.create(
                    model=MODEL, instructions=system, input=user, temperature=T.TEMPERATURE,
                    max_output_tokens=T.MAX_TOKENS, extra_body={"reasoning": {"effort": "none"}})
                budget.add(getattr(r.usage, "cost", 0.0))
                return (r.output_text or "").strip() or "[ERROR]"
            except RuntimeError:
                raise
            except Exception as e:  # network or rate limit
                if attempt == 3:
                    print(f"  [API ERROR] {e}", flush=True)
                    return "[ERROR]"
                time.sleep(3 * 2 ** attempt)
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("narcbench", type=Path)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--cost-cap", type=float, default=10.0)
    ap.add_argument("--types", nargs="+", default=None)
    ap.add_argument("--max-scenarios", type=int, default=None, help="Per family, for piloting.")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--env", type=Path, default=Path.home() / "projects/swarm-sprint/.env")
    args = ap.parse_args()

    sys.path.insert(0, str(args.narcbench))
    sys.path.insert(0, str(args.narcbench / "generation"))
    import transfer as T  # noqa: E402

    load_dotenv(args.env)
    budget = Budget(args.cost_cap)
    T.call_vllm = make_call(Adaption(), T, budget)

    tmp = args.out / "_tmp"
    tasks = []
    for type_name, (scenarios, runner) in T.SCENARIO_TYPES.items():
        if args.types and type_name not in args.types:
            continue
        for idx, sc in enumerate(scenarios[: args.max_scenarios]):
            for k in range(args.seeds):
                tasks.append((type_name, sc, runner, k, SEED_OFFSET + 1000 * k + idx))

    def run(task):
        type_name, sc, runner, k, seed = task
        work = tmp / f"s{k}_{type_name}_{sc['id']}"
        res = runner(sc, "implicit", seed, work)
        src = work / res["run_id"]
        new_id = f"{res['run_id']}_s{k}"
        dst = args.out / new_id
        if dst.exists():
            shutil.rmtree(dst)
        shutil.move(str(src), str(dst))
        cfg = json.loads((dst / "run_config.json").read_text())
        cfg.update(run_id=new_id, base_run_id=res["run_id"], seed_index=k)
        (dst / "run_config.json").write_text(json.dumps(cfg, indent=2))
        shutil.rmtree(work, ignore_errors=True)
        print(f"{new_id} done  spent ${budget.spent:.2f}", flush=True)
        return new_id

    args.out.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (args.out / f"{t[0]}__{t[1]['id']}__implicit_s{t[3]}").exists()]
    print(f"{len(tasks)} runs planned, {len(todo)} to generate", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(run, todo))
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"done. total cost ${budget.spent:.2f}")


if __name__ == "__main__":
    main()
